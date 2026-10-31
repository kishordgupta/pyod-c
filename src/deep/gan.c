/* Adversarial detectors: SO-GAAL / MO-GAAL (pyod.models.so_gaal, mo_gaal),
 * ALAD (pyod.models.alad) and AnoGAN (pyod.models.anogan). BCE is computed
 * from logits (softplus(z) - y z), which equals PyTorch's BCELoss on sigmoid
 * outputs but is numerically stable. */
#include "deep/common.h"

/* ---------------- GAAL ---------------- */

typedef struct {
    nn_linear l1, l2;
} gaal_gen;

typedef struct {
    pyod_gaal_params p;
    int multi;
    nn_params dps;
    nn_linear d1, d2; /* discriminator */
    pyod_rng r;
} gaal_impl;

static ad_tensor *gen_fwd(const gaal_gen *g, ad_tensor *z) {
    return ad_relu(nn_linear_fwd(&g->l2, ad_relu(nn_linear_fwd(&g->l1, z))));
}

static ad_tensor *disc_logits(const gaal_impl *m, ad_tensor *x) { return nn_linear_fwd(&m->d2, ad_relu(nn_linear_fwd(&m->d1, x))); }

/* Scores are 1 - D(x) (PyOD returns D(x); see README). */
static pyod_status gaal_scores(const gaal_impl *m, const pyod_matrix *x, double *out) {
    ad_tensor *z = disc_logits(m, ad_from_rows(x, NULL, x->rows));
    for (size_t i = 0; i < x->rows; i++) out[i] = 1.0 - 1.0 / (1.0 + exp(-z->data[i]));
    ad_tape_clear();
    return PYOD_OK;
}

static ad_tensor *filled(size_t n, double v) {
    ad_tensor *t = ad_const(2, AD_SHAPE(n, 1), NULL);
    for (size_t i = 0; i < n; i++) t->data[i] = v;
    return t;
}

static pyod_status gaal_fit(pyod_detector *dt, const pyod_matrix *x) {
    gaal_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t n = x->rows, d = x->cols, k = m->multi ? m->p.k : 1;
    if (m->p.stop_epochs == 0 || k == 0) return PYOD_ERR_PARAM;
    nn_params_free(&m->dps);
    rng_seed(&m->r, m->p.random_state);
    size_t h = (size_t)ceil(sqrt((double)n));
    nn_linear_init(&m->d1, &m->dps, d, h, 1, NN_INIT_KAIMING_NORMAL, sqrt(2.0), &m->r);
    nn_linear_init(&m->d2, &m->dps, h, 1, 1, NN_INIT_KAIMING_NORMAL, 1.0, &m->r);
    nn_params gps = {0};
    gaal_gen *gens = NEW(gaal_gen, k);
    ad_optimizer od, *og = NEW(ad_optimizer, k);
    size_t *perm = NEW(size_t, n), *start = NEW(size_t, k), *end = NEW(size_t, k);
    double *pred = NEW(double, n), *target = NEW(double, k);
    pyod_status st = gens && og && perm && start && end && pred && target && m->d1.w && m->d2.w ? PYOD_OK : PYOD_ERR_NOMEM;
    for (size_t i = 0; st == PYOD_OK && i < k; i++) {
        size_t before = gps.n;
        nn_linear_init(&gens[i].l1, &gps, d, d, 1, NN_INIT_TORCH, 1.0, &m->r);
        nn_linear_init(&gens[i].l2, &gps, d, d, 1, NN_INIT_TORCH, 1.0, &m->r);
        if (!gens[i].l1.w || !gens[i].l2.w) {
            st = PYOD_ERR_NOMEM;
            break;
        }
        for (size_t a = 0; a < d; a++) /* identity initialisation */
            for (size_t b = 0; b < d; b++) gens[i].l1.w->data[a * d + b] = gens[i].l2.w->data[a * d + b] = (double)(a == b);
        st = ad_opt_init(&og[i], AD_SGD, gps.p + before, gps.n - before, m->p.lr_g, 0.0, m->p.momentum);
    }
    if (st == PYOD_OK) st = ad_opt_init(&od, AD_SGD, m->dps.p, m->dps.n, m->p.lr_d, 0.0, m->p.momentum);
    size_t bs = n < 500 ? n : 500, inited = st == PYOD_OK;
    int stop = 0;
    for (size_t ep = 0; st == PYOD_OK && ep < m->p.stop_epochs * 3; ep++) {
        size_t nb = nn_batches(n, bs, 0, &m->r, perm);
        for (size_t b = 0; b < nb; b++) {
            size_t len = nn_batch_len(n, bs, b), block = (1 + k) * k / 2, nfake = 0;
            ad_tensor *real = ad_from_rows(x, perm + b * bs, len);
            ad_tensor *noise = ad_rand(2, AD_SHAPE(len, d), &m->r);
            /* Noise split among sub-generators in proportions k, k-1, ..., 1. */
            for (size_t i = 0; i < k; i++) {
                if (!m->multi) {
                    start[i] = 0, end[i] = len;
                } else {
                    start[i] = (k + (k - i + 1)) * i / 2 * (len / block);
                    end[i] = i == k - 1 ? len : (k + (k - i)) * (i + 1) / 2 * (len / block);
                    if (start[i] > len) start[i] = len;
                    if (end[i] > len) end[i] = len;
                }
                nfake += end[i] - start[i];
            }
            ad_tensor *loss_d;
            if (m->multi) {
                ad_tensor *all = real;
                for (size_t i = 0; i < k; i++)
                    if (end[i] > start[i]) all = ad_concat(all, ad_detach(gen_fwd(&gens[i], ad_narrow(noise, 0, start[i], end[i] - start[i]))), 0);
                ad_tensor *labels = ad_concat(filled(len, 1.0), filled(nfake, 0.0), 0);
                loss_d = ad_bce_logits(disc_logits(m, all), labels);
            } else {
                ad_tensor *fake = ad_detach(gen_fwd(&gens[0], noise));
                loss_d = ad_add(ad_bce_logits(disc_logits(m, real), filled(len, 1.0)), ad_bce_logits(disc_logits(m, fake), filled(len, 0.0)));
            }
            ad_zero_grad(m->dps.p, m->dps.n);
            ad_backward(loss_d);
            ad_opt_step(&od);
            if (!stop) {
                /* SO-GAAL pushes generated data toward "real"; MO-GAAL sub-generator i
                 * targets the i/k quantile of the discriminator's scores on the data. */
                if (m->multi) {
                    ad_tensor *zall = disc_logits(m, ad_from_rows(x, NULL, n));
                    for (size_t i = 0; i < n; i++) pred[i] = 1.0 / (1.0 + exp(-zall->data[i]));
                    for (size_t i = 0; i < k; i++) target[i] = vec_percentile(pred, n, (double)i / (double)k * 100.0);
                } else {
                    target[0] = 1.0;
                }
                for (size_t i = 0; i < k; i++) {
                    size_t len_i = end[i] - start[i];
                    if (!len_i) continue;
                    ad_tensor *out = disc_logits(m, gen_fwd(&gens[i], ad_narrow(noise, 0, start[i], len_i)));
                    ad_zero_grad(gps.p, gps.n);
                    ad_backward(ad_bce_logits(out, filled(len_i, target[i])));
                    ad_opt_step(&og[i]);
                }
            }
            ad_tape_clear();
        }
        if (ep + 1 > m->p.stop_epochs) stop = 1;
    }
    if (inited) {
        ad_opt_free(&od);
        for (size_t i = 0; i < k; i++) ad_opt_free(&og[i]);
    }
    ad_tape_clear();
    nn_params_free(&gps);
    free(gens), free(og), free(perm), free(start), free(end), free(pred), free(target);
    CHECK(st);
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    gaal_scores(m, x, s);
    return fit_state_contamination(dt, s, n, m->p.contamination);
}

static pyod_status gaal_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return gaal_scores(d->impl, x, out); }

static void gaal_destroy(void *p) {
    gaal_impl *m = p;
    nn_params_free(&m->dps);
    free(m);
}

static const pyod_vtable SO_GAAL_VT = {"SO_GAAL", gaal_fit, NULL, gaal_score, gaal_destroy};
static const pyod_vtable MO_GAAL_VT = {"MO_GAAL", gaal_fit, NULL, gaal_score, gaal_destroy};

pyod_gaal_params pyod_gaal_defaults(void) { return (pyod_gaal_params){20, 10, 0.01, 1e-4, 0.9, 0.1, 42}; }

static pyod_detector *gaal_create(const pyod_gaal_params *p, int multi) {
    gaal_impl *m = NEW(gaal_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_gaal_defaults();
    m->multi = multi;
    return detector_new(multi ? &MO_GAAL_VT : &SO_GAAL_VT, m);
}

pyod_detector *pyod_so_gaal_create(const pyod_gaal_params *p) { return gaal_create(p, 0); }
pyod_detector *pyod_mo_gaal_create(const pyod_gaal_params *p) { return gaal_create(p, 1); }

/* ---------------- ALAD ---------------- */

typedef struct {
    pyod_alad_params p;
    nn_params gps, dps;
    nn_dense enc, dec, dxx, dzz, dxz;
    std_scaler scaler;
    int scaled;
    pyod_rng r;
} alad_impl;

static size_t layers_with(size_t in, const size_t *hidden, size_t out, size_t *buf) {
    size_t n = 0, h = layer_count(hidden);
    buf[n++] = in;
    for (size_t i = 0; i < h; i++) buf[n++] = hidden[i];
    buf[n++] = out;
    return n;
}

static void alad_reset(alad_impl *m) {
    nn_dense_free(&m->enc), nn_dense_free(&m->dec), nn_dense_free(&m->dxx), nn_dense_free(&m->dzz), nn_dense_free(&m->dxz);
    nn_params_free(&m->gps), nn_params_free(&m->dps);
    if (m->scaled) scaler_free(&m->scaler);
    m->scaled = 0;
}

static pyod_status alad_scores(alad_impl *m, const pyod_matrix *x, double *out) {
    pyod_matrix xn;
    CHECK(mat_copy(x, &xn));
    if (m->scaled) scaler_transform(&m->scaler, x, xn.data);
    ad_tensor *xt = ad_from_rows(&xn, NULL, xn.rows);
    ad_tensor *rec = nn_dense_fwd(&m->dec, nn_dense_fwd(&m->enc, xt, 0, &m->r), 0, &m->r);
    ad_tensor *t = ad_sigmoid(nn_dense_fwd(&m->dxx, ad_concat(xt, xt, 1), 0, &m->r));
    ad_tensor *f = ad_sigmoid(nn_dense_fwd(&m->dxx, ad_concat(xt, rec, 1), 0, &m->r));
    for (size_t i = 0; i < xn.rows; i++) out[i] = (t->data[i] - f->data[i]) * (t->data[i] - f->data[i]);
    ad_tape_clear();
    pyod_matrix_free(&xn);
    return PYOD_OK;
}

static pyod_status alad_fit(pyod_detector *dt, const pyod_matrix *x) {
    alad_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t d = x->cols, z = m->p.latent_dim, n = x->rows, sz[10], ns;
    if (z == 0 || m->p.batch_size == 0) return PYOD_ERR_PARAM;
    alad_reset(m);
    rng_seed(&m->r, m->p.random_state);
    nn_act ag = (nn_act)m->p.activation_hidden_gen, ad_ = (nn_act)m->p.activation_hidden_disc, oa = (nn_act)m->p.output_activation;
    double p = m->p.dropout_rate;
    ns = layers_with(d, m->p.enc_layers, z, sz);
    CHECK(nn_dense_init(&m->enc, &m->gps, sz, ns, ag, m->p.has_output_activation, oa, p, NN_DROP_AFTER_LINEAR, &m->r));
    ns = layers_with(z, m->p.dec_layers, d, sz);
    CHECK(nn_dense_init(&m->dec, &m->gps, sz, ns, ag, m->p.has_output_activation, oa, p, NN_DROP_AFTER_LINEAR, &m->r));
    ns = layers_with(2 * d, m->p.disc_xx_layers, 1, sz);
    CHECK(nn_dense_init(&m->dxx, &m->dps, sz, ns, ad_, 0, NN_IDENTITY, p, NN_DROP_AFTER_LINEAR, &m->r));
    ns = layers_with(2 * z, m->p.disc_zz_layers, 1, sz);
    CHECK(nn_dense_init(&m->dzz, &m->dps, sz, ns, ad_, 0, NN_IDENTITY, p, NN_DROP_AFTER_LINEAR, &m->r));
    ns = layers_with(d + z, m->p.disc_xz_layers, 1, sz);
    CHECK(nn_dense_init(&m->dxz, &m->dps, sz, ns, ad_, 0, NN_IDENTITY, p, NN_DROP_AFTER_LINEAR, &m->r));
    if (m->p.preprocessing) {
        CHECK(scaler_fit(&m->scaler, x));
        m->scaled = 1;
    }
    pyod_matrix xn;
    CHECK(mat_copy(x, &xn));
    if (m->scaled) scaler_transform(&m->scaler, x, xn.data);
    ad_optimizer og, od;
    size_t bs = m->p.batch_size < n ? m->p.batch_size : n;
    size_t *idx = NEW(size_t, bs);
    pyod_status st = idx ? ad_opt_init(&og, AD_ADAM, m->gps.p, m->gps.n, m->p.learning_rate_gen, 0, 0) : PYOD_ERR_NOMEM;
    if (st == PYOD_OK) st = ad_opt_init(&od, AD_ADAM, m->dps.p, m->dps.n, m->p.learning_rate_disc, 0, 0);
    for (size_t ep = 0; st == PYOD_OK && ep < m->p.epochs; ep++) {
        rng_sample(&m->r, n, bs, idx);
        ad_tensor *xr = ad_from_rows(&xn, idx, bs);
        ad_tensor *x_real = ad_add(xr, ad_randn(2, AD_SHAPE(bs, d), 0.0, m->p.noise_std, &m->r));
        ad_tensor *z_real = ad_randn(2, AD_SHAPE(bs, z), 0.0, 1.0, &m->r);
        ad_tensor *one = filled(bs, 1.0), *zero = filled(bs, 0.0);
        /* Discriminator step (generator outputs detached). */
        ad_tensor *x_gen = ad_detach(nn_dense_fwd(&m->dec, z_real, 1, &m->r));
        ad_tensor *z_gen = ad_detach(nn_dense_fwd(&m->enc, x_real, 1, &m->r));
        ad_tensor *ld = ad_add(ad_bce_logits(nn_dense_fwd(&m->dxz, ad_concat(x_real, z_gen, 1), 1, &m->r), one),
                               ad_bce_logits(nn_dense_fwd(&m->dxz, ad_concat(x_gen, z_real, 1), 1, &m->r), zero));
        ld = ad_add(ld, ad_bce_logits(nn_dense_fwd(&m->dxx, ad_concat(x_real, x_real, 1), 1, &m->r), one));
        ld = ad_add(ld, ad_bce_logits(nn_dense_fwd(&m->dxx, ad_concat(x_real, x_gen, 1), 1, &m->r), zero));
        if (m->p.add_disc_zz_loss) {
            ld = ad_add(ld, ad_bce_logits(nn_dense_fwd(&m->dzz, ad_concat(z_real, z_real, 1), 1, &m->r), one));
            ld = ad_add(ld, ad_bce_logits(nn_dense_fwd(&m->dzz, ad_concat(z_real, z_gen, 1), 1, &m->r), zero));
        }
        ad_zero_grad(m->dps.p, m->dps.n);
        ad_backward(ld);
        ad_opt_step(&od);
        /* Generator step: flipped labels. */
        x_gen = nn_dense_fwd(&m->dec, z_real, 1, &m->r);
        z_gen = nn_dense_fwd(&m->enc, x_real, 1, &m->r);
        ad_tensor *lg = ad_add(ad_bce_logits(nn_dense_fwd(&m->dxz, ad_concat(x_gen, z_real, 1), 1, &m->r), one),
                               ad_bce_logits(nn_dense_fwd(&m->dxz, ad_concat(x_real, z_gen, 1), 1, &m->r), zero));
        lg = ad_add(lg, ad_bce_logits(nn_dense_fwd(&m->dxx, ad_concat(x_real, x_gen, 1), 1, &m->r), one));
        lg = ad_add(lg, ad_bce_logits(nn_dense_fwd(&m->dxx, ad_concat(x_real, x_real, 1), 1, &m->r), zero));
        if (m->p.add_disc_zz_loss) {
            lg = ad_add(lg, ad_bce_logits(nn_dense_fwd(&m->dzz, ad_concat(z_real, z_gen, 1), 1, &m->r), one));
            lg = ad_add(lg, ad_bce_logits(nn_dense_fwd(&m->dzz, ad_concat(z_real, z_real, 1), 1, &m->r), zero));
        }
        if (m->p.add_recon_loss) {
            ad_tensor *rec = nn_dense_fwd(&m->dec, nn_dense_fwd(&m->enc, x_real, 1, &m->r), 1, &m->r);
            lg = ad_add(lg, ad_mul_s(ad_mean(ad_pow_s(ad_sub(x_real, rec), 2.0)), m->p.lambda_recon_loss));
        }
        ad_zero_grad(m->gps.p, m->gps.n);
        ad_backward(lg);
        ad_opt_step(&og);
        ad_tape_clear();
    }
    if (idx && st == PYOD_OK) ad_opt_free(&og), ad_opt_free(&od);
    free(idx);
    pyod_matrix_free(&xn);
    CHECK(st);
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    alad_scores(m, x, s);
    return fit_state_contamination(dt, s, n, m->p.contamination);
}

static pyod_status alad_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return alad_scores(d->impl, x, out); }

static void alad_destroy(void *p) {
    alad_reset(p);
    free(p);
}

static const pyod_vtable ALAD_VT = {"ALAD", alad_fit, NULL, alad_score, alad_destroy};

pyod_alad_params pyod_alad_defaults(void) {
    pyod_alad_params p;
    memset(&p, 0, sizeof p);
    p.activation_hidden_gen = p.activation_hidden_disc = PYOD_ACT_TANH;
    p.dropout_rate = 0.2;
    p.latent_dim = 2;
    size_t dec[3] = {5, 10, 25}, enc[3] = {25, 10, 5};
    memcpy(p.dec_layers, dec, sizeof dec);
    memcpy(p.enc_layers, enc, sizeof enc);
    memcpy(p.disc_xx_layers, enc, sizeof enc);
    memcpy(p.disc_zz_layers, enc, sizeof enc);
    memcpy(p.disc_xz_layers, enc, sizeof enc);
    p.learning_rate_gen = p.learning_rate_disc = 1e-4;
    p.lambda_recon_loss = 0.1;
    p.noise_std = 0.1;
    p.add_disc_zz_loss = 1;
    p.epochs = 200;
    p.batch_size = 32;
    p.contamination = 0.1;
    p.random_state = 42;
    return p;
}

pyod_detector *pyod_alad_create(const pyod_alad_params *p) {
    alad_impl *m = NEW(alad_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_alad_defaults();
    return detector_new(&ALAD_VT, m);
}

/* ---------------- AnoGAN ---------------- */

typedef struct {
    pyod_anogan_params p;
    nn_params gps, dps, qps;
    nn_dense gen, disc;
    nn_linear query;
    std_scaler scaler;
    int scaled;
    double *query_state; /* query layer after the training samples */
    pyod_rng r;
} anogan_impl;

static void anogan_reset(anogan_impl *m) {
    nn_dense_free(&m->gen), nn_dense_free(&m->disc);
    nn_params_free(&m->gps), nn_params_free(&m->dps), nn_params_free(&m->qps);
    free(m->query_state);
    m->query_state = NULL;
    if (m->scaled) scaler_free(&m->scaler);
    m->scaled = 0;
}

/* Reconstruction + discriminator-output loss of the query for one sample. */
static ad_tensor *query_loss(anogan_impl *m, ad_tensor *sample, int train) {
    ad_tensor *z = nn_linear_fwd(&m->query, ad_const(2, AD_SHAPE(1, m->p.latent_dim_g), NULL));
    ad_tensor *g = nn_dense_fwd(&m->gen, z, train, &m->r);
    ad_tensor *dg = ad_sigmoid(nn_dense_fwd(&m->disc, g, train, &m->r));
    ad_tensor *dorig = ad_detach(ad_sigmoid(nn_dense_fwd(&m->disc, sample, train, &m->r)));
    return ad_add(ad_mean(ad_abs(ad_sub(sample, g))), ad_mean(ad_abs(ad_sub(dorig, dg))));
}

/* Optimise the (warm-started) query layer on each row; score = last loss. */
static pyod_status query_scores(anogan_impl *m, const pyod_matrix *xn, int train, double *out) {
    ad_optimizer oq;
    CHECK(ad_opt_init(&oq, AD_ADAM, m->qps.p, m->qps.n, m->p.learning_rate_query, 0, 0));
    for (size_t i = 0; i < xn->rows; i++) {
        double last = 0.0;
        for (size_t e = 0; e < m->p.epochs_query; e++) {
            ad_tensor *loss = query_loss(m, ad_from_rows(xn, &i, 1), train);
            last = ad_item(loss);
            ad_zero_grad(m->qps.p, m->qps.n);
            ad_backward(loss);
            ad_tape_clear();
            ad_opt_step(&oq);
        }
        out[i] = last;
    }
    ad_opt_free(&oq);
    return PYOD_OK;
}

static pyod_status anogan_fit(pyod_detector *dt, const pyod_matrix *x) {
    anogan_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t d = x->cols, n = x->rows, z = m->p.latent_dim_g, sz[10], ns;
    if (!layer_count(m->p.g_layers) || !layer_count(m->p.d_layers) || z == 0) return PYOD_ERR_PARAM;
    anogan_reset(m);
    rng_seed(&m->r, m->p.random_state);
    nn_act act = (nn_act)m->p.activation_hidden;
    ns = layers_with(z, m->p.g_layers, d, sz);
    CHECK(nn_dense_init(&m->gen, &m->gps, sz, ns, act, m->p.has_output_activation, (nn_act)m->p.output_activation, m->p.dropout_rate, NN_DROP_INPUT, &m->r));
    ns = layers_with(d, m->p.d_layers, 1, sz);
    CHECK(nn_dense_init(&m->disc, &m->dps, sz, ns, act, 0, NN_IDENTITY, m->p.dropout_rate, NN_DROP_INPUT, &m->r));
    nn_linear_init(&m->query, &m->qps, z, z, 1, NN_INIT_TORCH, 1.0, &m->r);
    if (!m->query.w) return PYOD_ERR_NOMEM;
    if (m->p.preprocessing) {
        CHECK(scaler_fit(&m->scaler, x));
        m->scaled = 1;
    }
    pyod_matrix xn;
    CHECK(mat_copy(x, &xn));
    if (m->scaled) scaler_transform(&m->scaler, x, xn.data);
    ad_optimizer og, od;
    size_t *perm = NEW(size_t, n);
    pyod_status st = perm ? ad_opt_init(&og, AD_ADAM, m->gps.p, m->gps.n, m->p.learning_rate, 0, 0) : PYOD_ERR_NOMEM;
    if (st == PYOD_OK) st = ad_opt_init(&od, AD_ADAM, m->dps.p, m->dps.n, m->p.learning_rate, 0, 0);
    for (size_t ep = 0; st == PYOD_OK && ep < m->p.epochs; ep++) {
        size_t nb = nn_batches(n, m->p.batch_size, 0, &m->r, perm);
        for (size_t b = 0; b < nb; b++) {
            size_t len = nn_batch_len(n, m->p.batch_size, b);
            ad_tensor *real = ad_from_rows(&xn, perm + b * m->p.batch_size, len);
            ad_tensor *fake = nn_dense_fwd(&m->gen, ad_rand(2, AD_SHAPE(len, z), &m->r), 1, &m->r);
            /* One-sided label smoothing (0.9) for real samples. */
            ad_tensor *ld = ad_add(ad_bce_logits(nn_dense_fwd(&m->disc, real, 1, &m->r), filled(len, 0.9)),
                                   ad_bce_logits(nn_dense_fwd(&m->disc, ad_detach(fake), 1, &m->r), filled(len, 0.0)));
            ad_zero_grad(m->dps.p, m->dps.n);
            ad_backward(ld);
            ad_opt_step(&od);
            ad_tensor *lg = ad_bce_logits(nn_dense_fwd(&m->disc, fake, 1, &m->r), filled(len, 1.0));
            ad_zero_grad(m->gps.p, m->gps.n);
            ad_backward(lg);
            ad_opt_step(&og);
            ad_tape_clear();
        }
    }
    if (perm && st == PYOD_OK) ad_opt_free(&og), ad_opt_free(&od);
    free(perm);
    /* Query phase: G and D fixed (dropout active, as PyOD leaves them in training mode). */
    nn_set_requires_grad(&m->gps, 0);
    nn_set_requires_grad(&m->dps, 0);
    double *s = NEW(double, n);
    if (st == PYOD_OK) st = s ? query_scores(m, &xn, 1, s) : PYOD_ERR_NOMEM;
    pyod_matrix_free(&xn);
    size_t qsz = nn_params_size(&m->qps);
    m->query_state = NEW(double, qsz);
    if (st == PYOD_OK && !m->query_state) st = PYOD_ERR_NOMEM;
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    nn_snapshot(&m->qps, m->query_state);
    return fit_state_contamination(dt, s, n, m->p.contamination);
}

/* New data gets the same per-sample query optimisation (PyOD reuses the last
 * training query state instead; see README), starting from the training state. */
static pyod_status anogan_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    anogan_impl *m = d->impl;
    pyod_matrix xn;
    CHECK(mat_copy(x, &xn));
    if (m->scaled) scaler_transform(&m->scaler, x, xn.data);
    nn_restore(&m->qps, m->query_state);
    pyod_status st = query_scores(m, &xn, 0, out);
    nn_restore(&m->qps, m->query_state);
    pyod_matrix_free(&xn);
    return st;
}

static void anogan_destroy(void *p) {
    anogan_reset(p);
    free(p);
}

static const pyod_vtable ANOGAN_VT = {"AnoGAN", anogan_fit, NULL, anogan_score, anogan_destroy};

pyod_anogan_params pyod_anogan_defaults(void) {
    pyod_anogan_params p;
    memset(&p, 0, sizeof p);
    p.activation_hidden = PYOD_ACT_TANH;
    p.dropout_rate = 0.2;
    p.latent_dim_g = 2;
    size_t g[5] = {20, 10, 3, 10, 20}, dl[3] = {20, 10, 5};
    memcpy(p.g_layers, g, sizeof g);
    memcpy(p.d_layers, dl, sizeof dl);
    p.epochs = 500;
    p.batch_size = 32;
    p.epochs_query = 20;
    p.learning_rate = 1e-3;
    p.learning_rate_query = 1e-2;
    p.contamination = 0.1;
    p.random_state = 42;
    return p;
}

pyod_detector *pyod_anogan_create(const pyod_anogan_params *p) {
    anogan_impl *m = NEW(anogan_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_anogan_defaults();
    return detector_new(&ANOGAN_VT, m);
}
