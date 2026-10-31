/* AutoEncoder (pyod.models.auto_encoder), VAE (pyod.models.vae) and Deep SVDD
 * (pyod.models.deep_svdd) on the C autodiff engine. */
#include "deep/common.h"

/* ---------------- AutoEncoder ---------------- */

typedef struct {
    pyod_autoencoder_params p;
    nn_params ps;
    nn_mlp enc, dec;
    nn_normalizer norm;
    pyod_rng r;
} ae_impl;

static void ae_reset(ae_impl *m) {
    nn_mlp_free(&m->enc), nn_mlp_free(&m->dec);
    nn_params_free(&m->ps);
    nn_normalizer_free(&m->norm);
}

static pyod_status ae_scores(ae_impl *m, const pyod_matrix *x, double *out) {
    pyod_matrix xn;
    CHECK(nn_normalizer_apply(&m->norm, x, &xn));
    ad_tensor *in = ad_from_rows(&xn, NULL, xn.rows);
    ad_tensor *rec = nn_mlp_fwd(&m->dec, nn_mlp_fwd(&m->enc, in, 0, &m->r), 0, &m->r);
    nn_row_distances(in, rec, out);
    ad_tape_clear();
    pyod_matrix_free(&xn);
    return PYOD_OK;
}

static pyod_status ae_fit(pyod_detector *dt, const pyod_matrix *x) {
    ae_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t d = x->cols, nh = layer_count(m->p.hidden_neurons), n = x->rows;
    if (nh == 0) return PYOD_ERR_PARAM;
    ae_reset(m);
    rng_seed(&m->r, m->p.train.random_state);
    nn_act act = (nn_act)m->p.hidden_activation;
    size_t prev = d;
    for (size_t i = 0; i < nh; i++) {
        CHECK(nn_mlp_push(&m->enc, &m->ps, prev, m->p.hidden_neurons[i], 1, act, m->p.batch_norm, m->p.dropout_rate, 1, 1, &m->r));
        prev = m->p.hidden_neurons[i];
    }
    for (size_t i = nh - 1; i-- > 0;) {
        CHECK(nn_mlp_push(&m->dec, &m->ps, prev, m->p.hidden_neurons[i], 1, act, m->p.batch_norm, m->p.dropout_rate, 1, 1, &m->r));
        prev = m->p.hidden_neurons[i];
    }
    CHECK(nn_mlp_push(&m->dec, &m->ps, prev, d, 0, NN_IDENTITY, 0, 0.0, 1, 0, &m->r));
    CHECK(nn_normalizer_fit(&m->norm, x, m->p.train.preprocessing));
    pyod_matrix xn;
    CHECK(nn_normalizer_apply(&m->norm, x, &xn));
    ad_optimizer opt;
    size_t *perm = NEW(size_t, n);
    pyod_status st = perm ? ad_opt_init(&opt, AD_ADAM, m->ps.p, m->ps.n, m->p.train.lr, m->p.train.weight_decay, 0) : PYOD_ERR_NOMEM;
    for (size_t ep = 0; st == PYOD_OK && ep < m->p.train.epochs; ep++) {
        size_t nb = nn_batches(n, m->p.train.batch_size, 1, &m->r, perm);
        for (size_t b = 0; b < nb; b++) {
            ad_tensor *xb = ad_from_rows(&xn, perm + b * m->p.train.batch_size, nn_batch_len(n, m->p.train.batch_size, b));
            ad_tensor *rec = nn_mlp_fwd(&m->dec, nn_mlp_fwd(&m->enc, xb, 1, &m->r), 1, &m->r);
            ad_zero_grad(m->ps.p, m->ps.n);
            ad_backward(ad_mean(ad_pow_s(ad_sub(rec, xb), 2.0)));
            ad_tape_clear();
            ad_opt_step(&opt);
        }
    }
    if (perm) ad_opt_free(&opt);
    free(perm);
    pyod_matrix_free(&xn);
    CHECK(st);
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    if ((st = ae_scores(m, x, s)) != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(dt, s, n, m->p.contamination);
}

static pyod_status ae_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return ae_scores(d->impl, x, out); }

static void ae_destroy(void *p) {
    ae_reset(p);
    free(p);
}

static const pyod_vtable AE_VT = {"AutoEncoder", ae_fit, NULL, ae_score, ae_destroy};

pyod_autoencoder_params pyod_autoencoder_defaults(void) {
    pyod_autoencoder_params p;
    memset(&p, 0, sizeof p);
    p.train = (pyod_train_params){10, 32, 1e-3, 1e-5, 1, 42};
    p.hidden_neurons[0] = 64;
    p.hidden_neurons[1] = 32;
    p.hidden_activation = PYOD_ACT_RELU;
    p.batch_norm = 1;
    p.dropout_rate = 0.2;
    p.contamination = 0.1;
    return p;
}

pyod_detector *pyod_autoencoder_create(const pyod_autoencoder_params *p) {
    ae_impl *m = NEW(ae_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_autoencoder_defaults();
    return detector_new(&AE_VT, m);
}

/* ---------------- VAE ---------------- */

typedef struct {
    pyod_vae_params p;
    nn_params ps;
    nn_mlp enc, dec;
    nn_linear mu, logvar;
    nn_normalizer norm;
    pyod_rng r;
} vae_impl;

static void vae_reset(vae_impl *m) {
    nn_mlp_free(&m->enc), nn_mlp_free(&m->dec);
    nn_params_free(&m->ps);
    nn_normalizer_free(&m->norm);
}

/* Returns the reconstruction; mu / logvar outputs optional. */
static ad_tensor *vae_fwd(vae_impl *m, ad_tensor *x, int train, ad_tensor **mu_out, ad_tensor **lv_out) {
    ad_tensor *h = nn_mlp_fwd(&m->enc, x, train, &m->r);
    ad_tensor *mu = nn_linear_fwd(&m->mu, h), *lv = nn_linear_fwd(&m->logvar, h);
    ad_tensor *sd = ad_exp(ad_mul_s(ad_clamp(lv, m->p.logvar_min, m->p.logvar_max), 0.5));
    ad_tensor *eps = ad_randn(sd->ndim, sd->shape, 0.0, 1.0, &m->r);
    if (mu_out) *mu_out = mu;
    if (lv_out) *lv_out = lv;
    return nn_mlp_fwd(&m->dec, ad_add(mu, ad_mul(eps, sd)), train, &m->r);
}

static pyod_status vae_scores(vae_impl *m, const pyod_matrix *x, double *out) {
    pyod_matrix xn;
    CHECK(nn_normalizer_apply(&m->norm, x, &xn));
    ad_tensor *in = ad_from_rows(&xn, NULL, xn.rows);
    nn_row_distances(in, vae_fwd(m, in, 0, NULL, NULL), out);
    ad_tape_clear();
    pyod_matrix_free(&xn);
    return PYOD_OK;
}

static pyod_status vae_fit(pyod_detector *dt, const pyod_matrix *x) {
    vae_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t d = x->cols, ne = layer_count(m->p.encoder_neurons), nd = layer_count(m->p.decoder_neurons), n = x->rows, z = m->p.latent_dim;
    if (!ne || !nd || !z || m->p.logvar_min >= m->p.logvar_max) return PYOD_ERR_PARAM;
    vae_reset(m);
    rng_seed(&m->r, m->p.train.random_state);
    nn_act act = (nn_act)m->p.hidden_activation;
    size_t prev = d;
    for (size_t i = 0; i < ne; i++) {
        CHECK(nn_mlp_push(&m->enc, &m->ps, prev, m->p.encoder_neurons[i], 1, act, m->p.batch_norm, m->p.dropout_rate, 1, 1, &m->r));
        prev = m->p.encoder_neurons[i];
    }
    nn_linear_init(&m->mu, &m->ps, prev, z, 1, NN_INIT_TORCH, 1.0, &m->r);
    nn_linear_init(&m->logvar, &m->ps, prev, z, 1, NN_INIT_TORCH, 1.0, &m->r);
    prev = z;
    for (size_t i = 0; i < nd; i++) {
        CHECK(nn_mlp_push(&m->dec, &m->ps, prev, m->p.decoder_neurons[i], 1, act, m->p.batch_norm, m->p.dropout_rate, 1, 1, &m->r));
        prev = m->p.decoder_neurons[i];
    }
    CHECK(nn_mlp_push(&m->dec, &m->ps, prev, d, 1, (nn_act)m->p.output_activation, 0, 0.0, 1, 1, &m->r));
    if (!m->mu.w || !m->logvar.w) return PYOD_ERR_NOMEM;
    CHECK(nn_normalizer_fit(&m->norm, x, m->p.train.preprocessing));
    pyod_matrix xn;
    CHECK(nn_normalizer_apply(&m->norm, x, &xn));
    ad_optimizer opt;
    size_t *perm = NEW(size_t, n);
    pyod_status st = perm ? ad_opt_init(&opt, AD_ADAM, m->ps.p, m->ps.n, m->p.train.lr, m->p.train.weight_decay, 0) : PYOD_ERR_NOMEM;
    for (size_t ep = 0; st == PYOD_OK && ep < m->p.train.epochs; ep++) {
        size_t nb = nn_batches(n, m->p.train.batch_size, 1, &m->r, perm);
        for (size_t b = 0; b < nb; b++) {
            ad_tensor *xb = ad_from_rows(&xn, perm + b * m->p.train.batch_size, nn_batch_len(n, m->p.train.batch_size, b));
            ad_tensor *mu, *lv;
            ad_tensor *rec = vae_fwd(m, xb, 1, &mu, &lv);
            ad_tensor *recon = ad_mean(ad_pow_s(ad_sub(rec, xb), 2.0));
            ad_tensor *lvc = ad_clamp(lv, m->p.logvar_min, m->p.logvar_max);
            ad_tensor *kl_terms = ad_sub(ad_sub(ad_add_s(lvc, 1.0), ad_pow_s(mu, 2.0)), ad_exp(lvc));
            ad_tensor *kl = ad_mean(ad_mul_s(ad_sum_dim(kl_terms, 1), -0.5));
            ad_tensor *loss = ad_add(recon, ad_mul_s(ad_clamp(kl, 0.0, m->p.capacity), m->p.beta));
            ad_zero_grad(m->ps.p, m->ps.n);
            ad_backward(loss);
            ad_tape_clear();
            ad_opt_step(&opt);
        }
    }
    if (perm) ad_opt_free(&opt);
    free(perm);
    pyod_matrix_free(&xn);
    CHECK(st);
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    /* Scores use a sampled reconstruction (as in PyOD), reseeded so repeated calls agree. */
    rng_seed(&m->r, m->p.train.random_state + 1);
    if ((st = vae_scores(m, x, s)) != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(dt, s, n, m->p.contamination);
}

static pyod_status vae_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    vae_impl *m = d->impl;
    rng_seed(&m->r, m->p.train.random_state + 1);
    return vae_scores(m, x, out);
}

static void vae_destroy(void *p) {
    vae_reset(p);
    free(p);
}

static const pyod_vtable VAE_VT = {"VAE", vae_fit, NULL, vae_score, vae_destroy};

pyod_vae_params pyod_vae_defaults(void) {
    pyod_vae_params p;
    memset(&p, 0, sizeof p);
    p.train = (pyod_train_params){30, 32, 1e-3, 1e-5, 1, 42};
    size_t enc[3] = {128, 64, 32}, dec[3] = {32, 64, 128};
    memcpy(p.encoder_neurons, enc, sizeof enc);
    memcpy(p.decoder_neurons, dec, sizeof dec);
    p.latent_dim = 2;
    p.hidden_activation = PYOD_ACT_RELU;
    p.output_activation = PYOD_ACT_IDENTITY;
    p.dropout_rate = 0.2;
    p.beta = 1.0;
    p.capacity = 0.0;
    p.logvar_min = -30.0;
    p.logvar_max = 20.0;
    p.contamination = 0.1;
    return p;
}

pyod_detector *pyod_vae_create(const pyod_vae_params *p) {
    vae_impl *m = NEW(vae_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_vae_defaults();
    return detector_new(&VAE_VT, m);
}

/* ---------------- Deep SVDD ---------------- */

typedef struct {
    pyod_deep_svdd_params p;
    nn_params ps;
    nn_mlp net;
    double *center;
    size_t out_dim;
    std_scaler scaler;
    int scaled;
    pyod_rng r;
} svdd_impl;

static void svdd_reset(svdd_impl *m) {
    nn_mlp_free(&m->net);
    nn_params_free(&m->ps);
    free(m->center);
    m->center = NULL;
    if (m->scaled) scaler_free(&m->scaler);
    m->scaled = 0;
}

static pyod_status svdd_input(const svdd_impl *m, const pyod_matrix *x, pyod_matrix *out) {
    CHECK(mat_copy(x, out));
    if (m->scaled) scaler_transform(&m->scaler, x, out->data);
    return PYOD_OK;
}

static pyod_status svdd_scores(svdd_impl *m, const pyod_matrix *x, double *out) {
    pyod_matrix xn;
    CHECK(svdd_input(m, x, &xn));
    ad_tensor *o = nn_mlp_fwd(&m->net, ad_from_rows(&xn, NULL, xn.rows), 0, &m->r);
    for (size_t i = 0; i < xn.rows; i++) {
        double s = 0.0;
        for (size_t j = 0; j < m->out_dim; j++) {
            double dd = o->data[i * m->out_dim + j] - m->center[j];
            s += dd * dd;
        }
        out[i] = s;
    }
    ad_tape_clear();
    pyod_matrix_free(&xn);
    return PYOD_OK;
}

static pyod_status svdd_fit(pyod_detector *dt, const pyod_matrix *x) {
    svdd_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t d = x->cols, nh = layer_count(m->p.hidden_neurons), n = x->rows;
    const size_t *h = m->p.hidden_neurons;
    if (nh < 2) return PYOD_ERR_PARAM;
    svdd_reset(m);
    rng_seed(&m->r, m->p.train.random_state);
    nn_act act = (nn_act)m->p.hidden_activation;
    /* InnerDeepSVDD._build_model: bias-free; dropout only on inner layers. */
    CHECK(nn_mlp_push(&m->net, &m->ps, d, h[0], 1, act, 0, 0.0, 0, 0, &m->r));
    for (size_t i = 1; i + 1 < nh; i++) CHECK(nn_mlp_push(&m->net, &m->ps, h[i - 1], h[i], 1, act, 0, m->p.dropout_rate, 0, 0, &m->r));
    CHECK(nn_mlp_push(&m->net, &m->ps, h[nh - 2], h[nh - 1], 1, act, 0, 0.0, 0, 0, &m->r));
    if (m->p.use_ae) {
        for (size_t j = nh - 1; j >= 1; j--) CHECK(nn_mlp_push(&m->net, &m->ps, h[j], h[j - 1], 1, act, 0, m->p.dropout_rate, 0, 0, &m->r));
        CHECK(nn_mlp_push(&m->net, &m->ps, h[0], d, 1, (nn_act)m->p.output_activation, 0, 0.0, 0, 0, &m->r));
    }
    m->out_dim = m->p.use_ae ? d : h[nh - 1];
    if (m->p.train.preprocessing) {
        CHECK(scaler_fit(&m->scaler, x));
        m->scaled = 1;
    }
    pyod_matrix xn;
    CHECK(svdd_input(m, x, &xn));
    m->center = NEW(double, m->out_dim);
    if (!m->center) {
        pyod_matrix_free(&xn);
        return PYOD_ERR_NOMEM;
    }
    pyod_status st = PYOD_OK;
    if (m->p.center) {
        if (m->p.center_len != 1 && m->p.center_len != m->out_dim) st = PYOD_ERR_PARAM;
        for (size_t j = 0; st == PYOD_OK && j < m->out_dim; j++) m->center[j] = m->p.center[m->p.center_len == 1 ? 0 : j];
    } else {
        ad_tensor *o = nn_mlp_fwd(&m->net, ad_from_rows(&xn, NULL, n), 0, &m->r);
        for (size_t j = 0; j < m->out_dim; j++) {
            double c = 0.0;
            for (size_t i = 0; i < n; i++) c += o->data[i * m->out_dim + j] / (double)n;
            if (fabs(c) < 0.1 && c != 0.0) c = c < 0 ? -0.1 : 0.1;
            m->center[j] = c;
        }
        ad_tape_clear();
    }
    int nonzero = 0;
    for (size_t j = 0; j < m->out_dim; j++) {
        if (!isfinite(m->center[j])) st = PYOD_ERR_PARAM;
        nonzero |= m->center[j] != 0.0;
    }
    if (!nonzero) st = PYOD_ERR_PARAM; /* all-zero centre collapses the hypersphere */
    ad_optimizer opt;
    size_t *perm = NEW(size_t, n), psz = nn_params_size(&m->ps);
    double *best = NEW(double, psz), best_loss = INFINITY;
    int have_best = 0;
    if (st == PYOD_OK) st = perm && best ? ad_opt_init(&opt, AD_ADAM, m->ps.p, m->ps.n, m->p.train.lr, m->p.train.weight_decay, 0) : PYOD_ERR_NOMEM;
    for (size_t ep = 0; st == PYOD_OK && ep < m->p.train.epochs; ep++) {
        double epoch_loss = 0.0;
        size_t nb = nn_batches(n, m->p.train.batch_size, 0, &m->r, perm);
        for (size_t b = 0; b < nb; b++) {
            ad_tensor *xb = ad_from_rows(&xn, perm + b * m->p.train.batch_size, nn_batch_len(n, m->p.train.batch_size, b));
            ad_tensor *out = nn_mlp_fwd(&m->net, xb, 1, &m->r);
            ad_tensor *c = ad_const(1, AD_SHAPE(m->out_dim), m->center);
            ad_tensor *loss = ad_mean(ad_sum_dim(ad_pow_s(ad_sub(out, c), 2.0), 1));
            /* + 1e-6 * sum of Frobenius norms of the weights */
            for (size_t q = 0; q < m->ps.n; q++) loss = ad_add(loss, ad_mul_s(ad_sqrt(ad_sum(ad_pow_s(m->ps.p[q], 2.0))), 1e-6));
            if (m->p.use_ae) loss = ad_add(loss, ad_mean(ad_pow_s(ad_sub(out, xb), 2.0)));
            epoch_loss += ad_item(loss);
            ad_zero_grad(m->ps.p, m->ps.n);
            ad_backward(loss);
            ad_tape_clear();
            ad_opt_step(&opt);
        }
        if (epoch_loss < best_loss) {
            best_loss = epoch_loss;
            nn_snapshot(&m->ps, best);
            have_best = 1;
        }
    }
    if (st == PYOD_OK) {
        ad_opt_free(&opt);
        if (have_best) nn_restore(&m->ps, best);
    }
    free(perm), free(best);
    pyod_matrix_free(&xn);
    CHECK(st);
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    if ((st = svdd_scores(m, x, s)) != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(dt, s, n, m->p.contamination);
}

static pyod_status svdd_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return svdd_scores(d->impl, x, out); }

static void svdd_destroy(void *p) {
    svdd_reset(p);
    free(p);
}

static const pyod_vtable SVDD_VT = {"DeepSVDD", svdd_fit, NULL, svdd_score, svdd_destroy};

pyod_deep_svdd_params pyod_deep_svdd_defaults(void) {
    pyod_deep_svdd_params p;
    memset(&p, 0, sizeof p);
    p.train = (pyod_train_params){100, 32, 1e-4, 5e-7, 1, 42};
    p.hidden_neurons[0] = 64;
    p.hidden_neurons[1] = 32;
    p.hidden_activation = PYOD_ACT_RELU;
    p.output_activation = PYOD_ACT_SIGMOID;
    p.dropout_rate = 0.2;
    p.contamination = 0.1;
    return p;
}

pyod_detector *pyod_deep_svdd_create(const pyod_deep_svdd_params *p) {
    svdd_impl *m = NEW(svdd_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_deep_svdd_defaults();
    return detector_new(&SVDD_VT, m);
}
