/* AE-1SVM (pyod.models.ae1svm), DevNet (pyod.models.devnet) and LUNAR
 * (pyod.models.lunar) on the C autodiff engine. */
#include "deep/common.h"

/* ---------------- AE-1SVM ---------------- */

typedef struct {
    pyod_ae1svm_params p;
    nn_params ps;
    nn_mlp enc, dec;
    ad_tensor *rff_w, *rff_b, *svm_w, *svm_b;
    double *mean, *std;
    pyod_rng r;
} ae1svm_impl;

static void ae1svm_reset(ae1svm_impl *m) {
    nn_mlp_free(&m->enc), nn_mlp_free(&m->dec);
    nn_params_free(&m->ps);
    free(m->mean), free(m->std);
    m->mean = m->std = NULL;
}

/* (x - mean) / (std + 1e-8), with zero stds replaced by 1e-6 as in PyOD. */
static pyod_status ae1svm_input(const ae1svm_impl *m, const pyod_matrix *x, pyod_matrix *out) {
    CHECK(mat_copy(x, out));
    if (m->mean)
        for (size_t i = 0; i < x->rows; i++)
            for (size_t j = 0; j < x->cols; j++) PYOD_AT(out, i, j) = (PYOD_AT(x, i, j) - m->mean[j]) / (m->std[j] + 1e-8);
    return PYOD_OK;
}

static pyod_status ae1svm_scores(ae1svm_impl *m, const pyod_matrix *x, double *out) {
    pyod_matrix xn;
    CHECK(ae1svm_input(m, x, &xn));
    ad_tensor *in = ad_from_rows(&xn, NULL, xn.rows);
    nn_row_distances(in, nn_mlp_fwd(&m->dec, nn_mlp_fwd(&m->enc, in, 0, &m->r), 0, &m->r), out);
    ad_tape_clear();
    pyod_matrix_free(&xn);
    return PYOD_OK;
}

static pyod_status ae1svm_fit(pyod_detector *dt, const pyod_matrix *x) {
    ae1svm_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t d = x->cols, n = x->rows, e = m->p.encoding_dim, k = m->p.kernel_approx_features, sz[11], ns = 0;
    if (!e || !k) return PYOD_ERR_PARAM;
    ae1svm_reset(m);
    rng_seed(&m->r, m->p.train.random_state);
    if (m->p.train.preprocessing) {
        m->mean = NEW(double, d);
        m->std = NEW(double, d);
        if (!m->mean || !m->std) return PYOD_ERR_NOMEM;
        col_mean(x, m->mean);
        for (size_t j = 0; j < d; j++) {
            double v = 0.0;
            for (size_t i = 0; i < n; i++) v += (PYOD_AT(x, i, j) - m->mean[j]) * (PYOD_AT(x, i, j) - m->mean[j]);
            m->std[j] = sqrt(v / (double)n);
            if (m->std[j] == 0.0) m->std[j] = 1e-6;
        }
    }
    sz[ns++] = d;
    for (size_t i = 0; i < layer_count(m->p.hidden_neurons); i++) sz[ns++] = m->p.hidden_neurons[i];
    sz[ns++] = e;
    nn_act act = (nn_act)m->p.hidden_activation;
    for (size_t i = 0; i + 1 < ns; i++)
        CHECK(nn_mlp_push(&m->enc, &m->ps, sz[i], sz[i + 1], 1, act, m->p.batch_norm, m->p.dropout_rate, 1, 0, &m->r));
    /* The decoder's last layer keeps the activation but has no batch norm or dropout. */
    for (size_t i = ns - 1; i > 0; i--) {
        int last = i == 1;
        CHECK(nn_mlp_push(&m->dec, &m->ps, sz[i], sz[i - 1], 1, act, last ? 0 : m->p.batch_norm, last ? 0.0 : m->p.dropout_rate, 1, 0, &m->r));
    }
    m->rff_w = nn_new_param(&m->ps, 2, AD_SHAPE(e, k));
    m->rff_b = nn_new_param(&m->ps, 1, AD_SHAPE(k));
    m->svm_w = nn_new_param(&m->ps, 2, AD_SHAPE(k, 1));
    m->svm_b = nn_new_param(&m->ps, 1, AD_SHAPE(1));
    if (!m->rff_w || !m->rff_b || !m->svm_w || !m->svm_b) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < e * k; i++) m->rff_w->data[i] = rng_normal(&m->r) * m->p.sigma;
    for (size_t i = 0; i < k; i++) m->rff_b->data[i] = rng_normal(&m->r) * 2.0 * PYOD_PI, m->svm_w->data[i] = rng_normal(&m->r);
    m->svm_b->data[0] = rng_normal(&m->r);
    pyod_matrix xn;
    CHECK(ae1svm_input(m, x, &xn));
    ad_optimizer opt;
    size_t *perm = NEW(size_t, n);
    pyod_status st = perm ? ad_opt_init(&opt, AD_ADAM, m->ps.p, m->ps.n, m->p.train.lr, m->p.train.weight_decay, 0) : PYOD_ERR_NOMEM;
    for (size_t ep = 0; st == PYOD_OK && ep < m->p.train.epochs; ep++) {
        size_t nb = nn_batches(n, m->p.train.batch_size, 0, &m->r, perm);
        for (size_t b = 0; b < nb; b++) {
            ad_tensor *xb = ad_from_rows(&xn, perm + b * m->p.train.batch_size, nn_batch_len(n, m->p.train.batch_size, b));
            ad_tensor *code = nn_mlp_fwd(&m->enc, xb, 1, &m->r);
            ad_tensor *rff = ad_cos(ad_add(ad_matmul(code, m->rff_w), m->rff_b));
            ad_tensor *svm = ad_add(ad_matmul(rff, m->svm_w), m->svm_b);
            ad_tensor *recon = ad_mean(ad_pow_s(ad_sub(xb, nn_mlp_fwd(&m->dec, code, 1, &m->r)), 2.0));
            ad_tensor *hinge = ad_mean(ad_clamp(ad_add_s(ad_neg(svm), 1.0), 0.0, INFINITY));
            ad_zero_grad(m->ps.p, m->ps.n);
            ad_backward(ad_add(ad_mul_s(recon, m->p.alpha), hinge));
            ad_tape_clear();
            ad_opt_step(&opt);
        }
    }
    if (perm && st == PYOD_OK) ad_opt_free(&opt);
    free(perm);
    pyod_matrix_free(&xn);
    CHECK(st);
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    ae1svm_scores(m, x, s);
    return fit_state_contamination(dt, s, n, m->p.contamination);
}

static pyod_status ae1svm_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return ae1svm_scores(d->impl, x, out); }

static void ae1svm_destroy(void *p) {
    ae1svm_reset(p);
    free(p);
}

static const pyod_vtable AE1SVM_VT = {"AE1SVM", ae1svm_fit, NULL, ae1svm_score, ae1svm_destroy};

pyod_ae1svm_params pyod_ae1svm_defaults(void) {
    pyod_ae1svm_params p;
    memset(&p, 0, sizeof p);
    p.train = (pyod_train_params){50, 32, 1e-3, 1e-5, 1, 42};
    p.hidden_neurons[0] = 64;
    p.hidden_neurons[1] = 32;
    p.encoding_dim = 32;
    p.kernel_approx_features = 1000;
    p.hidden_activation = PYOD_ACT_RELU;
    p.batch_norm = 1;
    p.dropout_rate = 0.2;
    p.alpha = 1.0;
    p.sigma = 1.0;
    p.contamination = 0.1;
    return p;
}

pyod_detector *pyod_ae1svm_create(const pyod_ae1svm_params *p) {
    ae1svm_impl *m = NEW(ae1svm_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_ae1svm_defaults();
    return detector_new(&AE1SVM_VT, m);
}

/* ---------------- DevNet ---------------- */

typedef struct {
    pyod_devnet_params p;
    nn_params ps;
    nn_linear hidden[3], score;
    size_t n_hidden;
    pyod_rng r;
} devnet_impl;

static ad_tensor *devnet_fwd(const devnet_impl *m, ad_tensor *x) {
    for (size_t i = 0; i < m->n_hidden; i++) x = ad_relu(nn_linear_fwd(&m->hidden[i], x));
    return nn_linear_fwd(&m->score, x);
}

static pyod_status devnet_scores(const devnet_impl *m, const pyod_matrix *x, double *out) {
    ad_tensor *s = devnet_fwd(m, ad_from_rows(x, NULL, x->rows));
    memcpy(out, s->data, x->rows * sizeof(double));
    ad_tape_clear();
    return PYOD_OK;
}

/* Deviation loss as in the paper (PyOD swaps its arguments; see README):
 * inliers pulled toward the N(0, 1) reference, labelled outliers pushed at
 * least confidence_margin standard deviations above it. Batches alternate
 * inliers and labelled outliers, as in the paper and PyOD's own (unused)
 * input_batch_generation_sup_sparse; with few labels, plain shuffled batches
 * let the inlier term dominate and training can collapse. */
static pyod_status devnet_fit_supervised(pyod_detector *dt, const pyod_matrix *x, const uint8_t *y) {
    devnet_impl *m = dt->impl;
    size_t d = x->cols, n = x->rows, widths[3], nw;
    switch (m->p.network_depth) {
    case 1: nw = 0; break;
    case 2: nw = 1, widths[0] = 1000; break;
    case 4: nw = 3, widths[0] = 1000, widths[1] = 250, widths[2] = 20; break;
    default: return PYOD_ERR_PARAM;
    }
    nn_params_free(&m->ps);
    rng_seed(&m->r, m->p.random_state);
    size_t prev = d;
    m->n_hidden = nw;
    for (size_t i = 0; i < nw; i++) {
        nn_linear_init(&m->hidden[i], &m->ps, prev, widths[i], 1, NN_INIT_TORCH, 1.0, &m->r);
        prev = widths[i];
    }
    nn_linear_init(&m->score, &m->ps, prev, 1, 1, NN_INIT_TORCH, 1.0, &m->r);
    if (!m->score.w) return PYOD_ERR_NOMEM;
    ad_optimizer opt;
    size_t bs = m->p.batch_size ? m->p.batch_size : 1, n_out = 0, n_in = 0;
    size_t *perm = NEW(size_t, n), *outl = NEW(size_t, n), *inl = NEW(size_t, n), *idx_buf = NEW(size_t, bs);
    double *lab = NEW(double, n > bs ? n : bs);
    for (size_t i = 0; perm && outl && inl && i < n; i++) {
        if (y[i]) outl[n_out++] = i;
        else inl[n_in++] = i;
    }
    int balanced = n_out > 0 && n_in > 0;
    pyod_status st = perm && outl && inl && idx_buf && lab ? ad_opt_init(&opt, AD_RMSPROP, m->ps.p, m->ps.n, m->p.lr, 1e-6, 0) : PYOD_ERR_NOMEM;
    for (size_t ep = 0; st == PYOD_OK && ep < m->p.epochs; ep++) {
        size_t nb = balanced ? (m->p.nb_batch ? m->p.nb_batch : 1) : nn_batches(n, bs, 0, &m->r, perm);
        for (size_t b = 0; b < nb; b++) {
            size_t len, *idx;
            if (balanced) {
                len = bs, idx = idx_buf;
                for (size_t i = 0; i < len; i++) idx[i] = i % 2 == 0 ? inl[rng_below(&m->r, n_in)] : outl[rng_below(&m->r, n_out)];
            } else {
                len = nn_batch_len(n, bs, b), idx = perm + b * bs;
            }
            for (size_t i = 0; i < len; i++) lab[i] = (double)y[idx[i]];
            ad_tensor *pred = devnet_fwd(m, ad_from_rows(x, idx, len));
            ad_tensor *yb = ad_const(2, AD_SHAPE(len, 1), lab);
            ad_tensor *ref = ad_randn(1, AD_SHAPE(5000), 0.0, 1.0, &m->r);
            double mu = vec_mean(ref->data, 5000), sd = 0.0;
            for (size_t i = 0; i < 5000; i++) sd += (ref->data[i] - mu) * (ref->data[i] - mu);
            sd = sqrt(sd / 4999.0); /* torch.std is unbiased */
            ad_tensor *dev = ad_mul_s(ad_add_s(pred, -mu), 1.0 / sd);
            ad_tensor *inlier = ad_abs(dev);
            ad_tensor *outlier = ad_clamp(ad_add_s(ad_neg(dev), m->p.confidence_margin), 0.0, INFINITY);
            ad_tensor *loss = ad_mean(ad_add(ad_mul(ad_add_s(ad_neg(yb), 1.0), inlier), ad_mul(yb, outlier)));
            ad_zero_grad(m->ps.p, m->ps.n);
            ad_backward(loss);
            ad_tape_clear();
            ad_opt_step(&opt);
        }
    }
    if (perm && outl && inl && idx_buf && lab && st == PYOD_OK) ad_opt_free(&opt);
    free(perm), free(outl), free(inl), free(idx_buf), free(lab);
    CHECK(st);
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    devnet_scores(m, x, s);
    double c = m->p.contamination < 1e-9 ? 1e-9 : m->p.contamination > 0.5 ? 0.5 : m->p.contamination;
    return fit_state_contamination(dt, s, n, c);
}

static pyod_status devnet_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return devnet_scores(d->impl, x, out); }

static void devnet_destroy(void *p) {
    devnet_impl *m = p;
    nn_params_free(&m->ps);
    free(m);
}

/* fit without labels is unsupported: DevNet is weakly supervised. */
static const pyod_vtable DEVNET_VT = {"DevNet", NULL, devnet_fit_supervised, devnet_score, devnet_destroy};

pyod_devnet_params pyod_devnet_defaults(void) { return (pyod_devnet_params){2, 512, 50, 20, 5.0, 1e-3, 0.1, 42}; }

pyod_detector *pyod_devnet_create(const pyod_devnet_params *p) {
    devnet_impl *m = NEW(devnet_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_devnet_defaults();
    return detector_new(&DEVNET_VT, m);
}

/* ---------------- LUNAR ---------------- */

typedef struct {
    pyod_lunar_params p;
    nn_params ps;
    nn_linear l[3], out;
    nn_layernorm ln;
    nn_batchnorm final_bn;
    int weight_model, have_bn;
    double *min, *range;
    pyod_matrix ref; /* scaled training samples + their negatives */
    pyod_rng r;
} lunar_impl;

static void lunar_reset(lunar_impl *m) {
    nn_params_free(&m->ps);
    if (m->have_bn) nn_batchnorm_free(&m->final_bn);
    m->have_bn = 0;
    free(m->min), free(m->range);
    m->min = m->range = NULL;
    pyod_matrix_free(&m->ref);
}

static ad_tensor *lunar_fwd(lunar_impl *m, ad_tensor *dist, int train) {
    ad_tensor *h = dist;
    for (int i = 0; i < 3; i++) h = m->weight_model ? ad_relu(nn_linear_fwd(&m->l[i], h)) : ad_tanh(nn_linear_fwd(&m->l[i], h));
    if (!m->weight_model) return ad_sigmoid(nn_linear_fwd(&m->out, h));
    ad_tensor *alpha = ad_softmax(nn_linear_fwd(&m->out, nn_layernorm_fwd(&m->ln, h)));
    return nn_batchnorm_fwd(&m->final_bn, ad_sum_dim(ad_mul(alpha, dist), 1), train);
}

static void lunar_scale(const lunar_impl *m, const pyod_matrix *x, pyod_matrix *out) {
    for (size_t i = 0; i < x->rows; i++)
        for (size_t j = 0; j < x->cols; j++)
            PYOD_AT(out, i, j) = m->min ? (PYOD_AT(x, i, j) - m->min[j]) / m->range[j] : PYOD_AT(x, i, j);
}

/* Distances to the k nearest reference rows (skipping the closest one when skip_self). */
static pyod_status lunar_dist(const lunar_impl *m, const pyod_matrix *x, int skip_self, pyod_matrix *out) {
    size_t k = m->p.n_neighbours;
    CHECK(pyod_matrix_alloc(out, x->rows, k));
    double *buf = NEW(double, k + 1);
    if (!buf) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < x->rows; i++) {
        k_nearest(&m->ref, &PYOD_AT(x, i, 0), k + (size_t)skip_self, SIZE_MAX, NULL, buf);
        memcpy(&PYOD_AT(out, i, 0), buf + skip_self, k * sizeof(double));
    }
    free(buf);
    return PYOD_OK;
}

/* Port of generate_negative_samples (uniform and/or perturbed copies). */
static pyod_status negatives(const lunar_impl *m, const pyod_matrix *x, pyod_rng *r, pyod_matrix *out) {
    size_t n = (size_t)(m->p.proportion * (double)x->rows), d = x->cols;
    CHECK(pyod_matrix_alloc(out, n, d));
    double lo = vec_min(x->data, x->rows * d), hi = vec_max(x->data, x->rows * d);
    pyod_matrix uni, sub;
    CHECK(pyod_matrix_alloc(&uni, n, d));
    if (pyod_matrix_alloc(&sub, n, d) != PYOD_OK) {
        pyod_matrix_free(&uni);
        return PYOD_ERR_NOMEM;
    }
    for (size_t i = 0; i < n * d; i++) uni.data[i] = lo + (hi - lo) * rng_f64(r);
    for (size_t i = 0; i < n; i++) {
        const double *base = &PYOD_AT(x, rng_below(r, x->rows), 0);
        for (size_t j = 0; j < d; j++) PYOD_AT(&sub, i, j) = rng_f64(r) < 0.3 ? base[j] + m->p.epsilon * rng_normal(r) : base[j];
    }
    for (size_t i = 0; i < n; i++) {
        const double *src;
        if (m->p.negative_sampling == PYOD_NEG_UNIFORM) src = &PYOD_AT(&uni, i, 0);
        else if (m->p.negative_sampling == PYOD_NEG_SUBSPACE) src = &PYOD_AT(&sub, i, 0);
        else {
            size_t pick = rng_below(r, 2 * n);
            src = pick < n ? &PYOD_AT(&uni, pick, 0) : &PYOD_AT(&sub, pick - n, 0);
        }
        memcpy(&PYOD_AT(out, i, 0), src, d * sizeof(double));
    }
    pyod_matrix_free(&uni);
    pyod_matrix_free(&sub);
    return PYOD_OK;
}

static pyod_status lunar_scores(lunar_impl *m, const pyod_matrix *x, double *out) {
    pyod_matrix xs, dist;
    CHECK(pyod_matrix_alloc(&xs, x->rows, x->cols));
    lunar_scale(m, x, &xs);
    pyod_status st = lunar_dist(m, &xs, 0, &dist);
    pyod_matrix_free(&xs);
    CHECK(st);
    ad_tensor *s = lunar_fwd(m, ad_from_rows(&dist, NULL, dist.rows), 0);
    memcpy(out, s->data, x->rows * sizeof(double));
    ad_tape_clear();
    pyod_matrix_free(&dist);
    return PYOD_OK;
}

static pyod_status lunar_fit(pyod_detector *dt, const pyod_matrix *x) {
    lunar_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t n = x->rows, d = x->cols, k = m->p.n_neighbours, n_val = (size_t)ceil((double)n * m->p.val_size);
    if (n_val < 1) n_val = 1;
    if (k == 0 || n < n_val + k + 2) return PYOD_ERR_PARAM;
    lunar_reset(m);
    rng_seed(&m->r, m->p.random_state);
    m->weight_model = m->p.model_type == PYOD_LUNAR_WEIGHT;
    size_t *perm = NEW(size_t, n);
    if (!perm) return PYOD_ERR_NOMEM;
    rng_sample(&m->r, n, n, perm);
    pyod_matrix tr = {0}, va = {0}, ntr = {0}, nva = {0}, trd = {0}, ntrd = {0}, vad = {0}, nvad = {0}, all_d = {0}, allv = {0};
    pyod_status st = mat_rows(x, perm + n_val, n - n_val, &tr);
    if (st == PYOD_OK) st = mat_rows(x, perm, n_val, &va);
    free(perm);
    if (st == PYOD_OK && m->p.scale) {
        m->min = NEW(double, d);
        m->range = NEW(double, d);
        if (!m->min || !m->range) st = PYOD_ERR_NOMEM;
        for (size_t j = 0; st == PYOD_OK && j < d; j++) {
            double lo = INFINITY, hi = -INFINITY;
            for (size_t i = 0; i < tr.rows; i++) lo = fmin(lo, PYOD_AT(&tr, i, j)), hi = fmax(hi, PYOD_AT(&tr, i, j));
            m->min[j] = lo;
            m->range[j] = hi - lo == 0.0 ? 1.0 : hi - lo;
        }
    }
    if (st == PYOD_OK) lunar_scale(m, &tr, &tr), lunar_scale(m, &va, &va);
    if (st == PYOD_OK) st = negatives(m, &tr, &m->r, &ntr);
    if (st == PYOD_OK) st = negatives(m, &va, &m->r, &nva);
    /* Neighbours are searched among training samples and their negatives. */
    if (st == PYOD_OK) st = pyod_matrix_alloc(&m->ref, tr.rows + ntr.rows, d);
    if (st == PYOD_OK) {
        memcpy(m->ref.data, tr.data, tr.rows * d * sizeof(double));
        memcpy(m->ref.data + tr.rows * d, ntr.data, ntr.rows * d * sizeof(double));
        st = lunar_dist(m, &tr, 1, &trd);
    }
    if (st == PYOD_OK) st = lunar_dist(m, &ntr, 0, &ntrd);
    if (st == PYOD_OK) st = lunar_dist(m, &va, 0, &vad);
    if (st == PYOD_OK) st = lunar_dist(m, &nva, 0, &nvad);
    size_t nt = trd.rows + ntrd.rows, nv = vad.rows + nvad.rows;
    double *ty = NEW(double, nt), *vs = NEW(double, nv);
    uint8_t *vy = NEW(uint8_t, nv);
    if (st == PYOD_OK && (!ty || !vs || !vy)) st = PYOD_ERR_NOMEM;
    if (st == PYOD_OK) st = pyod_matrix_alloc(&all_d, nt, k);
    if (st == PYOD_OK) st = pyod_matrix_alloc(&allv, nv, k);
    if (st == PYOD_OK) {
        memcpy(all_d.data, trd.data, trd.rows * k * sizeof(double));
        memcpy(all_d.data + trd.rows * k, ntrd.data, ntrd.rows * k * sizeof(double));
        memcpy(allv.data, vad.data, vad.rows * k * sizeof(double));
        memcpy(allv.data + vad.rows * k, nvad.data, nvad.rows * k * sizeof(double));
        for (size_t i = 0; i < nt; i++) ty[i] = i >= trd.rows;
        for (size_t i = 0; i < nv; i++) vy[i] = i >= vad.rows;
        size_t h = 256;
        for (int i = 0; i < 3; i++) nn_linear_init(&m->l[i], &m->ps, i ? h : k, h, 1, NN_INIT_TORCH, 1.0, &m->r);
        nn_linear_init(&m->out, &m->ps, h, m->weight_model ? k : 1, 1, NN_INIT_TORCH, 1.0, &m->r);
        if (m->weight_model) {
            nn_layernorm_init(&m->ln, &m->ps, h);
            st = nn_batchnorm_init(&m->final_bn, &m->ps, 1);
            m->have_bn = 1;
        }
    }
    size_t psz = nn_params_size(&m->ps);
    double *best = NEW(double, psz), *best_rm = NEW(double, 2), best_auc = -INFINITY;
    if (st == PYOD_OK && (!best || !best_rm)) st = PYOD_ERR_NOMEM;
    ad_optimizer opt;
    int inited = 0;
    if (st == PYOD_OK) inited = (st = ad_opt_init(&opt, AD_ADAM, m->ps.p, m->ps.n, m->p.lr, m->p.wd, 0)) == PYOD_OK;
    for (size_t ep = 0; st == PYOD_OK && ep < m->p.n_epochs; ep++) {
        /* Keep the epoch with the best validation AUC (evaluated before each step). */
        ad_tensor *vo = lunar_fwd(m, ad_from_rows(&allv, NULL, nv), 0);
        memcpy(vs, vo->data, nv * sizeof(double));
        ad_tape_clear();
        double auc = pyod_roc_auc(vy, vs, nv);
        if (auc >= best_auc) {
            best_auc = auc;
            nn_snapshot(&m->ps, best);
            if (m->have_bn) best_rm[0] = m->final_bn.running_mean[0], best_rm[1] = m->final_bn.running_var[0];
        }
        ad_tensor *pred = lunar_fwd(m, ad_from_rows(&all_d, NULL, nt), 1);
        ad_tensor *target = ad_const(pred->ndim, pred->shape, ty);
        ad_zero_grad(m->ps.p, m->ps.n);
        ad_backward(ad_sum(ad_pow_s(ad_sub(pred, target), 2.0)));
        ad_tape_clear();
        ad_opt_step(&opt);
    }
    if (inited) ad_opt_free(&opt);
    if (st == PYOD_OK) {
        nn_restore(&m->ps, best);
        if (m->have_bn) m->final_bn.running_mean[0] = best_rm[0], m->final_bn.running_var[0] = best_rm[1];
    }
    pyod_matrix *mats[] = {&tr, &va, &ntr, &nva, &trd, &ntrd, &vad, &nvad, &all_d, &allv};
    for (size_t i = 0; i < sizeof mats / sizeof mats[0]; i++) pyod_matrix_free(mats[i]);
    free(ty), free(vs), free(vy), free(best), free(best_rm);
    CHECK(st);
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    if ((st = lunar_scores(m, x, s)) != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(dt, s, n, m->p.contamination);
}

static pyod_status lunar_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return lunar_scores(d->impl, x, out); }

static void lunar_destroy(void *p) {
    lunar_reset(p);
    free(p);
}

static const pyod_vtable LUNAR_VT = {"LUNAR", lunar_fit, NULL, lunar_score, lunar_destroy};

pyod_lunar_params pyod_lunar_defaults(void) {
    return (pyod_lunar_params){PYOD_LUNAR_WEIGHT, 5, 200, PYOD_NEG_MIXED, 0.1, 0.1, 1.0, 1e-3, 0.1, 0.1, 1, 42};
}

pyod_detector *pyod_lunar_create(const pyod_lunar_params *p) {
    lunar_impl *m = NEW(lunar_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_lunar_defaults();
    return detector_new(&LUNAR_VT, m);
}
