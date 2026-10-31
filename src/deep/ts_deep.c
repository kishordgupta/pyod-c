/* LSTM-AD (pyod.models.ts_lstm), Anomaly Transformer
 * (pyod.models.ts_anomaly_transformer) and AudioAE (pyod.models.audio_ae). */
#include "deep/common.h"

/* Windows starting at `starts` (len w) of x, as a [n, w, c] tensor. */
static ad_tensor *windows3(const pyod_matrix *x, const size_t *starts, size_t n, size_t w) {
    size_t c = x->cols;
    ad_tensor *t = ad_const(3, AD_SHAPE(n, w, c), NULL);
    for (size_t i = 0; i < n; i++) memcpy(t->data + i * w * c, &PYOD_AT(x, starts[i], 0), w * c * sizeof(double));
    return t;
}

/* ---------------- LSTM-AD ---------------- */

typedef struct {
    pyod_lstm_ad_params p;
    nn_params ps;
    nn_lstm *layers;
    nn_linear out;
    double *mu, *prec;
    size_t c;
    pyod_rng r;
} lstm_impl;

static void lstm_reset(lstm_impl *m) {
    nn_params_free(&m->ps);
    free(m->layers), free(m->mu), free(m->prec);
    m->layers = NULL, m->mu = m->prec = NULL;
}

static ad_tensor *lstm_fwd(const lstm_impl *m, ad_tensor *x) {
    for (size_t i = 0; i < m->p.n_layers; i++) x = nn_lstm_fwd(&m->layers[i], x);
    size_t B = x->shape[0], T = x->shape[1], H = x->shape[2];
    return nn_linear_fwd(&m->out, ad_reshape(ad_narrow(x, 1, T - 1, 1), 2, AD_SHAPE(B, H)));
}

/* Forecast errors for every timestamp after the first window: (n - w) x c. */
static pyod_status lstm_errors(const lstm_impl *m, const pyod_matrix *x, pyod_matrix *err) {
    size_t w = m->p.window_size, np = x->rows - w, c = x->cols;
    CHECK(pyod_matrix_alloc(err, np, c));
    size_t *starts = NEW(size_t, np);
    if (!starts) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < np; i++) starts[i] = i;
    for (size_t s = 0; s < np; s += 256) {
        size_t len = np - s < 256 ? np - s : 256;
        ad_tensor *pred = lstm_fwd(m, windows3(x, starts + s, len, w));
        for (size_t i = 0; i < len; i++)
            for (size_t j = 0; j < c; j++) PYOD_AT(err, s + i, j) = PYOD_AT(x, s + i + w, j) - pred->data[i * c + j];
        ad_tape_clear();
    }
    free(starts);
    return PYOD_OK;
}

static pyod_status lstm_fit(pyod_detector *dt, const pyod_matrix *x) {
    lstm_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t w = m->p.window_size, c = x->cols, n = x->rows, H = m->p.hidden_size;
    if (w == 0 || m->p.n_layers == 0 || H == 0 || n < w + 10) return PYOD_ERR_PARAM;
    lstm_reset(m);
    m->c = c;
    rng_seed(&m->r, m->p.random_state);
    m->layers = NEW(nn_lstm, m->p.n_layers);
    if (!m->layers) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < m->p.n_layers; i++) nn_lstm_init(&m->layers[i], &m->ps, i ? H : c, H, &m->r);
    nn_linear_init(&m->out, &m->ps, H, c, 1, NN_INIT_TORCH, 1.0, &m->r);
    if (!m->out.w) return PYOD_ERR_NOMEM;
    size_t np = n - w, *perm = NEW(size_t, np);
    double *tgt = NEW(double, m->p.batch_size * c);
    ad_optimizer opt;
    pyod_status st = perm && tgt ? ad_opt_init(&opt, AD_ADAM, m->ps.p, m->ps.n, m->p.lr, 0, 0) : PYOD_ERR_NOMEM;
    for (size_t ep = 0; st == PYOD_OK && ep < m->p.epochs; ep++) {
        size_t nb = nn_batches(np, m->p.batch_size, 0, &m->r, perm);
        for (size_t b = 0; b < nb; b++) {
            size_t len = nn_batch_len(np, m->p.batch_size, b), *idx = perm + b * m->p.batch_size;
            for (size_t i = 0; i < len; i++) memcpy(tgt + i * c, &PYOD_AT(x, idx[i] + w, 0), c * sizeof(double));
            ad_tensor *pred = lstm_fwd(m, windows3(x, idx, len, w));
            ad_zero_grad(m->ps.p, m->ps.n);
            ad_backward(ad_mean(ad_pow_s(ad_sub(pred, ad_const(2, AD_SHAPE(len, c), tgt)), 2.0)));
            ad_tape_clear();
            ad_opt_step(&opt);
        }
    }
    if (perm && tgt && st == PYOD_OK) ad_opt_free(&opt);
    free(perm), free(tgt);
    CHECK(st);
    /* Mahalanobis distance of forecast errors (sample covariance + 1e-6 I). */
    pyod_matrix err;
    CHECK(lstm_errors(m, x, &err));
    m->mu = NEW(double, c);
    m->prec = NEW(double, c * c);
    double *cov = NEW(double, c * c), *s = NEW(double, n), *valid = NEW(double, np);
    st = m->mu && m->prec && cov && s && valid ? PYOD_OK : PYOD_ERR_NOMEM;
    if (st == PYOD_OK) {
        col_mean(&err, m->mu);
        covariance(&err, m->mu, 1, cov);
        for (size_t j = 0; j < c; j++) cov[j * c + j] += 1e-6;
        pinvh(cov, c, m->prec);
        mahalanobis_sq(&err, m->mu, m->prec, valid);
        for (size_t i = 0; i < n; i++) s[i] = i < w ? NAN : valid[i - w];
    }
    pyod_matrix_free(&err);
    free(cov), free(valid);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    return ts_fit_state(dt, s, n, m->p.contamination);
}

static pyod_status lstm_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    const lstm_impl *m = d->impl;
    size_t w = m->p.window_size;
    if (x->rows <= w) return PYOD_ERR_PARAM;
    pyod_matrix err;
    CHECK(lstm_errors(m, x, &err));
    for (size_t i = 0; i < w; i++) out[i] = d->state.threshold;
    mahalanobis_sq(&err, m->mu, m->prec, out + w);
    pyod_matrix_free(&err);
    return PYOD_OK;
}

static void lstm_destroy(void *p) {
    lstm_reset(p);
    free(p);
}

static const pyod_vtable LSTM_VT = {"LSTMAD", lstm_fit, NULL, lstm_score, lstm_destroy};

pyod_lstm_ad_params pyod_lstm_ad_defaults(void) { return (pyod_lstm_ad_params){50, 64, 2, 50, 32, 1e-3, 0.1, 42}; }

pyod_detector *pyod_lstm_ad_create(const pyod_lstm_ad_params *p) {
    lstm_impl *m = NEW(lstm_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_lstm_ad_defaults();
    return detector_new(&LSTM_VT, m);
}

/* ---------------- Anomaly Transformer ---------------- */

typedef struct {
    nn_linear q, k, v, o, ff1, ff2;
    ad_tensor *sigma;
    nn_layernorm n1, n2;
} at_layer;

typedef struct {
    pyod_anomaly_transformer_params p;
    nn_params ps;
    nn_linear embed, out;
    ad_tensor *pos;
    at_layer *layers;
    size_t c;
    pyod_rng r;
} at_impl;

typedef struct {
    ad_tensor *series, *prior;
} assoc;

static ad_tensor *at_layer_fwd(at_impl *m, at_layer *ly, ad_tensor *x, int train, assoc *as) {
    size_t B = x->shape[0], L = x->shape[1], D = x->shape[2], H = m->p.n_heads, dk = D / H;
    ad_tensor *q = ad_swap(ad_reshape(nn_linear_fwd(&ly->q, x), 4, AD_SHAPE(B, L, H, dk)), 1, 2);
    ad_tensor *k = ad_swap(ad_reshape(nn_linear_fwd(&ly->k, x), 4, AD_SHAPE(B, L, H, dk)), 1, 2);
    ad_tensor *v = ad_swap(ad_reshape(nn_linear_fwd(&ly->v, x), 4, AD_SHAPE(B, L, H, dk)), 1, 2);
    ad_tensor *series = ad_softmax(ad_mul_s(ad_matmul(q, ad_transpose(k)), 1.0 / sqrt((double)dk)));
    /* Prior association: Gaussian over |i - j| with a learnable width per head. */
    ad_tensor *dist = ad_const(4, AD_SHAPE(1, 1, L, L), NULL);
    for (size_t i = 0; i < L; i++)
        for (size_t j = 0; j < L; j++) dist->data[i * L + j] = ((double)i - (double)j) * ((double)i - (double)j);
    ad_tensor *s2 = ad_reshape(ad_pow_s(ad_clamp(ly->sigma, 1e-4, INFINITY), 2.0), 4, AD_SHAPE(1, H, 1, 1));
    ad_tensor *prior = ad_softmax(ad_div(ad_neg(dist), ad_mul_s(s2, 2.0)));
    as->series = series;
    as->prior = prior;
    ad_tensor *att = ad_reshape(ad_swap(ad_matmul(series, v), 1, 2), 3, AD_SHAPE(B, L, D));
    x = nn_layernorm_fwd(&ly->n1, ad_add(x, ad_dropout(nn_linear_fwd(&ly->o, att), m->p.dropout, train, &m->r)));
    ad_tensor *ff = ad_dropout(nn_linear_fwd(&ly->ff2, ad_dropout(ad_gelu(nn_linear_fwd(&ly->ff1, x)), m->p.dropout, train, &m->r)), m->p.dropout, train, &m->r);
    return nn_layernorm_fwd(&ly->n2, ad_add(x, ff));
}

static ad_tensor *at_fwd(at_impl *m, ad_tensor *x, int train, assoc *as) {
    ad_tensor *h = ad_add(nn_linear_fwd(&m->embed, ad_unfold3(x)), m->pos);
    for (size_t i = 0; i < m->p.n_layers; i++) h = at_layer_fwd(m, &m->layers[i], h, train, &as[i]);
    return nn_linear_fwd(&m->out, h);
}

/* Symmetric KL between series and prior associations, summed over layers: [B, L]. */
static ad_tensor *discrepancy(const at_impl *m, assoc *as, int detach_prior, int detach_series) {
    ad_tensor *total = NULL;
    for (size_t i = 0; i < m->p.n_layers; i++) {
        ad_tensor *s = detach_series ? ad_detach(as[i].series) : as[i].series;
        ad_tensor *p = detach_prior ? ad_detach(as[i].prior) : as[i].prior;
        ad_tensor *ls = ad_log(ad_add_s(s, 1e-8)), *lp = ad_log(ad_add_s(p, 1e-8));
        ad_tensor *kl = ad_add(ad_sum_dim(ad_mul(p, ad_sub(lp, ls)), 3), ad_sum_dim(ad_mul(s, ad_sub(ls, lp)), 3));
        size_t B = as[i].series->shape[0], L = as[i].series->shape[2];
        ad_tensor *layer = ad_reshape(ad_mean_dim(ad_mul_s(kl, 0.5), 1), 2, AD_SHAPE(B, L));
        total = total ? ad_add(total, layer) : layer;
    }
    return total;
}

static size_t at_starts(const at_impl *m, size_t n, size_t **starts) {
    size_t w = m->p.window_size, nw = (n - w) / m->p.step + 1;
    *starts = NEW(size_t, nw);
    if (!*starts) return 0;
    for (size_t i = 0; i < nw; i++) (*starts)[i] = i * m->p.step;
    return nw;
}

/* Per-timestamp scores averaged over covering windows (NAN where uncovered). */
static pyod_status at_scores(at_impl *m, const pyod_matrix *x, double *out) {
    size_t w = m->p.window_size, n = x->rows, *starts, nw = at_starts(m, n, &starts);
    double *acc = NEW(double, n);
    size_t *cnt = NEW(size_t, n);
    assoc *as = NEW(assoc, m->p.n_layers);
    if (!nw || !acc || !cnt || !as) {
        free(starts), free(acc), free(cnt), free(as);
        return PYOD_ERR_NOMEM;
    }
    for (size_t s = 0; s < nw; s += 64) {
        size_t len = nw - s < 64 ? nw - s : 64;
        ad_tensor *batch = windows3(x, starts + s, len, w);
        ad_tensor *rec = at_fwd(m, batch, 0, as);
        ad_tensor *err = ad_reshape(ad_mean_dim(ad_pow_s(ad_sub(batch, rec), 2.0), 2), 2, AD_SHAPE(len, w));
        ad_tensor *score = ad_mul(ad_softmax(ad_neg(discrepancy(m, as, 0, 0))), err);
        for (size_t b = 0; b < len; b++)
            for (size_t t = 0; t < w; t++) acc[starts[s + b] + t] += score->data[b * w + t], cnt[starts[s + b] + t]++;
        ad_tape_clear();
    }
    for (size_t i = 0; i < n; i++) out[i] = cnt[i] ? acc[i] / (double)cnt[i] : NAN;
    free(starts), free(acc), free(cnt), free(as);
    return PYOD_OK;
}

static void at_reset(at_impl *m) {
    nn_params_free(&m->ps);
    free(m->layers);
    m->layers = NULL;
}

static pyod_status at_fit(pyod_detector *dt, const pyod_matrix *x) {
    at_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t w = m->p.window_size, D = m->p.d_model, H = m->p.n_heads, c = x->cols, n = x->rows;
    if (w == 0 || m->p.step == 0 || n < w || H == 0 || D % H || m->p.n_layers == 0) return PYOD_ERR_PARAM;
    at_reset(m);
    m->c = c;
    rng_seed(&m->r, m->p.random_state);
    nn_linear_init(&m->embed, &m->ps, 3 * c, D, 1, NN_INIT_TORCH, 1.0, &m->r); /* Conv1d(k=3, pad=1) */
    m->pos = nn_new_param(&m->ps, 3, AD_SHAPE(1, w, D));
    m->layers = NEW(at_layer, m->p.n_layers);
    if (!m->embed.w || !m->pos || !m->layers) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < w * D; i++) m->pos->data[i] = 0.02 * rng_normal(&m->r);
    for (size_t i = 0; i < m->p.n_layers; i++) {
        at_layer *ly = &m->layers[i];
        nn_linear *lins[4] = {&ly->q, &ly->k, &ly->v, &ly->o};
        for (int t = 0; t < 4; t++) nn_linear_init(lins[t], &m->ps, D, D, 1, NN_INIT_TORCH, 1.0, &m->r);
        nn_linear_init(&ly->ff1, &m->ps, D, 4 * D, 1, NN_INIT_TORCH, 1.0, &m->r);
        nn_linear_init(&ly->ff2, &m->ps, 4 * D, D, 1, NN_INIT_TORCH, 1.0, &m->r);
        ly->sigma = nn_new_param(&m->ps, 1, AD_SHAPE(H));
        nn_layernorm_init(&ly->n1, &m->ps, D);
        nn_layernorm_init(&ly->n2, &m->ps, D);
        if (!ly->sigma || !ly->ff2.w || !ly->n2.beta) return PYOD_ERR_NOMEM;
        for (size_t h = 0; h < H; h++) ly->sigma->data[h] = 1.0;
    }
    nn_linear_init(&m->out, &m->ps, D, c, 1, NN_INIT_TORCH, 1.0, &m->r);
    if (!m->out.w) return PYOD_ERR_NOMEM;
    size_t *starts, nw = at_starts(m, n, &starts), *perm = NEW(size_t, nw ? nw : 1), *bst = NEW(size_t, m->p.batch_size);
    assoc *as = NEW(assoc, m->p.n_layers);
    ad_optimizer opt;
    pyod_status st = nw && perm && bst && as ? ad_opt_init(&opt, AD_ADAM, m->ps.p, m->ps.n, m->p.lr, 0, 0) : PYOD_ERR_NOMEM;
    for (size_t ep = 0; st == PYOD_OK && ep < m->p.epochs; ep++) {
        size_t nb = nn_batches(nw, m->p.batch_size, 0, &m->r, perm);
        for (size_t b = 0; b < nb; b++) {
            size_t len = nn_batch_len(nw, m->p.batch_size, b);
            for (size_t i = 0; i < len; i++) bst[i] = starts[perm[b * m->p.batch_size + i]];
            ad_tensor *batch = windows3(x, bst, len, w);
            /* Minimise phase: reconstruct, and enlarge the discrepancy against a fixed prior. */
            ad_tensor *rec = at_fwd(m, batch, 1, as);
            ad_tensor *loss = ad_sub(ad_mean(ad_pow_s(ad_sub(rec, batch), 2.0)), ad_mul_s(ad_mean(discrepancy(m, as, 1, 0)), m->p.lambda));
            ad_zero_grad(m->ps.p, m->ps.n);
            ad_backward(loss);
            ad_opt_step(&opt);
            /* Maximise phase: move the prior toward the (fixed) series association. */
            at_fwd(m, batch, 1, as);
            ad_zero_grad(m->ps.p, m->ps.n);
            ad_backward(ad_neg(ad_mean(discrepancy(m, as, 0, 1))));
            ad_opt_step(&opt);
            ad_tape_clear();
        }
    }
    if (nw && perm && bst && as && st == PYOD_OK) ad_opt_free(&opt);
    free(starts), free(perm), free(bst), free(as);
    CHECK(st);
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    if ((st = at_scores(m, x, s)) != PYOD_OK) {
        free(s);
        return st;
    }
    return ts_fit_state(dt, s, n, m->p.contamination);
}

static pyod_status at_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    at_impl *m = d->impl;
    if (x->rows < m->p.window_size) return PYOD_ERR_PARAM;
    CHECK(at_scores(m, x, out));
    ts_fill(out, x->rows, d->state.threshold);
    return PYOD_OK;
}

static void at_destroy(void *p) {
    at_reset(p);
    free(p);
}

static const pyod_vtable AT_VT = {"AnomalyTransformer", at_fit, NULL, at_score, at_destroy};

pyod_anomaly_transformer_params pyod_anomaly_transformer_defaults(void) {
    return (pyod_anomaly_transformer_params){100, 512, 8, 3, 10, 32, 1, 1e-4, 3.0, 0.1, 0.1, 42};
}

pyod_detector *pyod_anomaly_transformer_create(const pyod_anomaly_transformer_params *p) {
    at_impl *m = NEW(at_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_anomaly_transformer_defaults();
    return detector_new(&AT_VT, m);
}

/* ---------------- AudioAE ---------------- */

struct pyod_audio_ae {
    pyod_audio_ae_params p;
    pyod_detector *ae;
    double *scores;
    size_t n;
    double threshold;
};

/* All context windows of all clips (frame-major), and each window's clip. */
static pyod_status audio_windows(const pyod_audio_ae *a, const double *const *clips, const size_t *lens, size_t n,
                                 pyod_matrix *out, size_t **owner) {
    size_t total = 0, dim = a->p.n_mels * a->p.context, cap = 0;
    pyod_matrix *specs = NEW(pyod_matrix, n);
    if (!specs) return PYOD_ERR_NOMEM;
    pyod_status st = PYOD_OK;
    for (size_t c = 0; c < n && st == PYOD_OK; c++) {
        if (!clips[c] || !lens[c]) st = PYOD_ERR_EMPTY_INPUT;
        else st = pyod_log_mel_spectrogram(clips[c], lens[c], a->p.sr, a->p.n_mels, a->p.hop_length, &specs[c]);
        if (st == PYOD_OK) total += specs[c].cols >= a->p.context ? specs[c].cols - a->p.context + 1 : 1;
    }
    if (st == PYOD_OK) st = pyod_matrix_alloc(out, total, dim);
    *owner = st == PYOD_OK ? NEW(size_t, total) : NULL;
    if (st == PYOD_OK && !*owner) st = PYOD_ERR_NOMEM;
    for (size_t c = 0; c < n && st == PYOD_OK; c++) {
        size_t frames = specs[c].cols, nw = frames >= a->p.context ? frames - a->p.context + 1 : 1;
        for (size_t t = 0; t < nw; t++, cap++) {
            (*owner)[cap] = c;
            for (size_t f = 0; f < a->p.context; f++) /* short spectrograms are zero-padded */
                for (size_t mel = 0; mel < a->p.n_mels; mel++)
                    PYOD_AT(out, cap, f * a->p.n_mels + mel) = t + f < frames ? PYOD_AT(&specs[c], mel, t + f) : 0.0;
        }
    }
    for (size_t c = 0; c < n; c++) pyod_matrix_free(&specs[c]);
    free(specs);
    return st;
}

static void aggregate(const double *s, const size_t *owner, size_t m, size_t n, double *out) {
    size_t *cnt = NEW(size_t, n);
    for (size_t i = 0; i < n; i++) out[i] = 0.0;
    if (!cnt) return;
    for (size_t i = 0; i < m; i++) out[owner[i]] += s[i], cnt[owner[i]]++;
    for (size_t i = 0; i < n; i++) out[i] = cnt[i] ? out[i] / (double)cnt[i] : 0.0;
    free(cnt);
}

pyod_audio_ae_params pyod_audio_ae_defaults(void) {
    pyod_audio_ae_params p;
    memset(&p, 0, sizeof p);
    p.n_mels = 64;
    p.context = 5;
    p.hop_length = 512;
    p.epochs = 40;
    p.batch_size = 1024;
    size_t h[3] = {128, 32, 8};
    memcpy(p.hidden_neurons, h, sizeof h);
    p.sr = 22050.0;
    p.lr = 1e-3;
    p.contamination = 0.1;
    p.random_state = 42;
    return p;
}

pyod_audio_ae *pyod_audio_ae_create(const pyod_audio_ae_params *p) {
    pyod_audio_ae *a = NEW(pyod_audio_ae, 1);
    if (!a) return NULL;
    a->p = p ? *p : pyod_audio_ae_defaults();
    a->threshold = NAN;
    return a;
}

pyod_status pyod_audio_ae_fit(pyod_audio_ae *a, const double *const *clips, const size_t *lens, size_t n) {
    if (!a || !clips || !lens || n == 0) return PYOD_ERR_EMPTY_INPUT;
    CHECK(check_contamination(a->p.contamination));
    pyod_matrix frames;
    size_t *owner;
    CHECK(audio_windows(a, clips, lens, n, &frames, &owner));
    pyod_autoencoder_params ap = pyod_autoencoder_defaults();
    memset(ap.hidden_neurons, 0, sizeof ap.hidden_neurons);
    size_t nh = 0, dim = frames.cols;
    for (size_t i = 0; i < layer_count(a->p.hidden_neurons); i++)
        if (a->p.hidden_neurons[i] < dim) ap.hidden_neurons[nh++] = a->p.hidden_neurons[i];
    if (!nh) ap.hidden_neurons[0] = dim / 2 > 2 ? dim / 2 : 2;
    ap.train.epochs = a->p.epochs;
    ap.train.batch_size = a->p.batch_size < frames.rows ? a->p.batch_size : frames.rows;
    ap.train.lr = a->p.lr;
    ap.train.random_state = a->p.random_state;
    ap.contamination = a->p.contamination;
    pyod_free(a->ae);
    free(a->scores);
    a->scores = NULL;
    a->ae = pyod_autoencoder_create(&ap);
    pyod_status st = a->ae ? pyod_fit(a->ae, &frames) : PYOD_ERR_NOMEM;
    a->scores = NEW(double, n);
    if (st == PYOD_OK && !a->scores) st = PYOD_ERR_NOMEM;
    if (st == PYOD_OK) {
        aggregate(pyod_decision_scores(a->ae, NULL), owner, frames.rows, n, a->scores);
        a->n = n;
        a->threshold = vec_percentile(a->scores, n, 100.0 * (1.0 - a->p.contamination));
    }
    pyod_matrix_free(&frames);
    free(owner);
    return st;
}

pyod_status pyod_audio_ae_decision_function(const pyod_audio_ae *a, const double *const *clips, const size_t *lens,
                                            size_t n, double *out) {
    if (!a || !a->scores) return PYOD_ERR_NOT_FITTED;
    pyod_matrix frames;
    size_t *owner;
    CHECK(audio_windows(a, clips, lens, n, &frames, &owner));
    double *s = NEW(double, frames.rows);
    pyod_status st = s ? pyod_decision_function(a->ae, &frames, s) : PYOD_ERR_NOMEM;
    if (st == PYOD_OK) aggregate(s, owner, frames.rows, n, out);
    free(s), free(owner);
    pyod_matrix_free(&frames);
    return st;
}

const double *pyod_audio_ae_scores(const pyod_audio_ae *a, size_t *n) {
    if (!a || !a->scores) return NULL;
    if (n) *n = a->n;
    return a->scores;
}

double pyod_audio_ae_threshold(const pyod_audio_ae *a) { return a ? a->threshold : NAN; }

void pyod_audio_ae_free(pyod_audio_ae *a) {
    if (!a) return;
    pyod_free(a->ae);
    free(a->scores), free(a);
}
