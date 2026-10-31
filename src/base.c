/* Port of pyod.models.base.BaseDetector, plus data utilities and score
 * combination. */
#include "internal.h"

const char *pyod_strerror(pyod_status s) {
    switch (s) {
    case PYOD_OK: return "ok";
    case PYOD_ERR_NOT_FITTED: return "detector has not been fitted";
    case PYOD_ERR_EMPTY_INPUT: return "input contains no samples";
    case PYOD_ERR_DIMENSION: return "input has the wrong number of features";
    case PYOD_ERR_PARAM: return "invalid parameter";
    case PYOD_ERR_UNSUPPORTED: return "operation not supported by this detector";
    case PYOD_ERR_NOMEM: return "out of memory";
    case PYOD_ERR_NUMERIC: return "numerical failure";
    }
    return "unknown error";
}

pyod_matrix pyod_matrix_view(double *data, size_t rows, size_t cols) {
    pyod_matrix m = {data, rows, cols};
    return m;
}

pyod_status pyod_matrix_alloc(pyod_matrix *m, size_t rows, size_t cols) {
    m->rows = rows;
    m->cols = cols;
    m->data = NEW(double, rows * cols);
    return m->data ? PYOD_OK : PYOD_ERR_NOMEM;
}

void pyod_matrix_free(pyod_matrix *m) {
    if (!m) return;
    free(m->data);
    m->data = NULL;
    m->rows = m->cols = 0;
}

pyod_status mat_copy(const pyod_matrix *x, pyod_matrix *out) {
    CHECK(pyod_matrix_alloc(out, x->rows, x->cols));
    memcpy(out->data, x->data, x->rows * x->cols * sizeof(double));
    return PYOD_OK;
}

pyod_status mat_rows(const pyod_matrix *x, const size_t *idx, size_t k, pyod_matrix *out) {
    CHECK(pyod_matrix_alloc(out, k, x->cols));
    for (size_t i = 0; i < k; i++) memcpy(&PYOD_AT(out, i, 0), &PYOD_AT(x, idx[i], 0), x->cols * sizeof(double));
    return PYOD_OK;
}

pyod_status mat_cols(const pyod_matrix *x, const size_t *cols, size_t k, pyod_matrix *out) {
    CHECK(pyod_matrix_alloc(out, x->rows, k));
    for (size_t i = 0; i < x->rows; i++)
        for (size_t j = 0; j < k; j++) PYOD_AT(out, i, j) = PYOD_AT(x, i, cols[j]);
    return PYOD_OK;
}

pyod_detector *detector_new(const pyod_vtable *vt, void *impl) {
    if (!impl) return NULL;
    pyod_detector *d = NEW(pyod_detector, 1);
    if (!d) {
        vt->destroy(impl);
        return NULL;
    }
    d->vt = vt;
    d->impl = impl;
    d->state.threshold = NAN;
    return d;
}

void fit_state_clear(pyod_fit_state *s) {
    free(s->scores);
    free(s->labels);
    memset(s, 0, sizeof(*s));
    s->threshold = NAN;
}

pyod_status fit_state_threshold(pyod_detector *d, double *scores, size_t n, double threshold) {
    fit_state_clear(&d->state);
    uint8_t *labels = NEW(uint8_t, n);
    if (!labels) {
        free(scores);
        return PYOD_ERR_NOMEM;
    }
    size_t flagged = 0;
    for (size_t i = 0; i < n; i++) flagged += labels[i] = scores[i] > threshold;
    d->state.scores = scores;
    d->state.labels = labels;
    d->state.n = n;
    d->state.threshold = threshold;
    d->state.mu = vec_mean(scores, n);
    d->state.sigma = vec_std(scores, n);
    d->state.contamination = n ? (double)flagged / (double)n : 0.0;
    d->fitted = 1;
    return PYOD_OK;
}

pyod_status fit_state_contamination(pyod_detector *d, double *scores, size_t n, double contamination) {
    double th = vec_percentile(scores, n, 100.0 * (1.0 - contamination));
    CHECK(fit_state_threshold(d, scores, n, th));
    d->state.contamination = contamination;
    return PYOD_OK;
}

pyod_status check_matrix(const pyod_matrix *x) {
    if (!x || !x->data || x->rows == 0 || x->cols == 0) return PYOD_ERR_EMPTY_INPUT;
    for (size_t i = 0; i < x->rows * x->cols; i++)
        if (!isfinite(x->data[i])) return PYOD_ERR_PARAM;
    return PYOD_OK;
}

pyod_status check_contamination(double c) {
    return (c > 0.0 && c <= 0.5) ? PYOD_OK : PYOD_ERR_PARAM;
}

pyod_status pyod_fit(pyod_detector *d, const pyod_matrix *x) {
    if (!d) return PYOD_ERR_PARAM;
    CHECK(check_matrix(x));
    if (!d->vt->fit) return PYOD_ERR_UNSUPPORTED;
    d->fitted = 0;
    d->n_features = x->cols;
    pyod_status st = d->vt->fit(d, x);
    if (st != PYOD_OK) {
        d->fitted = 0;
        fit_state_clear(&d->state);
    }
    return st;
}

pyod_status pyod_fit_supervised(pyod_detector *d, const pyod_matrix *x, const uint8_t *y) {
    if (!d || !y) return PYOD_ERR_PARAM;
    CHECK(check_matrix(x));
    if (!d->vt->fit_supervised) return PYOD_ERR_UNSUPPORTED;
    d->fitted = 0;
    d->n_features = x->cols;
    pyod_status st = d->vt->fit_supervised(d, x, y);
    if (st != PYOD_OK) {
        d->fitted = 0;
        fit_state_clear(&d->state);
    }
    return st;
}

pyod_status pyod_decision_function(const pyod_detector *d, const pyod_matrix *x, double *out) {
    if (!d || !out) return PYOD_ERR_PARAM;
    if (!d->fitted) return PYOD_ERR_NOT_FITTED;
    CHECK(check_matrix(x));
    if (d->n_features && x->cols != d->n_features) return PYOD_ERR_DIMENSION;
    if (!d->vt->score) return PYOD_ERR_UNSUPPORTED;
    return d->vt->score(d, x, out);
}

pyod_status pyod_predict(const pyod_detector *d, const pyod_matrix *x, uint8_t *labels) {
    if (!labels || !x) return PYOD_ERR_PARAM;
    double *s = NEW(double, x->rows);
    if (!s) return PYOD_ERR_NOMEM;
    pyod_status st = pyod_decision_function(d, x, s);
    if (st == PYOD_OK)
        for (size_t i = 0; i < x->rows; i++) labels[i] = s[i] > d->state.threshold;
    free(s);
    return st;
}

pyod_status pyod_predict_proba(const pyod_detector *d, const pyod_matrix *x, pyod_proba_method m, double *out) {
    CHECK(pyod_decision_function(d, x, out));
    const pyod_fit_state *s = &d->state;
    if (m == PYOD_PROBA_LINEAR) {
        double lo = vec_min(s->scores, s->n), hi = vec_max(s->scores, s->n);
        double range = hi - lo == 0.0 ? 1.0 : hi - lo;
        for (size_t i = 0; i < x->rows; i++) out[i] = fmin(1.0, fmax(0.0, (out[i] - lo) / range));
    } else {
        for (size_t i = 0; i < x->rows; i++)
            out[i] = fmin(1.0, fmax(0.0, erf_((out[i] - s->mu) / (s->sigma * sqrt(2.0)))));
    }
    return PYOD_OK;
}

/* Binomial CDF P(X <= k), X ~ Bin(n, p), in log space. */
static double binom_cdf(uint64_t k, uint64_t n, double p) {
    if (p <= 0.0) return 1.0;
    if (p >= 1.0) return k >= n ? 1.0 : 0.0;
    double lp = log(p), lq = log(1.0 - p), log_coef = 0.0, total = 0.0;
    for (uint64_t i = 0; i <= (k < n ? k : n); i++) {
        if (i > 0) log_coef += log((double)(n - i + 1)) - log((double)i);
        total += exp(log_coef + (double)i * lp + (double)(n - i) * lq);
    }
    return total < 1.0 ? total : 1.0;
}

pyod_status pyod_predict_confidence(const pyod_detector *d, const pyod_matrix *x, double *out) {
    CHECK(pyod_decision_function(d, x, out));
    const pyod_fit_state *s = &d->state;
    double *train = NEW(double, s->n);
    if (!train) return PYOD_ERR_NOMEM;
    memcpy(train, s->scores, s->n * sizeof(double));
    qsort(train, s->n, sizeof(double), cmp_double);
    double n = (double)s->n;
    uint64_t k = s->n - (uint64_t)(n * s->contamination);
    for (size_t i = 0; i < x->rows; i++) {
        size_t lo = 0, hi = s->n; /* count of train <= score */
        while (lo < hi) {
            size_t mid = (lo + hi) / 2;
            if (train[mid] <= out[i]) lo = mid + 1;
            else hi = mid;
        }
        double p = (1.0 + (double)lo) / (2.0 + n);
        double conf = 1.0 - binom_cdf(k, s->n, p);
        out[i] = out[i] > s->threshold ? conf : 1.0 - conf;
    }
    free(train);
    return PYOD_OK;
}

pyod_status pyod_fit_predict(pyod_detector *d, const pyod_matrix *x, uint8_t *labels) {
    CHECK(pyod_fit(d, x));
    memcpy(labels, d->state.labels, d->state.n);
    return PYOD_OK;
}

const double *pyod_decision_scores(const pyod_detector *d, size_t *n) {
    if (!d || !d->fitted) return NULL;
    if (n) *n = d->state.n;
    return d->state.scores;
}

const uint8_t *pyod_labels(const pyod_detector *d, size_t *n) {
    if (!d || !d->fitted) return NULL;
    if (n) *n = d->state.n;
    return d->state.labels;
}

double pyod_threshold(const pyod_detector *d) { return d && d->fitted ? d->state.threshold : NAN; }

const char *pyod_name(const pyod_detector *d) { return d ? d->vt->name : NULL; }

void pyod_free(pyod_detector *d) {
    if (!d) return;
    if (d->vt->destroy) d->vt->destroy(d->impl);
    fit_state_clear(&d->state);
    free(d);
}

pyod_status pyod_apply_thresholder(const pyod_detector *d, pyod_thresholder t, void *ctx, double *threshold,
                                   uint8_t *labels) {
    if (!d || !t) return PYOD_ERR_PARAM;
    if (!d->fitted) return PYOD_ERR_NOT_FITTED;
    double th = t(d->state.scores, d->state.n, ctx);
    if (threshold) *threshold = th;
    if (labels)
        for (size_t i = 0; i < d->state.n; i++) labels[i] = d->state.scores[i] > th;
    return PYOD_OK;
}

/* ---------------- data utilities ---------------- */

pyod_status pyod_generate_data(size_t n, size_t n_features, double contamination, uint64_t seed, pyod_matrix *x,
                               uint8_t *y) {
    if (n == 0 || n_features == 0) return PYOD_ERR_PARAM;
    CHECK(pyod_matrix_alloc(x, n, n_features));
    pyod_rng r;
    rng_seed(&r, seed);
    size_t n_out = (size_t)((double)n * contamination);
    for (size_t i = 0; i < n - n_out; i++) {
        for (size_t j = 0; j < n_features; j++) {
            double z = 0.3 * rng_normal(&r);
            PYOD_AT(x, i, j) = z + 2.0 * ((double)rng_below(&r, 2) * 2.0 - 1.0);
        }
        y[i] = 0;
    }
    for (size_t i = n - n_out; i < n; i++) {
        for (size_t j = 0; j < n_features; j++) PYOD_AT(x, i, j) = rng_f64(&r) * 12.0 - 6.0;
        y[i] = 1;
    }
    return PYOD_OK;
}

double pyod_roc_auc(const uint8_t *y, const double *scores, size_t n) {
    double wins = 0.0, pos = 0.0, neg = 0.0;
    for (size_t i = 0; i < n; i++) *(y[i] ? &pos : &neg) += 1.0;
    for (size_t i = 0; i < n; i++) {
        if (!y[i]) continue;
        for (size_t j = 0; j < n; j++) {
            if (y[j]) continue;
            wins += scores[i] > scores[j] ? 1.0 : scores[i] == scores[j] ? 0.5 : 0.0;
        }
    }
    return pos > 0 && neg > 0 ? wins / (pos * neg) : NAN;
}

double pyod_precision_n(const uint8_t *y, const double *scores, size_t n) {
    size_t k = 0, hit = 0;
    for (size_t i = 0; i < n; i++) k += y[i];
    if (!k) return NAN;
    size_t *idx = NEW(size_t, n);
    if (!idx) return NAN;
    argsort(scores, n, idx);
    for (size_t i = 0; i < k; i++) hit += y[idx[n - 1 - i]];
    free(idx);
    return (double)hit / (double)k;
}

void pyod_comb_average(const pyod_matrix *s, const double *w, double *out) {
    for (size_t i = 0; i < s->rows; i++) {
        double acc = 0.0, wsum = 0.0;
        for (size_t j = 0; j < s->cols; j++) {
            double wj = w ? w[j] : 1.0;
            acc += PYOD_AT(s, i, j) * wj;
            wsum += wj;
        }
        out[i] = acc / wsum;
    }
}

void pyod_comb_maximization(const pyod_matrix *s, double *out) {
    for (size_t i = 0; i < s->rows; i++) out[i] = vec_max(&PYOD_AT(s, i, 0), s->cols);
}

void pyod_comb_median(const pyod_matrix *s, double *out) {
    for (size_t i = 0; i < s->rows; i++) out[i] = vec_median(&PYOD_AT(s, i, 0), s->cols);
}

static pyod_status bucketed(const pyod_matrix *s, size_t nb, int inner_max, double *out) {
    if (nb == 0 || nb > s->cols) return PYOD_ERR_PARAM;
    size_t per = s->cols / nb;
    double *vals = NEW(double, nb);
    if (!vals) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < s->rows; i++) {
        for (size_t b = 0; b < nb; b++) {
            size_t lo = b * per, hi = b == nb - 1 ? s->cols : (b + 1) * per;
            const double *row = &PYOD_AT(s, i, lo);
            vals[b] = inner_max ? vec_max(row, hi - lo) : vec_mean(row, hi - lo);
        }
        out[i] = inner_max ? vec_mean(vals, nb) : vec_max(vals, nb);
    }
    free(vals);
    return PYOD_OK;
}

pyod_status pyod_comb_aom(const pyod_matrix *s, size_t nb, double *out) { return bucketed(s, nb, 1, out); }
pyod_status pyod_comb_moa(const pyod_matrix *s, size_t nb, double *out) { return bucketed(s, nb, 0, out); }

void pyod_comb_majority_vote(const uint8_t *labels, size_t n, size_t k, uint8_t *out) {
    for (size_t i = 0; i < n; i++) {
        size_t c = 0;
        for (size_t j = 0; j < k; j++) c += labels[i * k + j];
        out[i] = c * 2 > k;
    }
}

void pyod_standardize(pyod_matrix *x) {
    std_scaler s;
    if (scaler_fit(&s, x) != PYOD_OK) return;
    scaler_transform(&s, x, x->data);
    scaler_free(&s);
}
