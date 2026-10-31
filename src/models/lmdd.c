/* LMDD: linear method for deviation detection (Arning et al., 1996). Port of
 * pyod.models.lmdd. Transductive: samples are scored against each other. */
#include "internal.h"

typedef struct {
    pyod_lmdd_params p;
} lmdd_impl;

/* Dissimilarity over all entries of the given rows. */
static double dissim(pyod_lmdd_measure kind, const pyod_matrix *x, const size_t *rows, size_t k, double *buf) {
    size_t d = x->cols, m = 0;
    for (size_t t = 0; t < k; t++)
        for (size_t j = 0; j < d; j++) buf[m++] = PYOD_AT(x, rows[t], j);
    double mean = vec_mean(buf, m), acc = 0.0;
    if (kind == PYOD_LMDD_IQR) return vec_percentile(buf, m, 75.0) - vec_percentile(buf, m, 25.0);
    for (size_t i = 0; i < m; i++) acc += kind == PYOD_LMDD_AAD ? fabs(buf[i] - mean) : (buf[i] - mean) * (buf[i] - mean);
    return acc / (double)m;
}

/* Port of __dis: smoothing factor of each row in the order `ord`. */
static void dis(const lmdd_impl *m, const pyod_matrix *x, const size_t *ord, double *res, size_t *tmp, double *buf) {
    size_t n = x->rows;
    for (size_t i = 0; i < n; i++) res[i] = 0.0;
    double var_max = -INFINITY;
    size_t j = 0;
    for (size_t i = 1; i < n; i++) {
        double v = dissim(m->p.dis_measure, x, ord, i + 1, buf) - dissim(m->p.dis_measure, x, ord, i, buf);
        if (v > var_max) var_max = v, j = i;
    }
    if (var_max > res[j]) {
        res[j] = var_max;
        for (size_t k = j + 1; k < n; k++) {
            memcpy(tmp, ord, j * sizeof(size_t));
            tmp[j] = ord[k];
            double a = dissim(m->p.dis_measure, x, tmp, j + 1, buf);
            memcpy(tmp, ord, (j + 1) * sizeof(size_t));
            tmp[j + 1] = ord[k];
            double diff = a - dissim(m->p.dis_measure, x, tmp, j + 2, buf);
            if (diff >= 0) res[k] = diff + var_max;
        }
    }
}

static pyod_status lmdd(const lmdd_impl *m, const pyod_matrix *x, double *out) {
    size_t n = x->rows;
    double *dv = NEW(double, n), *card = NEW(double, n), *res = NEW(double, n), *buf = NEW(double, (n + 1) * x->cols);
    size_t *perm = NEW(size_t, n), *tmp = NEW(size_t, n + 1);
    if (!dv || !card || !res || !buf || !perm || !tmp) {
        free(dv), free(card), free(res), free(buf), free(perm), free(tmp);
        return PYOD_ERR_NOMEM;
    }
    pyod_rng r;
    rng_seed(&r, m->p.random_state);
    for (size_t it = 0; it < m->p.n_iter; it++) {
        rng_sample(&r, n, n, perm);
        dis(m, x, perm, res, tmp, buf);
        size_t pos = 0;
        for (size_t i = 0; i < n; i++) pos += res[i] > 0.0;
        double cur_card = (double)(n - pos);
        for (size_t t = 0; t < n; t++)
            if (res[t] > dv[perm[t]]) dv[perm[t]] = res[t], card[perm[t]] = cur_card;
    }
    for (size_t i = 0; i < n; i++) out[i] = dv[i] * card[i];
    free(dv), free(card), free(res), free(buf), free(perm), free(tmp);
    return PYOD_OK;
}

static pyod_status lmdd_fit(pyod_detector *d, const pyod_matrix *x) {
    lmdd_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    if (m->p.n_iter == 0) return PYOD_ERR_PARAM;
    double *s = NEW(double, x->rows);
    if (!s) return PYOD_ERR_NOMEM;
    pyod_status st = lmdd(m, x, s);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(d, s, x->rows, m->p.contamination);
}

static pyod_status lmdd_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return lmdd(d->impl, x, out); }

static const pyod_vtable LMDD_VT = {"LMDD", lmdd_fit, NULL, lmdd_score, free};

pyod_lmdd_params pyod_lmdd_defaults(void) { return (pyod_lmdd_params){50, PYOD_LMDD_AAD, 0.1, 0}; }

pyod_detector *pyod_lmdd_create(const pyod_lmdd_params *p) {
    lmdd_impl *m = NEW(lmdd_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_lmdd_defaults();
    return detector_new(&LMDD_VT, m);
}
