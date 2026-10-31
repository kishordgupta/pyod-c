/* SOD: subspace outlier detection (Kriegel et al., 2009). Port of
 * pyod.models.sod. Transductive: samples are scored against each other. */
#include "internal.h"

typedef struct {
    pyod_sod_params p;
} sod_impl;

static pyod_status sod(const sod_impl *m, const pyod_matrix *x, double *out) {
    size_t n = x->rows, d = x->cols, k = m->p.n_neighbors, rs = m->p.ref_set;
    if (rs == 0 || rs > k || k >= n) return PYOD_ERR_PARAM;
    uint8_t *knn = NEW(uint8_t, n * n);
    size_t *nbr = NEW(size_t, n * k), *order = NEW(size_t, n);
    uint32_t *shared = NEW(uint32_t, n);
    double *means = NEW(double, d), *var = NEW(double, d);
    if (!knn || !nbr || !order || !shared || !means || !var) {
        free(knn), free(nbr), free(order), free(shared), free(means), free(var);
        return PYOD_ERR_NOMEM;
    }
    for (size_t i = 0; i < n; i++) {
        k_nearest(x, &PYOD_AT(x, i, 0), k, i, nbr + i * k, NULL);
        for (size_t t = 0; t < k; t++) knn[i * n + nbr[i * k + t]] = 1;
    }
    for (size_t i = 0; i < n; i++) {
        /* Reference set: points sharing the most neighbours with i, ranked with
         * PyOD's numba argsort so ties resolve identically. */
        for (size_t j = 0; j < n; j++) {
            uint32_t c = 0;
            for (size_t t = 0; t < k; t++) c += knn[i * n + nbr[j * k + t]];
            shared[j] = c;
        }
        shared[i] = UINT32_MAX;
        numba_argsort_u32(shared, n, order);
        for (size_t j = 0; j < d; j++) means[j] = var[j] = 0.0;
        for (size_t t = 0; t < rs; t++) {
            const double *r = &PYOD_AT(x, order[n - 2 - t], 0);
            for (size_t j = 0; j < d; j++) means[j] += r[j] / (double)rs;
        }
        for (size_t t = 0; t < rs; t++) {
            const double *r = &PYOD_AT(x, order[n - 2 - t], 0);
            for (size_t j = 0; j < d; j++) var[j] += (r[j] - means[j]) * (r[j] - means[j]) / (double)rs;
        }
        double var_total = 0.0;
        for (size_t j = 0; j < d; j++) var_total += var[j];
        double expect = m->p.alpha * var_total / (double)d, acc = 0.0;
        size_t rel = 0;
        for (size_t j = 0; j < d; j++)
            if (var[j] < expect) {
                double diff = PYOD_AT(x, i, j) - means[j];
                acc += diff * diff;
                rel++;
            }
        out[i] = rel ? sqrt(acc / (double)rel) : 0.0;
    }
    free(knn), free(nbr), free(order), free(shared), free(means), free(var);
    return PYOD_OK;
}

static pyod_status sod_fit(pyod_detector *d, const pyod_matrix *x) {
    sod_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    double *s = NEW(double, x->rows);
    if (!s) return PYOD_ERR_NOMEM;
    pyod_status st = sod(m, x, s);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(d, s, x->rows, m->p.contamination);
}

static pyod_status sod_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return sod(d->impl, x, out); }

static const pyod_vtable SOD_VT = {"SOD", sod_fit, NULL, sod_score, free};

pyod_sod_params pyod_sod_defaults(void) { return (pyod_sod_params){20, 10, 0.8, 0.1}; }

pyod_detector *pyod_sod_create(const pyod_sod_params *p) {
    sod_impl *m = NEW(sod_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_sod_defaults();
    return detector_new(&SOD_VT, m);
}
