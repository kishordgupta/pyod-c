/* COF: connectivity-based outlier factor (Tang et al., 2002). Port of
 * pyod.models.cof (fast method). Transductive: samples are scored against
 * each other, in fit and in decision_function alike. */
#include "internal.h"

typedef struct {
    pyod_cof_params p;
} cof_impl;

static pyod_status cof(const pyod_matrix *x, size_t k_req, double *out) {
    size_t n = x->rows, d = x->cols;
    size_t k = k_req < n - 1 ? k_req : n - 1;
    if (k < 1) k = 1;
    double *dist = NEW(double, n * n), *ac = NEW(double, n);
    size_t *order = NEW(size_t, n), *paths = NEW(size_t, n * k);
    if (!dist || !ac || !order || !paths) {
        free(dist), free(ac), free(order), free(paths);
        return PYOD_ERR_NOMEM;
    }
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) dist[i * n + j] = euclidean(&PYOD_AT(x, i, 0), &PYOD_AT(x, j, 0), d);
    for (size_t i = 0; i < n; i++) {
        argsort(dist + i * n, n, order);
        /* Set-based nearest path: each step costs the shortest link back to the path. */
        double acd = 0.0;
        for (size_t j = 0; j < k; j++) {
            double cost = INFINITY;
            for (size_t t = 0; t <= j; t++) cost = fmin(cost, dist[order[j + 1] * n + order[t]]);
            acd += 2.0 * (double)(k + 1 - (j + 1)) / (double)((k + 1) * k) * cost;
        }
        ac[i] = acd;
        memcpy(paths + i * k, order + 1, k * sizeof(size_t));
    }
    for (size_t g = 0; g < n; g++) {
        double denom = 0.0;
        for (size_t t = 0; t < k; t++) denom += ac[paths[g * k + t]];
        double v = ac[g] * (double)k / denom;
        out[g] = isnan(v) ? 0.0 : isinf(v) ? 1.7976931348623157e308 : v;
    }
    free(dist), free(ac), free(order), free(paths);
    return PYOD_OK;
}

static pyod_status cof_fit(pyod_detector *d, const pyod_matrix *x) {
    cof_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    if (x->rows < 2) return PYOD_ERR_PARAM;
    double *s = NEW(double, x->rows);
    if (!s) return PYOD_ERR_NOMEM;
    pyod_status st = cof(x, m->p.n_neighbors, s);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(d, s, x->rows, m->p.contamination);
}

static pyod_status cof_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    if (x->rows < 2) return PYOD_ERR_PARAM;
    return cof(x, ((const cof_impl *)d->impl)->p.n_neighbors, out);
}

static const pyod_vtable COF_VT = {"COF", cof_fit, NULL, cof_score, free};

pyod_cof_params pyod_cof_defaults(void) { return (pyod_cof_params){20, 0.1}; }

pyod_detector *pyod_cof_create(const pyod_cof_params *p) {
    cof_impl *m = NEW(cof_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_cof_defaults();
    return detector_new(&COF_VT, m);
}
