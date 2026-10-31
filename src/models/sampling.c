/* Rapid distance-based outlier detection via sampling (Sugiyama & Borgwardt,
 * 2013). Port of pyod.models.sampling. */
#include "internal.h"

typedef struct {
    pyod_sampling_params p;
    pyod_matrix subset;
} sampling_impl;

static void sampling_scores(const sampling_impl *m, const pyod_matrix *x, double *out) {
    for (size_t i = 0; i < x->rows; i++) {
        double best = INFINITY;
        for (size_t s = 0; s < m->subset.rows; s++) best = fmin(best, euclidean(&PYOD_AT(x, i, 0), &PYOD_AT(&m->subset, s, 0), x->cols));
        out[i] = best;
    }
}

static pyod_status sampling_fit(pyod_detector *d, const pyod_matrix *x) {
    sampling_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t k = m->p.subset_size;
    if (k == 0 || k > x->rows) return PYOD_ERR_PARAM;
    size_t *idx = NEW(size_t, k);
    double *s = NEW(double, x->rows);
    if (!idx || !s) {
        free(idx), free(s);
        return PYOD_ERR_NOMEM;
    }
    pyod_rng r;
    rng_seed(&r, m->p.random_state);
    rng_sample(&r, x->rows, k, idx);
    pyod_matrix_free(&m->subset);
    pyod_status st = mat_rows(x, idx, k, &m->subset);
    free(idx);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    sampling_scores(m, x, s);
    return fit_state_contamination(d, s, x->rows, m->p.contamination);
}

static pyod_status sampling_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    sampling_scores(d->impl, x, out);
    return PYOD_OK;
}

static void sampling_destroy(void *p) {
    sampling_impl *m = p;
    pyod_matrix_free(&m->subset);
    free(m);
}

static const pyod_vtable SAMPLING_VT = {"Sampling", sampling_fit, NULL, sampling_score, sampling_destroy};

pyod_sampling_params pyod_sampling_defaults(void) { return (pyod_sampling_params){20, 0.1, 0}; }

pyod_detector *pyod_sampling_create(const pyod_sampling_params *p) {
    sampling_impl *m = NEW(sampling_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_sampling_defaults();
    return detector_new(&SAMPLING_VT, m);
}
