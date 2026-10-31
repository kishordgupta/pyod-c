/* ABOD: angle-based outlier detection, fast variant (Kriegel et al., KDD 2008).
 * Port of pyod.models.abod with method='fast'. */
#include "internal.h"

typedef struct {
    pyod_abod_params p;
    pyod_matrix train;
} abod_impl;

/* Negated variance of distance-weighted cosines between neighbour pairs;
 * pairs containing the point itself are skipped, as in PyOD. */
static double abod_value(const abod_impl *m, const double *pt, const size_t *nb, size_t k) {
    size_t d = m->train.cols, cnt = 0;
    double sum = 0.0, sumsq = 0.0;
    double *va = NEW(double, d), *vb = NEW(double, d);
    if (!va || !vb) {
        free(va), free(vb);
        return NAN;
    }
    for (size_t a = 0; a < k; a++) {
        const double *pa = &PYOD_AT(&m->train, nb[a], 0);
        if (memcmp(pa, pt, d * sizeof(double)) == 0) continue;
        for (size_t b = a + 1; b < k; b++) {
            const double *pb = &PYOD_AT(&m->train, nb[b], 0);
            if (memcmp(pb, pt, d * sizeof(double)) == 0) continue;
            for (size_t j = 0; j < d; j++) {
                va[j] = pa[j] - pt[j];
                vb[j] = pb[j] - pt[j];
            }
            double w = dot(va, vb, d) / dot(va, va, d) / dot(vb, vb, d);
            sum += w;
            sumsq += w * w;
            cnt++;
        }
    }
    free(va), free(vb);
    if (!cnt) return 0.0;
    double mean = sum / (double)cnt;
    return -(sumsq / (double)cnt - mean * mean);
}

static pyod_status abod_fit(pyod_detector *d, const pyod_matrix *x) {
    abod_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t k = m->p.n_neighbors, n = x->rows;
    if (k < 2 || k >= n) return PYOD_ERR_PARAM;
    pyod_matrix_free(&m->train);
    CHECK(mat_copy(x, &m->train));
    size_t *nb = NEW(size_t, k);
    double *s = NEW(double, n);
    if (!nb || !s) {
        free(nb), free(s);
        return PYOD_ERR_NOMEM;
    }
    /* Like PyOD, the training query includes the point itself. */
    for (size_t i = 0; i < n; i++) {
        k_nearest(x, &PYOD_AT(x, i, 0), k, SIZE_MAX, nb, NULL);
        s[i] = abod_value(m, &PYOD_AT(x, i, 0), nb, k);
    }
    free(nb);
    return fit_state_contamination(d, s, n, m->p.contamination);
}

static pyod_status abod_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    const abod_impl *m = d->impl;
    size_t k = m->p.n_neighbors;
    size_t *nb = NEW(size_t, k);
    if (!nb) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < x->rows; i++) {
        k_nearest(&m->train, &PYOD_AT(x, i, 0), k, SIZE_MAX, nb, NULL);
        out[i] = abod_value(m, &PYOD_AT(x, i, 0), nb, k);
    }
    free(nb);
    return PYOD_OK;
}

static void abod_destroy(void *p) {
    abod_impl *m = p;
    pyod_matrix_free(&m->train);
    free(m);
}

static const pyod_vtable ABOD_VT = {"ABOD", abod_fit, NULL, abod_score, abod_destroy};

pyod_abod_params pyod_abod_defaults(void) { return (pyod_abod_params){5, 0.1}; }

pyod_detector *pyod_abod_create(const pyod_abod_params *p) {
    abod_impl *m = NEW(abod_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_abod_defaults();
    return detector_new(&ABOD_VT, m);
}
