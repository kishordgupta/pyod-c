/* QMCD: quasi-Monte Carlo discrepancy outlier detection (Fang & Ma, 2001).
 * Port of pyod.models.qmcd. */
#include "internal.h"

typedef struct {
    pyod_qmcd_params p;
    double *min, *range;
    pyod_matrix data; /* min-max normalised training data */
    int flipped;
} qmcd_impl;

static void normalize(const qmcd_impl *m, const pyod_matrix *x, double *out) {
    for (size_t i = 0; i < x->rows; i++)
        for (size_t j = 0; j < x->cols; j++) out[i * x->cols + j] = (PYOD_AT(x, i, j) - m->min[j]) / m->range[j];
}

/* Wrap-around L2 discrepancy of each `check` row against `data`. */
static void wad(const pyod_matrix *data, const double *check, size_t nc, double *out) {
    size_t n = data->rows, d = data->cols;
    for (size_t i = 0; i < nc; i++) {
        double dc = 0.0;
        for (size_t r = 0; r < n; r++) {
            double prod = 1.0;
            for (size_t j = 0; j < d; j++) {
                double v = fabs(check[i * d + j] - PYOD_AT(data, r, j));
                prod *= 1.5 - v + v * v;
            }
            dc += prod;
        }
        out[i] = -pow(4.0 / 3.0, (double)d) + dc / ((double)n * (double)n);
    }
}

static pyod_status qmcd_fit(pyod_detector *dt, const pyod_matrix *x) {
    qmcd_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t n = x->rows, d = x->cols;
    free(m->min), free(m->range);
    pyod_matrix_free(&m->data);
    m->min = NEW(double, d);
    m->range = NEW(double, d);
    double *s = NEW(double, n);
    if (!m->min || !m->range || !s || pyod_matrix_alloc(&m->data, n, d) != PYOD_OK) {
        free(s);
        return PYOD_ERR_NOMEM;
    }
    for (size_t j = 0; j < d; j++) {
        double lo = INFINITY, hi = -INFINITY;
        for (size_t i = 0; i < n; i++) lo = fmin(lo, PYOD_AT(x, i, j)), hi = fmax(hi, PYOD_AT(x, i, j));
        m->min[j] = lo;
        m->range[j] = hi - lo == 0.0 ? 1.0 : hi - lo;
    }
    normalize(m, x, m->data.data);
    wad(&m->data, m->data.data, n, s);
    m->flipped = vec_skew(s, n) < 0.0 || vec_kurtosis(s, n) < 0.0;
    if (m->flipped) {
        double mn = vec_min(s, n), mx = vec_max(s, n);
        for (size_t i = 0; i < n; i++) s[i] = mx + mn - s[i];
    }
    return fit_state_contamination(dt, s, n, m->p.contamination);
}

static pyod_status qmcd_score(const pyod_detector *dt, const pyod_matrix *x, double *out) {
    const qmcd_impl *m = dt->impl;
    double *xn = NEW(double, x->rows * x->cols);
    if (!xn) return PYOD_ERR_NOMEM;
    normalize(m, x, xn);
    wad(&m->data, xn, x->rows, out);
    free(xn);
    if (m->flipped) {
        double mn = vec_min(dt->state.scores, dt->state.n), mx = vec_max(dt->state.scores, dt->state.n);
        for (size_t i = 0; i < x->rows; i++) out[i] = mx + mn - out[i];
    }
    return PYOD_OK;
}

static void qmcd_destroy(void *p) {
    qmcd_impl *m = p;
    free(m->min), free(m->range);
    pyod_matrix_free(&m->data);
    free(m);
}

static const pyod_vtable QMCD_VT = {"QMCD", qmcd_fit, NULL, qmcd_score, qmcd_destroy};

pyod_qmcd_params pyod_qmcd_defaults(void) { return (pyod_qmcd_params){0.1}; }

pyod_detector *pyod_qmcd_create(const pyod_qmcd_params *p) {
    qmcd_impl *m = NEW(qmcd_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_qmcd_defaults();
    return detector_new(&QMCD_VT, m);
}
