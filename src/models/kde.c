/* KDE outlier detection (Latecki et al., 2007). Port of pyod.models.kde
 * (sklearn KernelDensity, Euclidean metric); the score is -log density. */
#include "internal.h"

typedef struct {
    pyod_kde_params p;
    pyod_matrix train;
} kde_impl;

static double log_vn(double n) { return 0.5 * n * log(PYOD_PI) - lgamma_(0.5 * n + 1.0); }

/* sklearn's _log_kernel_norm. */
static double log_norm(const kde_impl *m, double d) {
    double f;
    switch (m->p.kernel) {
    case PYOD_KERNEL_TOPHAT: f = log_vn(d); break;
    case PYOD_KERNEL_EPANECHNIKOV: f = log_vn(d) + log(2.0 / (d + 2.0)); break;
    case PYOD_KERNEL_EXPONENTIAL: f = log(2.0 * PYOD_PI) + log_vn(d - 1.0) + lgamma_(d); break;
    case PYOD_KERNEL_LINEAR: f = log_vn(d) - log(d + 1.0); break;
    default: f = 0.5 * d * log(2.0 * PYOD_PI);
    }
    return -f - d * log(m->p.bandwidth);
}

static double log_kernel(const kde_impl *m, double dist) {
    double u = dist / m->p.bandwidth;
    switch (m->p.kernel) {
    case PYOD_KERNEL_TOPHAT: return u < 1.0 ? 0.0 : -INFINITY;
    case PYOD_KERNEL_EPANECHNIKOV: return u < 1.0 ? log(1.0 - u * u) : -INFINITY;
    case PYOD_KERNEL_EXPONENTIAL: return -u;
    case PYOD_KERNEL_LINEAR: return u < 1.0 ? log(1.0 - u) : -INFINITY;
    default: return -0.5 * u * u;
    }
}

static pyod_status kde_scores(const kde_impl *m, const pyod_matrix *x, double *out) {
    size_t n = m->train.rows;
    double *logs = NEW(double, n);
    if (!logs) return PYOD_ERR_NOMEM;
    double norm = log_norm(m, (double)m->train.cols);
    for (size_t i = 0; i < x->rows; i++) {
        for (size_t t = 0; t < n; t++)
            logs[t] = log_kernel(m, euclidean(&PYOD_AT(x, i, 0), &PYOD_AT(&m->train, t, 0), x->cols));
        out[i] = -(logsumexp(logs, n) + norm - log((double)n));
    }
    free(logs);
    return PYOD_OK;
}

static pyod_status kde_fit(pyod_detector *d, const pyod_matrix *x) {
    kde_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    if (m->p.bandwidth <= 0.0) return PYOD_ERR_PARAM;
    pyod_matrix_free(&m->train);
    CHECK(mat_copy(x, &m->train));
    double *s = NEW(double, x->rows);
    if (!s) return PYOD_ERR_NOMEM;
    kde_scores(m, x, s);
    return fit_state_contamination(d, s, x->rows, m->p.contamination);
}

static pyod_status kde_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    return kde_scores(d->impl, x, out);
}

static void kde_destroy(void *p) {
    kde_impl *m = p;
    pyod_matrix_free(&m->train);
    free(m);
}

static const pyod_vtable KDE_VT = {"KDE", kde_fit, NULL, kde_score, kde_destroy};

pyod_kde_params pyod_kde_defaults(void) { return (pyod_kde_params){1.0, PYOD_KERNEL_GAUSSIAN, 0.1}; }

pyod_detector *pyod_kde_create(const pyod_kde_params *p) {
    kde_impl *m = NEW(kde_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_kde_defaults();
    return detector_new(&KDE_VT, m);
}
