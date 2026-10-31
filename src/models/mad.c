/* MAD: median absolute deviation for univariate data (Iglewicz & Hoaglin, 1993).
 * Port of pyod.models.mad; labels use the fixed modified-z-score threshold. */
#include "internal.h"

typedef struct {
    double threshold, median, median_diff;
} mad_impl;

static void mad_scores(const mad_impl *m, const pyod_matrix *x, double *out) {
    for (size_t i = 0; i < x->rows; i++) {
        double z = 0.6745 * fabs(x->data[i] - m->median) / m->median_diff;
        out[i] = isnan(z) ? 0.0 : isinf(z) ? 1.7976931348623157e308 : z;
    }
}

static pyod_status mad_fit(pyod_detector *d, const pyod_matrix *x) {
    mad_impl *m = d->impl;
    if (x->cols != 1) return PYOD_ERR_DIMENSION;
    size_t n = x->rows;
    double *diff = NEW(double, n), *s = NEW(double, n);
    if (!diff || !s) {
        free(diff), free(s);
        return PYOD_ERR_NOMEM;
    }
    m->median = vec_median(x->data, n);
    for (size_t i = 0; i < n; i++) diff[i] = fabs(x->data[i] - m->median);
    m->median_diff = vec_median(diff, n);
    free(diff);
    mad_scores(m, x, s);
    return fit_state_threshold(d, s, n, m->threshold);
}

static pyod_status mad_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    mad_scores(d->impl, x, out);
    return PYOD_OK;
}

static const pyod_vtable MAD_VT = {"MAD", mad_fit, NULL, mad_score, free};

pyod_mad_params pyod_mad_defaults(void) { return (pyod_mad_params){3.5}; }

pyod_detector *pyod_mad_create(const pyod_mad_params *p) {
    mad_impl *m = NEW(mad_impl, 1);
    if (!m) return NULL;
    m->threshold = p ? p->threshold : 3.5;
    return detector_new(&MAD_VT, m);
}
