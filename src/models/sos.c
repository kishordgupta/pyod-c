/* SOS: stochastic outlier selection (Janssens et al., 2012). Port of
 * pyod.models.sos. Transductive: samples are scored against each other. */
#include "internal.h"

typedef struct {
    pyod_sos_params p;
} sos_impl;

static double perplexity_h(const double *dist, size_t n, double beta, double *a) {
    double sum = 0.0, dsum = 0.0;
    for (size_t i = 0; i < n; i++) {
        a[i] = exp(-dist[i] * beta);
        sum += a[i];
        dsum += dist[i] * a[i];
    }
    return log(sum) + beta * dsum / sum;
}

static pyod_status sos(const sos_impl *m, const pyod_matrix *x, double *out) {
    size_t n = x->rows;
    if (n < 2) return PYOD_ERR_PARAM;
    double *di = NEW(double, n - 1), *a = NEW(double, n - 1), log_u = log(m->p.perplexity);
    if (!di || !a) {
        free(di), free(a);
        return PYOD_ERR_NOMEM;
    }
    for (size_t i = 0; i < n; i++) out[i] = 1.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0, k = 0; j < n; j++)
            if (j != i) di[k++] = euclidean(&PYOD_AT(x, i, 0), &PYOD_AT(x, j, 0), x->cols);
        double beta = 1.0, bmin = -INFINITY, bmax = INFINITY;
        double diff = perplexity_h(di, n - 1, beta, a) - log_u;
        for (int tries = 0; (isnan(diff) || fabs(diff) > m->p.eps) && tries < 5000; tries++) {
            if (isnan(diff)) beta /= 10.0;
            else if (diff > 0) bmin = beta, beta = isinf(bmax) ? beta * 2.0 : (beta + bmax) / 2.0;
            else bmax = beta, beta = isinf(bmin) ? beta / 2.0 : (beta + bmin) / 2.0;
            diff = perplexity_h(di, n - 1, beta, a) - log_u;
        }
        double sum = 0.0;
        for (size_t k = 0; k < n - 1; k++) sum += a[k];
        /* Outlier probability = product over i of (1 - binding probability). */
        for (size_t j = 0, k = 0; j < n; j++)
            if (j != i) out[j] *= 1.0 - a[k++] / sum;
    }
    free(di), free(a);
    return PYOD_OK;
}

static pyod_status sos_fit(pyod_detector *d, const pyod_matrix *x) {
    sos_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    double *s = NEW(double, x->rows);
    if (!s) return PYOD_ERR_NOMEM;
    pyod_status st = sos(m, x, s);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(d, s, x->rows, m->p.contamination);
}

static pyod_status sos_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return sos(d->impl, x, out); }

static const pyod_vtable SOS_VT = {"SOS", sos_fit, NULL, sos_score, free};

pyod_sos_params pyod_sos_defaults(void) { return (pyod_sos_params){4.5, 1e-5, 0.1}; }

pyod_detector *pyod_sos_create(const pyod_sos_params *p) {
    sos_impl *m = NEW(sos_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_sos_defaults();
    return detector_new(&SOS_VT, m);
}
