/* LODA: lightweight on-line detector of anomalies (Pevný, 2016).
 * Port of pyod.models.loda with a fixed number of bins. */
#include "internal.h"

typedef struct {
    pyod_loda_params p;
    size_t d;
    double *proj;   /* cuts x d */
    double *hist;   /* cuts x n_bins */
    double *limits; /* cuts x (n_bins + 1) */
} loda_impl;

static void loda_scores(const loda_impl *m, const pyod_matrix *x, double *out) {
    size_t nb = m->p.n_bins, cuts = m->p.n_random_cuts;
    for (size_t i = 0; i < x->rows; i++) {
        double s = 0.0;
        for (size_t c = 0; c < cuts; c++) {
            double v = dot(m->proj + c * m->d, &PYOD_AT(x, i, 0), m->d);
            const double *lim = m->limits + c * (nb + 1);
            size_t idx = 0; /* np.searchsorted(limits[:n_bins - 1], v, side='left') */
            while (idx < nb - 1 && lim[idx] < v) idx++;
            s += -(1.0 / (double)cuts) * log(m->hist[c * nb + idx]);
        }
        out[i] = s / (double)cuts;
    }
}

static pyod_status loda_fit(pyod_detector *dt, const pyod_matrix *x) {
    loda_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t n = x->rows, d = x->cols, nb = m->p.n_bins, cuts = m->p.n_random_cuts;
    if (nb < 2 || cuts == 0) return PYOD_ERR_PARAM;
    free(m->proj), free(m->hist), free(m->limits);
    m->d = d;
    m->proj = NEW(double, cuts * d);
    m->hist = NEW(double, cuts * nb);
    m->limits = NEW(double, cuts * (nb + 1));
    double *pv = NEW(double, n), *s = NEW(double, n);
    size_t n_zero = d - (size_t)sqrt((double)d);
    size_t *zero = NEW(size_t, d);
    if (!m->proj || !m->hist || !m->limits || !pv || !s || !zero) {
        free(pv), free(s), free(zero);
        return PYOD_ERR_NOMEM;
    }
    pyod_rng r;
    rng_seed(&r, m->p.random_state);
    for (size_t c = 0; c < cuts; c++) {
        double *p = m->proj + c * d;
        for (size_t j = 0; j < d; j++) p[j] = rng_normal(&r);
        rng_sample(&r, d, n_zero, zero);
        for (size_t t = 0; t < n_zero; t++) p[zero[t]] = 0.0;
        for (size_t i = 0; i < n; i++) pv[i] = dot(p, &PYOD_AT(x, i, 0), d);
        double lo = vec_min(pv, n), hi = vec_max(pv, n);
        if (lo == hi) lo -= 0.5, hi += 0.5;
        double *lim = m->limits + c * (nb + 1), *h = m->hist + c * nb;
        for (size_t b = 0; b <= nb; b++) lim[b] = lo + (hi - lo) * (double)b / (double)nb;
        for (size_t b = 0; b < nb; b++) h[b] = 1e-12;
        for (size_t i = 0; i < n; i++) {
            size_t b = (size_t)((pv[i] - lo) / (hi - lo) * (double)nb);
            h[b < nb ? b : nb - 1] += 1.0;
        }
        double total = 0.0;
        for (size_t b = 0; b < nb; b++) total += h[b];
        for (size_t b = 0; b < nb; b++) h[b] /= total;
    }
    free(pv), free(zero);
    loda_scores(m, x, s);
    return fit_state_contamination(dt, s, n, m->p.contamination);
}

static pyod_status loda_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    loda_scores(d->impl, x, out);
    return PYOD_OK;
}

static void loda_destroy(void *p) {
    loda_impl *m = p;
    free(m->proj), free(m->hist), free(m->limits), free(m);
}

static const pyod_vtable LODA_VT = {"LODA", loda_fit, NULL, loda_score, loda_destroy};

pyod_loda_params pyod_loda_defaults(void) { return (pyod_loda_params){10, 100, 0.1, 0}; }

pyod_detector *pyod_loda_create(const pyod_loda_params *p) {
    loda_impl *m = NEW(loda_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_loda_defaults();
    return detector_new(&LODA_VT, m);
}
