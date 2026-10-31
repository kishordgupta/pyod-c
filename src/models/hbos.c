/* HBOS: histogram-based outlier score (Goldstein & Dengel, 2012).
 * Port of pyod.models.hbos with a fixed number of bins. */
#include "internal.h"

typedef struct {
    pyod_hbos_params p;
    double *hist;  /* d x n_bins densities */
    double *edges; /* d x (n_bins + 1) */
    size_t d;
} hbos_impl;

/* Summed log2 densities per row (higher = more normal). */
static void log_density(const hbos_impl *m, const pyod_matrix *x, double *out) {
    size_t nb = m->p.n_bins;
    for (size_t i = 0; i < x->rows; i++) out[i] = 0.0;
    for (size_t j = 0; j < m->d; j++) {
        const double *h = m->hist + j * nb, *e = m->edges + j * (nb + 1);
        double min_score = INFINITY;
        for (size_t b = 0; b < nb; b++) min_score = fmin(min_score, log2(h[b] + m->p.alpha));
        for (size_t i = 0; i < x->rows; i++) {
            double v = PYOD_AT(x, i, j);
            size_t idx = 0; /* np.digitize(right=True): first edge >= v */
            while (idx <= nb && e[idx] < v) idx++;
            double s;
            if (idx == 0)
                s = e[0] - v <= (e[1] - e[0]) * m->p.tol ? log2(h[0] + m->p.alpha) : min_score;
            else if (idx == nb + 1)
                s = v - e[nb] <= (e[nb] - e[nb - 1]) * m->p.tol ? log2(h[nb - 1] + m->p.alpha) : min_score;
            else
                s = log2(h[idx - 1] + m->p.alpha);
            out[i] += s;
        }
    }
}

static pyod_status hbos_fit(pyod_detector *d, const pyod_matrix *x) {
    hbos_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t nb = m->p.n_bins, n = x->rows;
    if (nb < 2) return PYOD_ERR_PARAM;
    free(m->hist), free(m->edges);
    m->d = x->cols;
    m->hist = NEW(double, m->d * nb);
    m->edges = NEW(double, m->d * (nb + 1));
    double *s = NEW(double, n);
    if (!m->hist || !m->edges || !s) {
        free(s);
        return PYOD_ERR_NOMEM;
    }
    for (size_t j = 0; j < m->d; j++) {
        double lo = INFINITY, hi = -INFINITY;
        for (size_t i = 0; i < n; i++) {
            lo = fmin(lo, PYOD_AT(x, i, j));
            hi = fmax(hi, PYOD_AT(x, i, j));
        }
        if (lo == hi) lo -= 0.5, hi += 0.5;
        double *e = m->edges + j * (nb + 1), *h = m->hist + j * nb;
        for (size_t b = 0; b <= nb; b++) e[b] = lo + (hi - lo) * (double)b / (double)nb;
        for (size_t i = 0; i < n; i++) {
            size_t b = (size_t)((PYOD_AT(x, i, j) - lo) / (hi - lo) * (double)nb);
            h[b < nb ? b : nb - 1] += 1.0;
        }
        for (size_t b = 0; b < nb; b++) h[b] /= (double)n * (e[b + 1] - e[b]);
    }
    log_density(m, x, s);
    for (size_t i = 0; i < n; i++) s[i] = -s[i];
    return fit_state_contamination(d, s, n, m->p.contamination);
}

static pyod_status hbos_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    log_density(d->impl, x, out);
    for (size_t i = 0; i < x->rows; i++) out[i] = -out[i];
    return PYOD_OK;
}

static void hbos_destroy(void *p) {
    hbos_impl *m = p;
    free(m->hist), free(m->edges), free(m);
}

static const pyod_vtable HBOS_VT = {"HBOS", hbos_fit, NULL, hbos_score, hbos_destroy};

pyod_hbos_params pyod_hbos_defaults(void) { return (pyod_hbos_params){10, 0.1, 0.5, 0.1}; }

pyod_detector *pyod_hbos_create(const pyod_hbos_params *p) {
    hbos_impl *m = NEW(hbos_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_hbos_defaults();
    return detector_new(&HBOS_VT, m);
}
