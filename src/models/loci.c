/* LOCI: local correlation integral (Papadimitriou et al., 2003). Port of
 * pyod.models.loci; O(n^3). Transductive: samples are scored against each other. */
#include "internal.h"

typedef struct {
    pyod_loci_params p;
} loci_impl;

static pyod_status loci(const loci_impl *m, const pyod_matrix *x, double *out) {
    size_t n = x->rows;
    double *dist = NEW(double, n * n), *crit = NEW(double, 2 * n), *counts = NEW(double, n);
    if (!dist || !crit || !counts) {
        free(dist), free(crit), free(counts);
        return PYOD_ERR_NOMEM;
    }
    double max_dist = 0.0, a = m->p.alpha;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            dist[i * n + j] = euclidean(&PYOD_AT(x, i, 0), &PYOD_AT(x, j, 0), x->cols);
            max_dist = fmax(max_dist, dist[i * n + j]);
        }
    double r_max = max_dist / a;
    for (size_t p = 0; p < n; p++) {
        size_t nc = 0;
        for (size_t j = 0; j < n; j++) {
            double dv = dist[p * n + j];
            if (dv > 0.0 && dv <= r_max) crit[nc++] = dv, crit[nc++] = dv / a;
        }
        qsort(crit, nc, sizeof(double), cmp_double);
        double score = 0.0;
        for (size_t c = 0; c < nc; c++) {
            double r = crit[c];
            size_t m_cnt = 0;
            for (size_t j = 0; j < n; j++) {
                if (dist[p * n + j] > r) continue;
                double cnt = 0.0;
                for (size_t t = 0; t < n; t++) cnt += dist[j * n + t] < r * a;
                counts[m_cnt++] = cnt;
            }
            double n_hat = vec_mean(counts, m_cnt), own = 0.0;
            for (size_t t = 0; t < n; t++) own += dist[p * n + t] < r * a;
            double mdef = 1.0 - own / n_hat, sigma = vec_std(counts, m_cnt) / n_hat;
            if (n_hat >= 20.0) {
                score = mdef / sigma;
                if (mdef > m->p.k * sigma) break;
            }
        }
        out[p] = score;
    }
    free(dist), free(crit), free(counts);
    return PYOD_OK;
}

static pyod_status loci_fit(pyod_detector *d, const pyod_matrix *x) {
    loci_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    double *s = NEW(double, x->rows);
    if (!s) return PYOD_ERR_NOMEM;
    pyod_status st = loci(m, x, s);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(d, s, x->rows, m->p.contamination);
}

static pyod_status loci_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return loci(d->impl, x, out); }

static const pyod_vtable LOCI_VT = {"LOCI", loci_fit, NULL, loci_score, free};

pyod_loci_params pyod_loci_defaults(void) { return (pyod_loci_params){0.5, 3.0, 0.1}; }

pyod_detector *pyod_loci_create(const pyod_loci_params *p) {
    loci_impl *m = NEW(loci_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_loci_defaults();
    return detector_new(&LOCI_VT, m);
}
