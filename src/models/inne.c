/* INNE: isolation using nearest-neighbour ensembles (Bandaragoda et al., 2018).
 * Port of pyod.models.inne. */
#include "internal.h"

typedef struct {
    pyod_inne_params p;
    size_t psi, d, n_ens;
    double *centroids; /* n_ens x psi x d */
    double *radius;    /* n_ens x psi (squared distance to the nearest other centroid) */
    double *ratio;     /* n_ens x psi */
} inne_impl;

static void inne_scores(const inne_impl *m, const pyod_matrix *x, double *out) {
    size_t psi = m->psi, d = m->d;
    for (size_t i = 0; i < x->rows; i++) {
        double total = 0.0;
        for (size_t e = 0; e < m->n_ens; e++) {
            double best_r = INFINITY, score = 1.0; /* uncovered samples score 1 */
            for (size_t c = 0; c < psi; c++) {
                double rad = m->radius[e * psi + c];
                if (sq_euclidean(&PYOD_AT(x, i, 0), m->centroids + (e * psi + c) * d, d) <= rad && rad < best_r) {
                    best_r = rad;
                    score = m->ratio[e * psi + c];
                }
            }
            total += score;
        }
        out[i] = total / (double)m->n_ens;
    }
}

static pyod_status inne_fit(pyod_detector *dt, const pyod_matrix *x) {
    inne_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t n = x->rows, d = x->cols, T = m->p.n_estimators;
    size_t psi = m->p.max_samples ? m->p.max_samples : 8;
    if (psi > n) psi = n;
    if (psi < 2 || T == 0) return PYOD_ERR_PARAM;
    free(m->centroids), free(m->radius), free(m->ratio);
    m->psi = psi, m->d = d, m->n_ens = T;
    m->centroids = NEW(double, T * psi * d);
    m->radius = NEW(double, T * psi);
    m->ratio = NEW(double, T * psi);
    size_t *idx = NEW(size_t, psi), *nn = NEW(size_t, psi);
    double *s = NEW(double, n);
    if (!m->centroids || !m->radius || !m->ratio || !idx || !nn || !s) {
        free(idx), free(nn), free(s);
        return PYOD_ERR_NOMEM;
    }
    pyod_rng r;
    rng_seed(&r, m->p.random_state);
    for (size_t e = 0; e < T; e++) {
        rng_sample(&r, n, psi, idx);
        double *cen = m->centroids + e * psi * d, *rad = m->radius + e * psi;
        for (size_t c = 0; c < psi; c++) memcpy(cen + c * d, &PYOD_AT(x, idx[c], 0), d * sizeof(double));
        for (size_t a = 0; a < psi; a++) {
            rad[a] = INFINITY;
            for (size_t b = 0; b < psi; b++) {
                if (a == b) continue;
                double v = sq_euclidean(cen + a * d, cen + b * d, d);
                if (v < rad[a]) rad[a] = v, nn[a] = b;
            }
        }
        for (size_t a = 0; a < psi; a++)
            m->ratio[e * psi + a] = 1.0 - (rad[nn[a]] + PYOD_DBL_EPS) / (rad[a] + PYOD_DBL_EPS);
    }
    free(idx), free(nn);
    inne_scores(m, x, s);
    return fit_state_contamination(dt, s, n, m->p.contamination);
}

static pyod_status inne_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    inne_scores(d->impl, x, out);
    return PYOD_OK;
}

static void inne_destroy(void *p) {
    inne_impl *m = p;
    free(m->centroids), free(m->radius), free(m->ratio), free(m);
}

static const pyod_vtable INNE_VT = {"INNE", inne_fit, NULL, inne_score, inne_destroy};

pyod_inne_params pyod_inne_defaults(void) { return (pyod_inne_params){200, 0, 0.1, 0}; }

pyod_detector *pyod_inne_create(const pyod_inne_params *p) {
    inne_impl *m = NEW(inne_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_inne_defaults();
    return detector_new(&INNE_VT, m);
}
