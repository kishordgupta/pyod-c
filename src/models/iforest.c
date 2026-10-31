/* Isolation Forest (Liu et al., ICDM 2008). Port of pyod.models.iforest;
 * scores are 2^(-E[h(x)] / c(psi)), which differ from PyOD's by a constant. */
#include "internal.h"

typedef struct {
    pyod_iforest_params p;
    itree_node **trees;
    size_t n_trees, psi;
} iforest_impl;

static void iforest_scores(const iforest_impl *m, const pyod_matrix *x, double *out) {
    double cn = itree_c(m->psi);
    for (size_t i = 0; i < x->rows; i++) {
        double h = 0.0;
        for (size_t t = 0; t < m->n_trees; t++) h += itree_path(m->trees[t], &PYOD_AT(x, i, 0), 0.0);
        out[i] = pow(2.0, -(h / (double)m->n_trees) / cn);
    }
}

static void free_trees(iforest_impl *m) {
    for (size_t t = 0; t < m->n_trees; t++) itree_free(m->trees[t]);
    free(m->trees);
    m->trees = NULL;
    m->n_trees = 0;
}

static pyod_status iforest_fit(pyod_detector *d, const pyod_matrix *x) {
    iforest_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    if (m->p.n_estimators == 0 || x->rows < 2) return PYOD_ERR_PARAM;
    free_trees(m);
    size_t n = x->rows, psi = m->p.max_samples ? m->p.max_samples : 256;
    m->psi = psi < 2 ? 2 : psi > n ? n : psi;
    size_t max_depth = (size_t)ceil(log2((double)m->psi));
    m->trees = NEW(itree_node *, m->p.n_estimators);
    size_t *idx = NEW(size_t, m->psi);
    double *s = NEW(double, n);
    if (!m->trees || !idx || !s) {
        free(idx), free(s);
        return PYOD_ERR_NOMEM;
    }
    pyod_rng r;
    rng_seed(&r, m->p.random_state);
    for (size_t t = 0; t < m->p.n_estimators; t++) {
        rng_sample(&r, n, m->psi, idx);
        m->trees[t] = itree_build(x, idx, m->psi, 0, max_depth, &r);
        m->n_trees++;
    }
    free(idx);
    iforest_scores(m, x, s);
    return fit_state_contamination(d, s, n, m->p.contamination);
}

static pyod_status iforest_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    iforest_scores(d->impl, x, out);
    return PYOD_OK;
}

static void iforest_destroy(void *p) {
    free_trees(p);
    free(p);
}

static const pyod_vtable IFOREST_VT = {"IForest", iforest_fit, NULL, iforest_score, iforest_destroy};

pyod_iforest_params pyod_iforest_defaults(void) { return (pyod_iforest_params){100, 0, 0.1, 42}; }

pyod_detector *pyod_iforest_create(const pyod_iforest_params *p) {
    iforest_impl *m = NEW(iforest_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_iforest_defaults();
    return detector_new(&IFOREST_VT, m);
}
