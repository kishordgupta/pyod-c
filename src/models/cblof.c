/* CBLOF: clustering-based local outlier factor (He et al., 2003).
 * Port of pyod.models.cblof with k-means clustering. */
#include "internal.h"

typedef struct {
    pyod_cblof_params p;
    double *centers;
    size_t *sizes;
    int *large;
    size_t k, d;
} cblof_impl;

static void cblof_scores(const cblof_impl *m, const pyod_matrix *x, const size_t *labels, double *out) {
    for (size_t i = 0; i < x->rows; i++) {
        const double *r = &PYOD_AT(x, i, 0);
        size_t l = labels ? labels[i] : nearest_center(m->centers, m->k, m->d, r, NULL);
        double s;
        if (m->large[l]) {
            s = euclidean(r, m->centers + l * m->d, m->d);
        } else {
            s = INFINITY;
            for (size_t c = 0; c < m->k; c++)
                if (m->large[c]) s = fmin(s, euclidean(r, m->centers + c * m->d, m->d));
        }
        out[i] = m->p.use_weights ? s * (double)m->sizes[l] : s;
    }
}

static pyod_status cblof_fit(pyod_detector *dt, const pyod_matrix *x) {
    cblof_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t n = x->rows, k = m->p.n_clusters, d = x->cols;
    if (!(m->p.alpha > 0.0 && m->p.alpha < 1.0) || m->p.beta <= 1.0 || k < 2 || k > n) return PYOD_ERR_PARAM;
    free(m->centers), free(m->sizes), free(m->large);
    m->k = k, m->d = d;
    m->centers = NEW(double, k * d);
    m->sizes = NEW(size_t, k);
    m->large = NEW(int, k);
    size_t *labels = NEW(size_t, n), *order = NEW(size_t, k);
    double *negsize = NEW(double, k), *s = NEW(double, n);
    pyod_status st = PYOD_ERR_NOMEM;
    if (!m->centers || !m->sizes || !m->large || !labels || !order || !negsize || !s) goto done;
    pyod_rng r;
    rng_seed(&r, m->p.random_state);
    kmeans(x, k, 1, 300, &r, m->centers, labels);
    for (size_t i = 0; i < n; i++) m->sizes[labels[i]]++;
    /* _set_small_large_clusters */
    for (size_t c = 0; c < k; c++) negsize[c] = -(double)m->sizes[c];
    argsort(negsize, k, order);
    size_t threshold = 0, first_alpha = 0, first_beta = 0;
    for (size_t i = 1; i < k && !threshold; i++) {
        size_t covered = 0;
        for (size_t t = 0; t < i; t++) covered += m->sizes[order[t]];
        int a = (double)covered >= (double)n * m->p.alpha;
        int b = (double)m->sizes[order[i - 1]] / (double)m->sizes[order[i]] >= m->p.beta;
        if (a && !first_alpha) first_alpha = i;
        if (b && !first_beta) first_beta = i;
        if (a && b) threshold = i;
    }
    if (!threshold) threshold = first_alpha ? first_alpha : first_beta;
    st = PYOD_ERR_PARAM; /* could not separate small and large clusters; change n_clusters */
    if (!threshold) goto done;
    for (size_t t = 0; t < threshold; t++) m->large[order[t]] = 1;
    cblof_scores(m, x, labels, s);
    st = fit_state_contamination(dt, s, n, m->p.contamination);
    s = NULL;
done:
    free(labels), free(order), free(negsize), free(s);
    return st;
}

static pyod_status cblof_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    cblof_scores(d->impl, x, NULL, out);
    return PYOD_OK;
}

static void cblof_destroy(void *p) {
    cblof_impl *m = p;
    free(m->centers), free(m->sizes), free(m->large), free(m);
}

static const pyod_vtable CBLOF_VT = {"CBLOF", cblof_fit, NULL, cblof_score, cblof_destroy};

pyod_cblof_params pyod_cblof_defaults(void) { return (pyod_cblof_params){8, 0.9, 5.0, 0, 0.1, 0}; }

pyod_detector *pyod_cblof_create(const pyod_cblof_params *p) {
    cblof_impl *m = NEW(cblof_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_cblof_defaults();
    return detector_new(&CBLOF_VT, m);
}
