/* kNN outlier detection (Ramaswamy et al., 2000; Angiulli & Pizzuti, 2002) and
 * LOF (Breunig et al., 2000). Ports of pyod.models.knn and pyod.models.lof,
 * Euclidean distance, brute-force search. */
#include "internal.h"

typedef struct {
    pyod_knn_params p;
    pyod_matrix train;
} knn_impl;

static double knn_aggregate(pyod_knn_method m, double *dist, size_t k) {
    if (m == PYOD_KNN_LARGEST) return dist[k - 1];
    if (m == PYOD_KNN_MEAN) return vec_mean(dist, k);
    return vec_median(dist, k);
}

static pyod_status knn_fit(pyod_detector *d, const pyod_matrix *x) {
    knn_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t k = m->p.n_neighbors, n = x->rows;
    if (k == 0 || k >= n) return PYOD_ERR_PARAM;
    double *dist = NEW(double, k), *s = NEW(double, n);
    if (!dist || !s) {
        free(dist), free(s);
        return PYOD_ERR_NOMEM;
    }
    for (size_t i = 0; i < n; i++) {
        k_nearest(x, &PYOD_AT(x, i, 0), k, i, NULL, dist);
        s[i] = knn_aggregate(m->p.method, dist, k);
    }
    free(dist);
    pyod_matrix_free(&m->train);
    if (mat_copy(x, &m->train) != PYOD_OK) {
        free(s);
        return PYOD_ERR_NOMEM;
    }
    return fit_state_contamination(d, s, n, m->p.contamination);
}

static pyod_status knn_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    const knn_impl *m = d->impl;
    size_t k = m->p.n_neighbors;
    double *dist = NEW(double, k);
    if (!dist) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < x->rows; i++) {
        k_nearest(&m->train, &PYOD_AT(x, i, 0), k, SIZE_MAX, NULL, dist);
        out[i] = knn_aggregate(m->p.method, dist, k);
    }
    free(dist);
    return PYOD_OK;
}

static void knn_destroy(void *p) {
    knn_impl *m = p;
    pyod_matrix_free(&m->train);
    free(m);
}

static const pyod_vtable KNN_VT = {"KNN", knn_fit, NULL, knn_score, knn_destroy};

pyod_knn_params pyod_knn_defaults(void) { return (pyod_knn_params){5, PYOD_KNN_LARGEST, 0.1}; }

pyod_detector *pyod_knn_create(const pyod_knn_params *p) {
    knn_impl *m = NEW(knn_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_knn_defaults();
    return detector_new(&KNN_VT, m);
}

/* ---------------- LOF ---------------- */

typedef struct {
    pyod_lof_params p;
    pyod_matrix train;
    double *k_distance, *lrd;
    size_t k;
} lof_impl;

static double lof_lrd(const lof_impl *m, const double *q, const size_t *nb, size_t k) {
    double reach = 0.0;
    for (size_t t = 0; t < k; t++)
        reach += fmax(m->k_distance[nb[t]], euclidean(q, &PYOD_AT(&m->train, nb[t], 0), m->train.cols));
    return 1.0 / (reach / (double)k + 1e-10);
}

static double lof_value(const lof_impl *m, double lrd, const size_t *nb, size_t k) {
    double s = 0.0;
    for (size_t t = 0; t < k; t++) s += m->lrd[nb[t]];
    return s / (double)k / lrd;
}

static pyod_status lof_fit(pyod_detector *d, const pyod_matrix *x) {
    lof_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t n = x->rows;
    if (n < 2) return PYOD_ERR_PARAM;
    m->k = m->p.n_neighbors < 1 ? 1 : m->p.n_neighbors > n - 1 ? n - 1 : m->p.n_neighbors;
    size_t k = m->k;
    pyod_matrix_free(&m->train);
    free(m->k_distance), free(m->lrd);
    CHECK(mat_copy(x, &m->train));
    size_t *nbr = NEW(size_t, n * k);
    double *dist = NEW(double, k), *s = NEW(double, n);
    m->k_distance = NEW(double, n);
    m->lrd = NEW(double, n);
    if (!nbr || !dist || !s || !m->k_distance || !m->lrd) {
        free(nbr), free(dist), free(s);
        return PYOD_ERR_NOMEM;
    }
    for (size_t i = 0; i < n; i++) {
        k_nearest(x, &PYOD_AT(x, i, 0), k, i, nbr + i * k, dist);
        m->k_distance[i] = dist[k - 1];
    }
    for (size_t i = 0; i < n; i++) m->lrd[i] = lof_lrd(m, &PYOD_AT(x, i, 0), nbr + i * k, k);
    for (size_t i = 0; i < n; i++) s[i] = lof_value(m, m->lrd[i], nbr + i * k, k);
    free(nbr), free(dist);
    return fit_state_contamination(d, s, n, m->p.contamination);
}

static pyod_status lof_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    const lof_impl *m = d->impl;
    size_t *nb = NEW(size_t, m->k);
    if (!nb) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < x->rows; i++) {
        const double *q = &PYOD_AT(x, i, 0);
        k_nearest(&m->train, q, m->k, SIZE_MAX, nb, NULL);
        out[i] = lof_value(m, lof_lrd(m, q, nb, m->k), nb, m->k);
    }
    free(nb);
    return PYOD_OK;
}

static void lof_destroy(void *p) {
    lof_impl *m = p;
    pyod_matrix_free(&m->train);
    free(m->k_distance), free(m->lrd), free(m);
}

static const pyod_vtable LOF_VT = {"LOF", lof_fit, NULL, lof_score, lof_destroy};

pyod_lof_params pyod_lof_defaults(void) { return (pyod_lof_params){20, 0.1}; }

pyod_detector *pyod_lof_create(const pyod_lof_params *p) {
    lof_impl *m = NEW(lof_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_lof_defaults();
    return detector_new(&LOF_VT, m);
}
