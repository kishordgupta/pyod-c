/* Kernel PCA outlier detection (Hoffmann, 2007). Port of pyod.models.kpca:
 * the score is the feature-space reconstruction error. */
#include "internal.h"

typedef struct {
    pyod_kpca_params p;
    double gamma, k_fit_all;
    pyod_matrix fit_x;
    double *k_fit_rows;    /* column means of the training kernel */
    double *scaled_alphas; /* n_sel x m, eigenvectors / sqrt(lambda) */
    size_t n_sel;
} kpca_impl;

static double kern(const kpca_impl *m, const double *a, const double *b, size_t d) {
    return pyod__svm_kernel(m->p.kernel, m->p.degree, m->p.coef0, m->gamma, a, b, d);
}

static pyod_status kpca_scores(const kpca_impl *m, const pyod_matrix *x, double *out) {
    size_t mm = m->fit_x.rows, d = x->cols;
    double *k = NEW(double, mm);
    if (!k) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < x->rows; i++) {
        const double *r = &PYOD_AT(x, i, 0);
        double row_mean = 0.0;
        for (size_t t = 0; t < mm; t++) row_mean += (k[t] = kern(m, r, &PYOD_AT(&m->fit_x, t, 0), d)) / (double)mm;
        for (size_t t = 0; t < mm; t++) k[t] = k[t] - m->k_fit_rows[t] - row_mean + m->k_fit_all;
        double proj = 0.0;
        for (size_t c = 0; c < m->n_sel; c++) {
            double v = dot(m->scaled_alphas + c * mm, k, mm);
            proj += v * v;
        }
        out[i] = kern(m, r, r, d) - 2.0 * row_mean + m->k_fit_all - proj;
    }
    free(k);
    return PYOD_OK;
}

static pyod_status kpca_fit(pyod_detector *dt, const pyod_matrix *x) {
    kpca_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    pyod_matrix_free(&m->fit_x);
    free(m->k_fit_rows), free(m->scaled_alphas);
    m->k_fit_rows = m->scaled_alphas = NULL;
    if (m->p.sampling) {
        if (m->p.subset_size == 0 || m->p.subset_size > x->rows) return PYOD_ERR_PARAM;
        size_t *idx = NEW(size_t, m->p.subset_size);
        if (!idx) return PYOD_ERR_NOMEM;
        pyod_rng r;
        rng_seed(&r, m->p.random_state);
        rng_sample(&r, x->rows, m->p.subset_size, idx);
        pyod_status st = mat_rows(x, idx, m->p.subset_size, &m->fit_x);
        free(idx);
        CHECK(st);
    } else {
        CHECK(mat_copy(x, &m->fit_x));
    }
    m->gamma = m->p.gamma > 0.0 ? m->p.gamma : 1.0 / (double)x->cols;
    size_t n = m->fit_x.rows, d = x->cols;
    double *k = NEW(double, n * n), *vals = NEW(double, n), *vecs = NEW(double, n * n), *s = NEW(double, x->rows);
    m->k_fit_rows = NEW(double, n);
    pyod_status st = PYOD_ERR_NOMEM;
    if (!k || !vals || !vecs || !s || !m->k_fit_rows) goto done;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) k[i * n + j] = kern(m, &PYOD_AT(&m->fit_x, i, 0), &PYOD_AT(&m->fit_x, j, 0), d);
    m->k_fit_all = 0.0;
    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) m->k_fit_rows[j] += k[i * n + j] / (double)n;
        m->k_fit_all += m->k_fit_rows[j] / (double)n;
    }
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) k[i * n + j] = k[i * n + j] - m->k_fit_rows[j] - m->k_fit_rows[i] + m->k_fit_all;
    sym_eigen(k, n, vals, vecs);
    /* sklearn _check_psd_eigenvalues: keep positive, non-negligible eigenvalues. */
    double mx = fmax(vals[0], 0.0);
    size_t n_comp = m->p.n_components && m->p.n_components < n ? m->p.n_components : n, kept = 0;
    for (size_t c = 0; c < n_comp; c++) kept += vals[c] > 1e-12 * mx && vals[c] > 0.0;
    m->n_sel = m->p.n_selected_components && m->p.n_selected_components < kept ? m->p.n_selected_components : kept;
    m->scaled_alphas = NEW(double, m->n_sel * n);
    if (!m->scaled_alphas) goto done;
    for (size_t c = 0, t = 0; c < n_comp && t < m->n_sel; c++) {
        if (!(vals[c] > 1e-12 * mx && vals[c] > 0.0)) continue;
        for (size_t i = 0; i < n; i++) m->scaled_alphas[t * n + i] = vecs[c * n + i] / sqrt(vals[c]);
        t++;
    }
    if ((st = kpca_scores(m, x, s)) != PYOD_OK) goto done;
    st = fit_state_contamination(dt, s, x->rows, m->p.contamination);
    s = NULL;
done:
    free(k), free(vals), free(vecs), free(s);
    return st;
}

static pyod_status kpca_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return kpca_scores(d->impl, x, out); }

static void kpca_destroy(void *p) {
    kpca_impl *m = p;
    pyod_matrix_free(&m->fit_x);
    free(m->k_fit_rows), free(m->scaled_alphas), free(m);
}

static const pyod_vtable KPCA_VT = {"KPCA", kpca_fit, NULL, kpca_score, kpca_destroy};

pyod_kpca_params pyod_kpca_defaults(void) {
    return (pyod_kpca_params){0, 0, PYOD_SVM_RBF, 3, 1.0, 0.0, 0, 20, 0.1, 0};
}

pyod_detector *pyod_kpca_create(const pyod_kpca_params *p) {
    kpca_impl *m = NEW(kpca_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_kpca_defaults();
    return detector_new(&KPCA_VT, m);
}
