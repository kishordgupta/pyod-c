/* PCA-based outlier detection (Shyu et al., 2003). Port of pyod.models.pca:
 * the score is the sum over the selected (lowest-variance) axes of each
 * sample's distance to the axis divided by its explained-variance ratio. */
#include "internal.h"

typedef struct {
    pyod_pca_params p;
    std_scaler scaler;
    int scaled;
    double *axes;    /* n_sel x d */
    double *weights; /* n_sel */
    size_t n_sel, d;
} pca_impl;

static void pca_scores(const pca_impl *m, const pyod_matrix *x, double *out) {
    double *row = NEW(double, m->d);
    if (!row) return;
    for (size_t i = 0; i < x->rows; i++) {
        for (size_t j = 0; j < m->d; j++)
            row[j] = m->scaled ? (PYOD_AT(x, i, j) - m->scaler.mean[j]) / m->scaler.scale[j] : PYOD_AT(x, i, j);
        double s = 0.0;
        for (size_t c = 0; c < m->n_sel; c++) s += euclidean(row, m->axes + c * m->d, m->d) / m->weights[c];
        out[i] = s;
    }
    free(row);
}

static pyod_status pca_fit(pyod_detector *dt, const pyod_matrix *x) {
    pca_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t n = x->rows, d = x->cols;
    m->d = d;
    if (m->scaled) scaler_free(&m->scaler);
    m->scaled = m->p.standardization;
    pyod_matrix xs;
    CHECK(mat_copy(x, &xs));
    if (m->scaled) {
        CHECK(scaler_fit(&m->scaler, x));
        scaler_transform(&m->scaler, x, xs.data);
    }
    double *mean = NEW(double, d), *cov = NEW(double, d * d), *vals = NEW(double, d), *vecs = NEW(double, d * d);
    double *s = NEW(double, n);
    pyod_status st = PYOD_ERR_NOMEM;
    if (!mean || !cov || !vals || !vecs || !s) goto done;
    col_mean(&xs, mean);
    covariance(&xs, mean, 1, cov);
    sym_eigen(cov, d, vals, vecs);
    /* sklearn svd_flip: make the largest-magnitude entry of each axis positive. */
    for (size_t c = 0; c < d; c++) {
        double big = 0.0;
        for (size_t j = 0; j < d; j++)
            if (fabs(vecs[c * d + j]) > fabs(big)) big = vecs[c * d + j];
        if (big < 0)
            for (size_t j = 0; j < d; j++) vecs[c * d + j] = -vecs[c * d + j];
    }
    double total = 0.0;
    for (size_t c = 0; c < d; c++) total += fmax(vals[c], 0.0);
    size_t kmax = n < d ? n : d;
    size_t k = m->p.n_components ? m->p.n_components : kmax;
    size_t sel = m->p.n_selected_components ? m->p.n_selected_components : k;
    st = PYOD_ERR_PARAM;
    if (k > kmax || sel > k) goto done;
    free(m->axes), free(m->weights);
    m->axes = NEW(double, sel * d);
    m->weights = NEW(double, sel);
    st = PYOD_ERR_NOMEM;
    if (!m->axes || !m->weights) goto done;
    m->n_sel = sel;
    for (size_t t = 0; t < sel; t++) {
        size_t c = k - sel + t;
        memcpy(m->axes + t * d, vecs + c * d, d * sizeof(double));
        /* Guard against zero-variance axes, which make PyOD divide by zero. */
        m->weights[t] = m->p.weighted ? fmax(fmax(vals[c], 0.0) / total, PYOD_DBL_EPS) : 1.0;
    }
    pca_scores(m, x, s);
    st = fit_state_contamination(dt, s, n, m->p.contamination);
    s = NULL;
done:
    pyod_matrix_free(&xs);
    free(mean), free(cov), free(vals), free(vecs), free(s);
    return st;
}

static pyod_status pca_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    pca_scores(d->impl, x, out);
    return PYOD_OK;
}

static void pca_destroy(void *p) {
    pca_impl *m = p;
    if (m->scaled) scaler_free(&m->scaler);
    free(m->axes), free(m->weights), free(m);
}

static const pyod_vtable PCA_VT = {"PCA", pca_fit, NULL, pca_score, pca_destroy};

pyod_pca_params pyod_pca_defaults(void) { return (pyod_pca_params){0, 0, 1, 1, 0.1}; }

pyod_detector *pyod_pca_create(const pyod_pca_params *p) {
    pca_impl *m = NEW(pca_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_pca_defaults();
    return detector_new(&PCA_VT, m);
}
