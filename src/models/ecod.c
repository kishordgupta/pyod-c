/* ECOD (Li et al., TKDE 2022) and COPOD (Li et al., ICDM 2020): ports of
 * pyod.models.ecod and pyod.models.copod, sharing one tail-probability routine. */
#include "internal.h"

typedef enum { AGG_MAX, AGG_SKEW_OR_MEAN } tail_agg;

typedef struct {
    double contamination;
    tail_agg agg;
    pyod_matrix train;
} ecdf_impl;

/* Score the last n_tail rows of x, using all rows of x to build the ECDFs. */
static pyod_status tail_scores(const pyod_matrix *x, size_t n_tail, tail_agg agg, double *out) {
    size_t n = x->rows, d = x->cols;
    double *left = NEW(double, n * d), *right = NEW(double, n * d), *col = NEW(double, n), *sk = NEW(double, d);
    pyod_matrix neg = {NEW(double, n * d), n, d};
    pyod_status st = PYOD_ERR_NOMEM;
    if (!left || !right || !col || !sk || !neg.data) goto done;
    for (size_t i = 0; i < n * d; i++) neg.data[i] = -x->data[i];
    column_ecdf(x, left);
    column_ecdf(&neg, right);
    for (size_t j = 0; j < d; j++) {
        for (size_t i = 0; i < n; i++) col[i] = PYOD_AT(x, i, j);
        sk[j] = sign(vec_skew(col, n));
    }
    for (size_t i = n - n_tail; i < n; i++) {
        double s = 0.0;
        for (size_t j = 0; j < d; j++) {
            double ul = -log(left[i * d + j]), ur = -log(right[i * d + j]);
            double us = ul * -sign(sk[j] - 1.0) + ur * sign(sk[j] + 1.0);
            s += agg == AGG_MAX ? fmax(fmax(ul, ur), us) : fmax(us, (ul + ur) / 2.0);
        }
        out[i - (n - n_tail)] = s;
    }
    st = PYOD_OK;
done:
    free(left), free(right), free(col), free(sk), free(neg.data);
    return st;
}

static pyod_status ecdf_fit(pyod_detector *d, const pyod_matrix *x) {
    ecdf_impl *m = d->impl;
    CHECK(check_contamination(m->contamination));
    double *s = NEW(double, x->rows);
    if (!s) return PYOD_ERR_NOMEM;
    pyod_status st = tail_scores(x, x->rows, m->agg, s);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    pyod_matrix_free(&m->train);
    CHECK(mat_copy(x, &m->train));
    return fit_state_contamination(d, s, x->rows, m->contamination);
}

/* As in PyOD, new rows are scored together with the stored training rows. */
static pyod_status ecdf_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    const ecdf_impl *m = d->impl;
    pyod_matrix all;
    CHECK(pyod_matrix_alloc(&all, m->train.rows + x->rows, x->cols));
    memcpy(all.data, m->train.data, m->train.rows * x->cols * sizeof(double));
    memcpy(all.data + m->train.rows * x->cols, x->data, x->rows * x->cols * sizeof(double));
    pyod_status st = tail_scores(&all, x->rows, m->agg, out);
    pyod_matrix_free(&all);
    return st;
}

static void ecdf_destroy(void *p) {
    ecdf_impl *m = p;
    pyod_matrix_free(&m->train);
    free(m);
}

static const pyod_vtable ECOD_VT = {"ECOD", ecdf_fit, NULL, ecdf_score, ecdf_destroy};
static const pyod_vtable COPOD_VT = {"COPOD", ecdf_fit, NULL, ecdf_score, ecdf_destroy};

pyod_ecod_params pyod_ecod_defaults(void) { return (pyod_ecod_params){0.1}; }
pyod_copod_params pyod_copod_defaults(void) { return (pyod_copod_params){0.1}; }

pyod_detector *pyod_ecod_create(const pyod_ecod_params *p) {
    ecdf_impl *m = NEW(ecdf_impl, 1);
    if (!m) return NULL;
    m->contamination = p ? p->contamination : 0.1;
    m->agg = AGG_MAX;
    return detector_new(&ECOD_VT, m);
}

pyod_detector *pyod_copod_create(const pyod_copod_params *p) {
    ecdf_impl *m = NEW(ecdf_impl, 1);
    if (!m) return NULL;
    m->contamination = p ? p->contamination : 0.1;
    m->agg = AGG_SKEW_OR_MEAN;
    return detector_new(&COPOD_VT, m);
}
