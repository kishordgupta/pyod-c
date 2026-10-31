/* One-class SVM (Schölkopf et al., 2001). Port of pyod.models.ocsvm (sklearn
 * OneClassSVM, i.e. libsvm): SMO with second-order working-set selection;
 * the score is the negated decision value. */
#include "internal.h"

typedef struct {
    pyod_ocsvm_params p;
    double gamma, rho;
    pyod_matrix sv;
    double *coef;
} ocsvm_impl;

static double kernel(const ocsvm_impl *m, const double *a, const double *b, size_t d) {
    switch (m->p.kernel) {
    case PYOD_SVM_LINEAR: return dot(a, b, d);
    case PYOD_SVM_POLY: return pow(m->gamma * dot(a, b, d) + m->p.coef0, m->p.degree);
    case PYOD_SVM_SIGMOID: return tanh(m->gamma * dot(a, b, d) + m->p.coef0);
    default: return exp(-m->gamma * sq_euclidean(a, b, d));
    }
}

/* Kernel row i (precomputed matrix if available, else computed into buf). */
static const double *qrow(const ocsvm_impl *m, const pyod_matrix *x, const double *full, size_t i, double *buf) {
    if (full) return full + i * x->rows;
    for (size_t j = 0; j < x->rows; j++) buf[j] = kernel(m, &PYOD_AT(x, i, 0), &PYOD_AT(x, j, 0), x->cols);
    return buf;
}

/* libsvm solve_one_class: min 0.5 a'Qa  s.t. 0 <= a_i <= 1, sum a = nu * l. */
static pyod_status smo(ocsvm_impl *m, const pyod_matrix *x, double *alpha) {
    size_t l = x->rows, d = x->cols;
    const double TAU = 1e-12;
    double *full = l <= 6000 ? NEW(double, l * l) : NULL;
    double *qd = NEW(double, l), *g = NEW(double, l), *bi = NEW(double, l), *bj = NEW(double, l);
    if (!qd || !g || !bi || !bj) {
        free(full), free(qd), free(g), free(bi), free(bj);
        return PYOD_ERR_NOMEM;
    }
    if (full)
        for (size_t i = 0; i < l; i++)
            for (size_t j = 0; j < l; j++) full[i * l + j] = kernel(m, &PYOD_AT(x, i, 0), &PYOD_AT(x, j, 0), d);
    for (size_t i = 0; i < l; i++) qd[i] = kernel(m, &PYOD_AT(x, i, 0), &PYOD_AT(x, i, 0), d);
    double total = m->p.nu * (double)l;
    size_t n_full = (size_t)total;
    for (size_t i = 0; i < l; i++) alpha[i] = i < n_full ? 1.0 : 0.0;
    if (n_full < l) alpha[n_full] = total - (double)n_full;
    for (size_t i = 0; i < l; i++) {
        if (alpha[i] <= 0.0) continue;
        const double *qi = qrow(m, x, full, i, bi);
        for (size_t k = 0; k < l; k++) g[k] += alpha[i] * qi[k];
    }
    size_t max_iter = m->p.max_iter ? m->p.max_iter : (100 * l > 10000000 ? 100 * l : 10000000);
    for (size_t it = 0; it < max_iter; it++) {
        double gmax = -INFINITY, gmax2 = -INFINITY, obj_min = INFINITY;
        size_t i = SIZE_MAX, j = SIZE_MAX;
        for (size_t t = 0; t < l; t++)
            if (alpha[t] < 1.0 && -g[t] >= gmax) gmax = -g[t], i = t;
        if (i == SIZE_MAX) break;
        const double *qi = qrow(m, x, full, i, bi);
        for (size_t t = 0; t < l; t++) {
            if (alpha[t] <= 0.0) continue;
            double grad_diff = gmax + g[t];
            if (g[t] >= gmax2) gmax2 = g[t];
            if (grad_diff > 0.0) {
                double quad = qd[i] + qd[t] - 2.0 * qi[t];
                double obj = -(grad_diff * grad_diff) / (quad > 0.0 ? quad : TAU);
                if (obj <= obj_min) j = t, obj_min = obj;
            }
        }
        if (gmax + gmax2 < m->p.tol || j == SIZE_MAX) break;
        const double *qj = qrow(m, x, full, j, bj);
        double old_i = alpha[i], old_j = alpha[j];
        double quad = qd[i] + qd[j] - 2.0 * qi[j];
        if (quad <= 0.0) quad = TAU;
        double delta = (g[i] - g[j]) / quad, sum = alpha[i] + alpha[j];
        alpha[i] -= delta;
        alpha[j] += delta;
        if (sum > 1.0) {
            if (alpha[i] > 1.0) alpha[i] = 1.0, alpha[j] = sum - 1.0;
        } else if (alpha[j] < 0.0) {
            alpha[j] = 0.0, alpha[i] = sum;
        }
        if (sum > 1.0) {
            if (alpha[j] > 1.0) alpha[j] = 1.0, alpha[i] = sum - 1.0;
        } else if (alpha[i] < 0.0) {
            alpha[i] = 0.0, alpha[j] = sum;
        }
        double di = alpha[i] - old_i, dj = alpha[j] - old_j;
        for (size_t k = 0; k < l; k++) g[k] += qi[k] * di + qj[k] * dj;
    }
    /* libsvm calculate_rho */
    double ub = INFINITY, lb = -INFINITY, sum_free = 0.0;
    size_t n_free = 0;
    for (size_t t = 0; t < l; t++) {
        if (alpha[t] >= 1.0) lb = fmax(lb, g[t]);
        else if (alpha[t] <= 0.0) ub = fmin(ub, g[t]);
        else n_free++, sum_free += g[t];
    }
    m->rho = n_free ? sum_free / (double)n_free : (ub + lb) / 2.0;
    free(full), free(qd), free(g), free(bi), free(bj);
    return PYOD_OK;
}

static void ocsvm_scores(const ocsvm_impl *m, const pyod_matrix *x, double *out) {
    for (size_t i = 0; i < x->rows; i++) {
        double f = 0.0;
        for (size_t s = 0; s < m->sv.rows; s++) f += m->coef[s] * kernel(m, &PYOD_AT(&m->sv, s, 0), &PYOD_AT(x, i, 0), x->cols);
        out[i] = -(f - m->rho);
    }
}

static pyod_status ocsvm_fit(pyod_detector *d, const pyod_matrix *x) {
    ocsvm_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    if (!(m->p.nu > 0.0 && m->p.nu <= 1.0)) return PYOD_ERR_PARAM;
    m->gamma = m->p.gamma > 0.0 ? m->p.gamma : 1.0 / (double)x->cols;
    size_t l = x->rows;
    double *alpha = NEW(double, l);
    if (!alpha) return PYOD_ERR_NOMEM;
    pyod_status st = smo(m, x, alpha);
    if (st != PYOD_OK) {
        free(alpha);
        return st;
    }
    size_t nsv = 0;
    for (size_t i = 0; i < l; i++) nsv += alpha[i] > 0.0;
    pyod_matrix_free(&m->sv);
    free(m->coef);
    m->coef = NEW(double, nsv);
    if (!m->coef || pyod_matrix_alloc(&m->sv, nsv, x->cols) != PYOD_OK) {
        free(alpha);
        return PYOD_ERR_NOMEM;
    }
    for (size_t i = 0, k = 0; i < l; i++)
        if (alpha[i] > 0.0) {
            memcpy(&PYOD_AT(&m->sv, k, 0), &PYOD_AT(x, i, 0), x->cols * sizeof(double));
            m->coef[k++] = alpha[i];
        }
    free(alpha);
    double *s = NEW(double, l);
    if (!s) return PYOD_ERR_NOMEM;
    ocsvm_scores(m, x, s);
    return fit_state_contamination(d, s, l, m->p.contamination);
}

static pyod_status ocsvm_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    ocsvm_scores(d->impl, x, out);
    return PYOD_OK;
}

static void ocsvm_destroy(void *p) {
    ocsvm_impl *m = p;
    pyod_matrix_free(&m->sv);
    free(m->coef), free(m);
}

static const pyod_vtable OCSVM_VT = {"OCSVM", ocsvm_fit, NULL, ocsvm_score, ocsvm_destroy};

pyod_ocsvm_params pyod_ocsvm_defaults(void) { return (pyod_ocsvm_params){PYOD_SVM_RBF, 3, 0.0, 0.0, 0.5, 1e-3, 0.1, 0}; }

pyod_detector *pyod_ocsvm_create(const pyod_ocsvm_params *p) {
    ocsvm_impl *m = NEW(ocsvm_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_ocsvm_defaults();
    return detector_new(&OCSVM_VT, m);
}

/* Shared with KPCA. */
double pyod__svm_kernel(pyod_svm_kernel k, int degree, double coef0, double gamma, const double *a, const double *b, size_t d) {
    ocsvm_impl m;
    memset(&m, 0, sizeof m);
    m.p.kernel = k;
    m.p.degree = degree;
    m.p.coef0 = coef0;
    m.gamma = gamma;
    return kernel(&m, a, b, d);
}
