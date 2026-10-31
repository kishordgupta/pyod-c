/* R-Graph: outlier detection by random walks on an elastic-net
 * self-representation graph (You et al., CVPR 2017). Port of pyod.models.rgraph.
 * PyOD solves the lasso with LARS; we use coordinate descent to the same optimum. */
#include "internal.h"

typedef struct {
    pyod_rgraph_params p;
    std_scaler scaler;
    int scaled;
    pyod_matrix train;
} rgraph_impl;

/* min_c 0.5 ||y - sum_j c_j atoms_j||^2 + alpha ||c||_1 (atoms are rows of x). */
static void lasso(const pyod_matrix *x, const size_t *atoms, size_t m, const double *y, double alpha, size_t max_iter, double *c) {
    size_t d = x->cols;
    double *gram = NEW(double, m * m), *b = NEW(double, m);
    if (!gram || !b) {
        free(gram), free(b);
        return;
    }
    for (size_t i = 0; i < m; i++) {
        b[i] = dot(&PYOD_AT(x, atoms[i], 0), y, d);
        for (size_t j = 0; j < m; j++) gram[i * m + j] = dot(&PYOD_AT(x, atoms[i], 0), &PYOD_AT(x, atoms[j], 0), d);
        c[i] = 0.0;
    }
    /* Coordinate descent needs more sweeps than LARS needs steps to reach the optimum. */
    for (size_t it = 0; it < (max_iter ? max_iter : 1) * 10; it++) {
        double max_delta = 0.0;
        for (size_t j = 0; j < m; j++) {
            if (gram[j * m + j] == 0.0) continue;
            double rho = b[j];
            for (size_t k = 0; k < m; k++)
                if (k != j) rho -= gram[j * m + k] * c[k];
            double nv = sign(rho) * fmax(fabs(rho) - alpha, 0.0) / gram[j * m + j];
            max_delta = fmax(max_delta, fabs(nv - c[j]));
            c[j] = nv;
        }
        if (max_delta < 1e-12) break;
    }
    free(gram), free(b);
}

/* Indices of the k largest values. */
static void top_k(const double *v, size_t n, size_t k, size_t *out) {
    size_t *idx = NEW(size_t, n);
    double *neg = NEW(double, n);
    if (idx && neg) {
        for (size_t i = 0; i < n; i++) neg[i] = -v[i];
        argsort(neg, n, idx);
        memcpy(out, idx, k * sizeof(size_t));
    }
    free(idx), free(neg);
}

/* Port of active_support_elastic_net (tau = 1, i.e. lasso); c has x->rows entries. */
static void active_support_lasso(const rgraph_impl *m, const pyod_matrix *x, const double *y, double alpha, double *c) {
    size_t n = x->rows, d = x->cols, support_size = m->p.support_size, ns;
    size_t *supp = NEW(size_t, n), *added = NEW(size_t, n), *active = NEW(size_t, n), *ord = NEW(size_t, n);
    double *cs = NEW(double, n), *coh = NEW(double, n), *delta = NEW(double, d), *tmp = NEW(double, n);
    if (!supp || !added || !active || !ord || !cs || !coh || !delta || !tmp) goto done;
    if (n <= support_size) {
        for (size_t i = 0; i < n; i++) supp[i] = i;
        ns = n;
    } else {
        /* L2 initialisation: ridge solution, projected back onto the samples. */
        double *a = NEW(double, d * d), *ainv = NEW(double, d * d), *w = NEW(double, d);
        if (!a || !ainv || !w) {
            free(a), free(ainv), free(w);
            goto done;
        }
        for (size_t r = 0; r < n; r++)
            for (size_t i = 0; i < d; i++)
                for (size_t j = 0; j < d; j++) a[i * d + j] += PYOD_AT(x, r, i) * PYOD_AT(x, r, j);
        for (size_t i = 0; i < d; i++) a[i * d + i] += alpha;
        pinvh(a, d, ainv);
        for (size_t i = 0; i < d; i++) w[i] = dot(ainv + i * d, y, d);
        for (size_t r = 0; r < n; r++) tmp[r] = fabs(dot(&PYOD_AT(x, r, 0), w, d));
        top_k(tmp, n, support_size, supp);
        ns = support_size;
        free(a), free(ainv), free(w);
    }
    double curr_obj = INFINITY;
    for (size_t it = 0; it < m->p.maxiter; it++) {
        lasso(x, supp, ns, y, alpha, m->p.maxiter_lasso, cs);
        double l1 = 0.0, l2 = 0.0;
        for (size_t k = 0; k < d; k++) {
            double rec = 0.0;
            for (size_t t = 0; t < ns; t++) rec += PYOD_AT(x, supp[t], k) * cs[t];
            delta[k] = (y[k] - rec) / alpha;
            l2 += delta[k] * delta[k];
        }
        for (size_t t = 0; t < ns; t++) l1 += fabs(cs[t]);
        double obj = l1 + alpha / 2.0 * l2;
        if (curr_obj - obj < 1e-10 * curr_obj) break;
        curr_obj = obj;
        for (size_t r = 0; r < n; r++) coh[r] = fabs(dot(delta, &PYOD_AT(x, r, 0), d));
        for (size_t t = 0; t < ns; t++) coh[supp[t]] = 0.0;
        size_t na = 0, nact = 0;
        for (size_t r = 0; r < n; r++)
            if (coh[r] > 1.0 + 1e-10) added[na++] = r;
        if (!na) break;
        for (size_t t = 0; t < ns; t++)
            if (fabs(cs[t]) > 1e-10) active[nact++] = supp[t];
        if ((double)nact > 0.8 * (double)support_size) {
            size_t grown = (size_t)round((double)(nact > support_size ? nact : support_size) * 1.1);
            support_size = grown < n ? grown : n;
        }
        if (na + nact > support_size) {
            size_t keep = support_size - nact;
            for (size_t t = 0; t < na; t++) tmp[t] = coh[added[t]];
            top_k(tmp, na, keep, ord);
            for (size_t t = 0; t < keep; t++) ord[t] = added[ord[t]];
            memcpy(added, ord, keep * sizeof(size_t));
            na = keep;
        }
        memcpy(supp, active, nact * sizeof(size_t));
        memcpy(supp + nact, added, na * sizeof(size_t));
        ns = nact + na;
    }
    for (size_t r = 0; r < n; r++) c[r] = 0.0;
    for (size_t t = 0; t < ns; t++) c[supp[t]] = cs[t];
done:
    free(supp), free(added), free(active), free(ord), free(cs), free(coh), free(delta), free(tmp);
}

/* Port of _decision_function: negative mean visiting probability of a random walk. */
static pyod_status rgraph_scores(const rgraph_impl *m, const pyod_matrix *xn_in, double *out) {
    size_t n = xn_in->rows, d = xn_in->cols;
    pyod_matrix x;
    CHECK(mat_copy(xn_in, &x));
    double *trans = NEW(double, n * n), *y = NEW(double, d), *c = NEW(double, n), *absc = NEW(double, n);
    double *pi = NEW(double, n), *next = NEW(double, n), *bar = NEW(double, n);
    size_t *nz = NEW(size_t, n), *all = NEW(size_t, n);
    pyod_status st = PYOD_ERR_NOMEM;
    if (!trans || !y || !c || !absc || !pi || !next || !bar || !nz || !all) goto done;
    for (size_t i = 0; i < n; i++) all[i] = i;
    for (size_t i = 0; i < n; i++) {
        memcpy(y, &PYOD_AT(&x, i, 0), d * sizeof(double));
        memset(&PYOD_AT(&x, i, 0), 0, d * sizeof(double));
        double alpha = 0.0;
        for (size_t j = 0; j < n; j++)
            if (j != i) alpha = fmax(alpha, fabs(dot(&PYOD_AT(&x, j, 0), y, d)));
        alpha /= m->p.gamma;
        if (alpha == 0.0) memset(c, 0, n * sizeof(double));
        else if (m->p.active_support) active_support_lasso(m, &x, y, alpha, c);
        else lasso(&x, all, n, y, alpha, m->p.maxiter_lasso, c);
        size_t k = 0;
        for (size_t j = 0; j < n; j++) absc[j] = fabs(c[j]);
        for (size_t j = 0; j < n; j++)
            if (c[j] != 0.0) k++;
        if (k > m->p.n_nonzero) {
            top_k(absc, n, m->p.n_nonzero, nz);
            k = m->p.n_nonzero;
        } else {
            for (size_t j = 0, t = 0; j < n; j++)
                if (c[j] != 0.0) nz[t++] = j;
        }
        double total = 0.0;
        for (size_t t = 0; t < k; t++) total += absc[nz[t]];
        if (total > 0.0)
            for (size_t t = 0; t < k; t++) trans[i * n + nz[t]] = absc[nz[t]] / total;
        memcpy(&PYOD_AT(&x, i, 0), y, d * sizeof(double));
    }
    for (size_t i = 0; i < n; i++) pi[i] = 1.0 / (double)n, bar[i] = 0.0;
    for (size_t s = 0; s < m->p.transition_steps; s++) {
        memset(next, 0, n * sizeof(double));
        for (size_t i = 0; i < n; i++)
            if (pi[i] != 0.0)
                for (size_t j = 0; j < n; j++) next[j] += pi[i] * trans[i * n + j];
        memcpy(pi, next, n * sizeof(double));
        for (size_t i = 0; i < n; i++) bar[i] += pi[i];
    }
    for (size_t i = 0; i < n; i++) out[i] = -bar[i] / (double)m->p.transition_steps;
    st = PYOD_OK;
done:
    pyod_matrix_free(&x);
    free(trans), free(y), free(c), free(absc), free(pi), free(next), free(bar), free(nz), free(all);
    return st;
}

static pyod_status normalized_scores(const rgraph_impl *m, const pyod_matrix *x, double *out) {
    pyod_matrix xn;
    CHECK(mat_copy(x, &xn));
    if (m->scaled) scaler_transform(&m->scaler, x, xn.data);
    pyod_status st = rgraph_scores(m, &xn, out);
    pyod_matrix_free(&xn);
    return st;
}

static pyod_status rgraph_fit(pyod_detector *d, const pyod_matrix *x) {
    rgraph_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    if (m->scaled) scaler_free(&m->scaler);
    m->scaled = m->p.preprocessing;
    if (m->scaled) CHECK(scaler_fit(&m->scaler, x));
    double *s = NEW(double, x->rows);
    if (!s) return PYOD_ERR_NOMEM;
    pyod_status st = normalized_scores(m, x, s);
    pyod_matrix_free(&m->train);
    if (st == PYOD_OK) st = mat_copy(x, &m->train);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(d, s, x->rows, m->p.contamination);
}

/* Each block of test samples is scored jointly with the training data. */
static pyod_status rgraph_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    const rgraph_impl *m = d->impl;
    size_t bs = m->p.blocksize_test_data ? m->p.blocksize_test_data : 1, nt = m->train.rows, dd = x->cols;
    for (size_t start = 0; start < x->rows; start += bs) {
        size_t len = x->rows - start < bs ? x->rows - start : bs;
        pyod_matrix all;
        CHECK(pyod_matrix_alloc(&all, nt + len, dd));
        memcpy(all.data, m->train.data, nt * dd * sizeof(double));
        memcpy(all.data + nt * dd, &PYOD_AT(x, start, 0), len * dd * sizeof(double));
        double *s = NEW(double, nt + len);
        pyod_status st = s ? normalized_scores(m, &all, s) : PYOD_ERR_NOMEM;
        if (st == PYOD_OK) memcpy(out + start, s + nt, len * sizeof(double));
        free(s);
        pyod_matrix_free(&all);
        CHECK(st);
    }
    return PYOD_OK;
}

static void rgraph_destroy(void *p) {
    rgraph_impl *m = p;
    if (m->scaled) scaler_free(&m->scaler);
    pyod_matrix_free(&m->train);
    free(m);
}

static const pyod_vtable RGRAPH_VT = {"RGraph", rgraph_fit, NULL, rgraph_score, rgraph_destroy};

pyod_rgraph_params pyod_rgraph_defaults(void) {
    return (pyod_rgraph_params){10, 10, 100, 40, 1000, 10, 50.0, 0.1, 1, 1};
}

pyod_detector *pyod_rgraph_create(const pyod_rgraph_params *p) {
    rgraph_impl *m = NEW(rgraph_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_rgraph_defaults();
    return detector_new(&RGRAPH_VT, m);
}
