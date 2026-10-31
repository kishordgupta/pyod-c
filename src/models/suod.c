/* SUOD: accelerating large-scale unsupervised outlier detection (Zhao et al.,
 * MLSys 2021). Port of pyod.models.suod and the suod package it wraps (without
 * the cost-based scheduler, which only affects speed). Members can be fitted on
 * a Johnson-Lindenstrauss projection and/or approximated at prediction time by
 * a random-forest regressor; member scores are standardised and combined. */
#include "internal.h"

typedef struct {
    pyod_suod_params p;
    pyod_suod_member *members;
    size_t n;
    double **proj;       /* d x k per projected member, else NULL */
    rf_regressor **approx;
    std_scaler scaler;
    int have_scaler;
    size_t d, k;
} suod_impl;

static pyod_status projected(const suod_impl *m, size_t e, const pyod_matrix *x, pyod_matrix *out) {
    if (!m->proj[e]) return mat_copy(x, out);
    CHECK(pyod_matrix_alloc(out, x->rows, m->k));
    mat_mul(x->data, m->proj[e], out->data, x->rows, x->cols, m->k);
    return PYOD_OK;
}

static void suod_clear(suod_impl *m) {
    for (size_t e = 0; e < m->n; e++) {
        if (m->proj) free(m->proj[e]);
        if (m->approx) rf_free(m->approx[e]);
    }
    free(m->proj), free(m->approx);
    m->proj = NULL, m->approx = NULL;
    if (m->have_scaler) scaler_free(&m->scaler);
    m->have_scaler = 0;
}

static void suod_combine(const suod_impl *m, pyod_matrix *cols, double *out) {
    scaler_transform(&m->scaler, cols, cols->data);
    if (m->p.combination == PYOD_COMBINE_MAX) pyod_comb_maximization(cols, out);
    else pyod_comb_average(cols, NULL, out);
}

static pyod_status suod_fit(pyod_detector *dt, const pyod_matrix *x) {
    suod_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    if (m->n == 0 || !(m->p.target_dim_frac > 0.0 && m->p.target_dim_frac <= 1.0)) return PYOD_ERR_PARAM;
    suod_clear(m);
    size_t n = x->rows, d = x->cols;
    m->d = d;
    m->k = (size_t)(m->p.target_dim_frac * (double)d);
    if (m->k < 1) m->k = 1;
    m->proj = NEW(double *, m->n);
    m->approx = NEW(rf_regressor *, m->n);
    pyod_matrix cols;
    if (!m->proj || !m->approx) return PYOD_ERR_NOMEM;
    CHECK(pyod_matrix_alloc(&cols, n, m->n));
    pyod_rng r;
    rng_seed(&r, m->p.random_state);
    pyod_status st = PYOD_OK;
    for (size_t e = 0; e < m->n && st == PYOD_OK; e++) {
        if (m->members[e].project) { /* JL "basic": N(0, 1) / sqrt(k), shape d x k */
            m->proj[e] = NEW(double, d * m->k);
            if (!m->proj[e]) {
                st = PYOD_ERR_NOMEM;
                break;
            }
            for (size_t i = 0; i < d * m->k; i++) m->proj[e][i] = rng_normal(&r) / sqrt((double)m->k);
        }
        pyod_matrix xp;
        if ((st = projected(m, e, x, &xp)) != PYOD_OK) break;
        st = pyod_fit(m->members[e].detector, &xp);
        const double *s = st == PYOD_OK ? pyod_decision_scores(m->members[e].detector, NULL) : NULL;
        if (s && m->members[e].approximate) {
            m->approx[e] = rf_fit(&xp, s, 50, rng_u64(&r));
            if (!m->approx[e]) st = PYOD_ERR_NOMEM;
        }
        if (s)
            for (size_t i = 0; i < n; i++) PYOD_AT(&cols, i, e) = s[i];
        pyod_matrix_free(&xp);
    }
    double *out = NEW(double, n);
    if (st == PYOD_OK && (!out || scaler_fit(&m->scaler, &cols) != PYOD_OK)) st = PYOD_ERR_NOMEM;
    if (st != PYOD_OK) {
        free(out);
        pyod_matrix_free(&cols);
        return st;
    }
    m->have_scaler = 1;
    suod_combine(m, &cols, out);
    pyod_matrix_free(&cols);
    return fit_state_contamination(dt, out, n, m->p.contamination);
}

static pyod_status suod_score(const pyod_detector *dt, const pyod_matrix *x, double *out) {
    const suod_impl *m = dt->impl;
    pyod_matrix cols;
    CHECK(pyod_matrix_alloc(&cols, x->rows, m->n));
    double *buf = NEW(double, x->rows);
    pyod_status st = buf ? PYOD_OK : PYOD_ERR_NOMEM;
    for (size_t e = 0; e < m->n && st == PYOD_OK; e++) {
        pyod_matrix xp;
        if ((st = projected(m, e, x, &xp)) != PYOD_OK) break;
        if (m->approx[e]) rf_predict(m->approx[e], &xp, buf);
        else st = pyod_decision_function(m->members[e].detector, &xp, buf);
        for (size_t i = 0; i < x->rows; i++) PYOD_AT(&cols, i, e) = buf[i];
        pyod_matrix_free(&xp);
    }
    if (st == PYOD_OK) suod_combine(m, &cols, out);
    free(buf);
    pyod_matrix_free(&cols);
    return st;
}

static void suod_destroy(void *p) {
    suod_impl *m = p;
    suod_clear(m);
    for (size_t e = 0; e < m->n; e++) pyod_free(m->members[e].detector);
    free(m->members);
    free(m);
}

static const pyod_vtable SUOD_VT = {"SUOD", suod_fit, NULL, suod_score, suod_destroy};

pyod_suod_params pyod_suod_defaults(void) { return (pyod_suod_params){0.5, PYOD_COMBINE_AVERAGE, 0.1, 0}; }

pyod_detector *pyod_suod_create(pyod_suod_member *members, size_t n, const pyod_suod_params *p) {
    suod_impl *m = NEW(suod_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_suod_defaults();
    if (members && n) {
        m->members = NEW(pyod_suod_member, n);
        if (!m->members) {
            free(m);
            return NULL;
        }
        memcpy(m->members, members, n * sizeof(pyod_suod_member));
        m->n = n;
    } else { /* PyOD's default members, with suod's default acceleration flags */
        m->members = NEW(pyod_suod_member, 8);
        if (!m->members) {
            free(m);
            return NULL;
        }
        pyod_lof_params l = pyod_lof_defaults();
        pyod_hbos_params h = pyod_hbos_defaults();
        pyod_iforest_params f = pyod_iforest_defaults();
        l.n_neighbors = 15;
        m->members[0] = (pyod_suod_member){pyod_lof_create(&l), 1, 1};
        l.n_neighbors = 20;
        m->members[1] = (pyod_suod_member){pyod_lof_create(&l), 1, 1};
        m->members[2] = (pyod_suod_member){pyod_hbos_create(&h), 0, 0};
        h.n_bins = 20;
        m->members[3] = (pyod_suod_member){pyod_hbos_create(&h), 0, 0};
        m->members[4] = (pyod_suod_member){pyod_copod_create(NULL), 0, 0};
        size_t trees[3] = {50, 100, 150};
        for (int t = 0; t < 3; t++) {
            f.n_estimators = trees[t];
            f.random_state = (uint64_t)t + 1;
            m->members[5 + t] = (pyod_suod_member){pyod_iforest_create(&f), 0, 0};
        }
        m->n = 8;
        for (size_t e = 0; e < 8; e++)
            if (!m->members[e].detector) {
                suod_destroy(m);
                return NULL;
            }
    }
    return detector_new(&SUOD_VT, m);
}
