/* XGBOD: improving supervised outlier detection with unsupervised
 * representation learning (Zhao & Hryniewicki, IJCNN 2018). Port of
 * pyod.models.xgbod: a bank of unsupervised detectors adds outlier scores as
 * features before a supervised classifier is trained. */
#include "internal.h"

typedef struct {
    pyod_xgbod_params p;
    pyod_detector **members;
    int *standardize;
    size_t *k_req; /* default bank: neighbours a KNN/LOF member needs (0 = none) */
    size_t n;
    pyod_classifier clf;
    std_scaler scaler;
    int have_scaler;
} xgbod_impl;

/* x with each member's scores appended as extra columns (allocated). */
static pyod_status augment(const xgbod_impl *m, const pyod_matrix *x, int training, pyod_matrix *out) {
    size_t n = x->rows, d = x->cols;
    CHECK(pyod_matrix_alloc(out, n, d + m->n));
    pyod_matrix xn;
    pyod_status st = mat_copy(x, &xn);
    double *buf = NEW(double, n);
    if (st != PYOD_OK || !buf) {
        free(buf);
        pyod_matrix_free(&xn);
        pyod_matrix_free(out);
        return st != PYOD_OK ? st : PYOD_ERR_NOMEM;
    }
    scaler_transform(&m->scaler, x, xn.data);
    for (size_t i = 0; i < n; i++) memcpy(&PYOD_AT(out, i, 0), &PYOD_AT(x, i, 0), d * sizeof(double));
    for (size_t e = 0; e < m->n && st == PYOD_OK; e++) {
        const double *s = buf;
        if (training) s = pyod_decision_scores(m->members[e], NULL);
        else st = pyod_decision_function(m->members[e], m->standardize[e] ? &xn : x, buf);
        for (size_t i = 0; st == PYOD_OK && i < n; i++) PYOD_AT(out, i, d + e) = s[i];
    }
    free(buf);
    pyod_matrix_free(&xn);
    if (st != PYOD_OK) pyod_matrix_free(out);
    return st;
}

static pyod_status xgbod_fit_supervised(pyod_detector *dt, const pyod_matrix *x, const uint8_t *y) {
    xgbod_impl *m = dt->impl;
    if (!m->clf.self || m->n == 0) return PYOD_ERR_PARAM;
    /* Like PyOD, keep only the default KNN / LOF members with k < n. */
    if (m->k_req) {
        size_t kept = 0;
        for (size_t e = 0; e < m->n; e++) {
            if (m->k_req[e] && m->k_req[e] >= x->rows) {
                pyod_free(m->members[e]);
                continue;
            }
            m->members[kept] = m->members[e];
            m->standardize[kept] = m->standardize[e];
            m->k_req[kept++] = m->k_req[e];
        }
        m->n = kept;
    }
    if (m->have_scaler) scaler_free(&m->scaler);
    CHECK(scaler_fit(&m->scaler, x));
    m->have_scaler = 1;
    pyod_matrix xn;
    CHECK(mat_copy(x, &xn));
    scaler_transform(&m->scaler, x, xn.data);
    pyod_status st = PYOD_OK;
    for (size_t e = 0; e < m->n && st == PYOD_OK; e++) st = pyod_fit(m->members[e], m->standardize[e] ? &xn : x);
    pyod_matrix_free(&xn);
    CHECK(st);
    pyod_matrix xa;
    CHECK(augment(m, x, 1, &xa));
    double *s = NEW(double, x->rows);
    st = s ? m->clf.fit(m->clf.self, &xa, y) : PYOD_ERR_NOMEM;
    if (st == PYOD_OK) st = m->clf.predict_proba(m->clf.self, &xa, s);
    pyod_matrix_free(&xa);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_threshold(dt, s, x->rows, 0.5);
}

static pyod_status xgbod_score(const pyod_detector *dt, const pyod_matrix *x, double *out) {
    const xgbod_impl *m = dt->impl;
    pyod_matrix xa;
    CHECK(augment(m, x, 0, &xa));
    pyod_status st = m->clf.predict_proba(m->clf.self, &xa, out);
    pyod_matrix_free(&xa);
    return st;
}

static void xgbod_destroy(void *p) {
    xgbod_impl *m = p;
    for (size_t e = 0; e < m->n; e++) pyod_free(m->members[e]);
    free(m->members), free(m->standardize), free(m->k_req);
    if (m->clf.destroy) m->clf.destroy(m->clf.self);
    if (m->have_scaler) scaler_free(&m->scaler);
    free(m);
}

/* fit without labels is unsupported: XGBOD is supervised. */
static const pyod_vtable XGBOD_VT = {"XGBOD", NULL, xgbod_fit_supervised, xgbod_score, xgbod_destroy};

pyod_xgbod_params pyod_xgbod_defaults(void) { return (pyod_xgbod_params){0}; }

/* Port of _init_detectors. All KNN / LOF sizes are created here; fit drops
 * those with k >= n_samples, as PyOD does. */
static int default_bank(xgbod_impl *m, uint64_t seed) {
    size_t ks[8] = {1, 3, 5, 10, 20, 30, 40, 50}, bins[7] = {5, 10, 15, 20, 25, 30, 50}, trees[7] = {10, 20, 50, 70, 100, 150, 200};
    double nus[11] = {0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99};
    size_t cap = 16 + 7 + 11 + 7;
    m->members = NEW(pyod_detector *, cap);
    m->standardize = NEW(int, cap);
    m->k_req = NEW(size_t, cap);
    if (!m->members || !m->standardize || !m->k_req) return 0;
    for (int i = 0; i < 8; i++) {
        pyod_knn_params kp = pyod_knn_defaults();
        pyod_lof_params lp = pyod_lof_defaults();
        kp.n_neighbors = lp.n_neighbors = ks[i];
        m->standardize[m->n] = 1;
        m->k_req[m->n] = ks[i];
        m->members[m->n++] = pyod_knn_create(&kp);
        m->standardize[m->n] = 1;
        m->k_req[m->n] = ks[i];
        m->members[m->n++] = pyod_lof_create(&lp);
    }
    for (int i = 0; i < 7; i++) {
        pyod_hbos_params hp = pyod_hbos_defaults();
        hp.n_bins = bins[i];
        m->standardize[m->n] = 0;
        m->members[m->n++] = pyod_hbos_create(&hp);
    }
    for (int i = 0; i < 11; i++) {
        pyod_ocsvm_params op = pyod_ocsvm_defaults();
        op.nu = nus[i];
        m->standardize[m->n] = 1;
        m->members[m->n++] = pyod_ocsvm_create(&op);
    }
    for (int i = 0; i < 7; i++) {
        pyod_iforest_params fp = pyod_iforest_defaults();
        fp.n_estimators = trees[i];
        fp.random_state = seed;
        m->standardize[m->n] = 0;
        m->members[m->n++] = pyod_iforest_create(&fp);
    }
    for (size_t e = 0; e < m->n; e++)
        if (!m->members[e]) return 0;
    return 1;
}

pyod_detector *pyod_xgbod_create(pyod_detector **members, const int *standardize, size_t n,
                                 const pyod_classifier *classifier, const pyod_xgbod_params *p) {
    xgbod_impl *m = NEW(xgbod_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_xgbod_defaults();
    if (classifier) {
        m->clf = *classifier;
    } else {
        pyod_gbdt_params gp = pyod_gbdt_defaults();
        gp.max_depth = 3;
        gp.learning_rate = 0.1;
        gp.random_state = m->p.random_state;
        m->clf = pyod_gbdt_classifier(&gp);
    }
    if (members && n) {
        m->members = NEW(pyod_detector *, n);
        m->standardize = NEW(int, n);
        if (m->members && m->standardize) {
            memcpy(m->members, members, n * sizeof(pyod_detector *));
            for (size_t i = 0; i < n; i++) m->standardize[i] = standardize ? standardize[i] : 1;
            m->n = n;
        }
    } else if (!default_bank(m, m->p.random_state)) {
        xgbod_destroy(m);
        return NULL;
    }
    if (!m->clf.self || !m->members) {
        xgbod_destroy(m);
        return NULL;
    }
    return detector_new(&XGBOD_VT, m);
}
