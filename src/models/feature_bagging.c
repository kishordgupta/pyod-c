/* Feature Bagging (Lazarevic & Kumar, KDD 2005). Port of
 * pyod.models.feature_bagging: members are fitted on random feature subsets and
 * their raw scores averaged (or maximised). */
#include "internal.h"

typedef struct {
    pyod_feature_bagging_params p;
    pyod_detector **members;
    size_t **features, *n_feat, n_members;
} fb_impl;

size_t bagging_indices(pyod_rng *r, int bootstrap, size_t n_features, size_t min, size_t max, size_t *out) {
    size_t span = max > min ? max - min : 1;
    size_t k = min + rng_below(r, span);
    if (bootstrap)
        for (size_t i = 0; i < k; i++) out[i] = rng_below(r, n_features);
    else
        rng_sample(r, n_features, k, out);
    return k;
}

static pyod_detector *default_lof(uint64_t seed, void *ctx) {
    (void)seed, (void)ctx;
    return pyod_lof_create(NULL);
}

static void fb_clear(fb_impl *m) {
    for (size_t i = 0; i < m->n_members; i++) {
        pyod_free(m->members[i]);
        free(m->features[i]);
    }
    free(m->members), free(m->features), free(m->n_feat);
    m->members = NULL, m->features = NULL, m->n_feat = NULL, m->n_members = 0;
}

static pyod_status member_scores(const fb_impl *m, const pyod_matrix *x, pyod_matrix *cols, int training) {
    CHECK(pyod_matrix_alloc(cols, x->rows, m->n_members));
    double *buf = NEW(double, x->rows);
    if (!buf) return PYOD_ERR_NOMEM;
    for (size_t e = 0; e < m->n_members; e++) {
        const double *s = buf;
        if (training) {
            s = pyod_decision_scores(m->members[e], NULL);
        } else {
            pyod_matrix sub;
            pyod_status st = mat_cols(x, m->features[e], m->n_feat[e], &sub);
            if (st == PYOD_OK) st = pyod_decision_function(m->members[e], &sub, buf);
            pyod_matrix_free(&sub);
            if (st != PYOD_OK) {
                free(buf);
                return st;
            }
        }
        for (size_t i = 0; i < x->rows; i++) PYOD_AT(cols, i, e) = s[i];
    }
    free(buf);
    return PYOD_OK;
}

static void combine(pyod_combination c, const pyod_matrix *cols, double *out) {
    if (c == PYOD_COMBINE_MAX) pyod_comb_maximization(cols, out);
    else pyod_comb_average(cols, NULL, out);
}

static pyod_status fb_fit(pyod_detector *dt, const pyod_matrix *x) {
    fb_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t d = x->cols, min = d / 2, max = (size_t)(m->p.max_features * (double)d), T = m->p.n_estimators;
    if (min == 0 || max < min || T == 0) return PYOD_ERR_PARAM;
    fb_clear(m);
    m->members = NEW(pyod_detector *, T);
    m->features = NEW(size_t *, T);
    m->n_feat = NEW(size_t, T);
    if (!m->members || !m->features || !m->n_feat) return PYOD_ERR_NOMEM;
    pyod_factory f = m->p.base_estimator ? m->p.base_estimator : default_lof;
    pyod_rng r;
    rng_seed(&r, m->p.random_state);
    for (size_t e = 0; e < T; e++) {
        uint64_t seed = rng_u64(&r);
        pyod_rng fr;
        rng_seed(&fr, seed);
        m->features[e] = NEW(size_t, d);
        if (!m->features[e]) return PYOD_ERR_NOMEM;
        m->n_feat[e] = bagging_indices(&fr, m->p.bootstrap_features, d, min, max + 1, m->features[e]);
        m->members[e] = f(seed, m->p.factory_ctx);
        m->n_members++;
        if (!m->members[e]) return PYOD_ERR_NOMEM;
        pyod_matrix sub;
        CHECK(mat_cols(x, m->features[e], m->n_feat[e], &sub));
        pyod_status st = pyod_fit(m->members[e], &sub);
        pyod_matrix_free(&sub);
        CHECK(st);
    }
    pyod_matrix cols;
    CHECK(member_scores(m, x, &cols, 1));
    double *s = NEW(double, x->rows);
    if (!s) {
        pyod_matrix_free(&cols);
        return PYOD_ERR_NOMEM;
    }
    combine(m->p.combination, &cols, s);
    pyod_matrix_free(&cols);
    return fit_state_contamination(dt, s, x->rows, m->p.contamination);
}

static pyod_status fb_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    const fb_impl *m = d->impl;
    pyod_matrix cols;
    CHECK(member_scores(m, x, &cols, 0));
    combine(m->p.combination, &cols, out);
    pyod_matrix_free(&cols);
    return PYOD_OK;
}

static void fb_destroy(void *p) {
    fb_clear(p);
    free(p);
}

static const pyod_vtable FB_VT = {"FeatureBagging", fb_fit, NULL, fb_score, fb_destroy};

pyod_feature_bagging_params pyod_feature_bagging_defaults(void) {
    return (pyod_feature_bagging_params){NULL, NULL, 10, 1.0, 0, PYOD_COMBINE_AVERAGE, 0.1, 0};
}

pyod_detector *pyod_feature_bagging_create(const pyod_feature_bagging_params *p) {
    fb_impl *m = NEW(fb_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_feature_bagging_defaults();
    return detector_new(&FB_VT, m);
}
