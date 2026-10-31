/* LSCP: locally selective combination of parallel outlier ensembles (Zhao et
 * al., SDM 2019). Port of pyod.models.lscp. For each sample, LSCP finds its
 * local region in the training data, measures each detector's Pearson
 * correlation with a pseudo ground truth there, and averages the scores of
 * the most competent detectors. */
#include "internal.h"

#define LR_ITERATIONS 20
#define LR_THRESHOLD (LR_ITERATIONS / 2)

typedef struct {
    pyod_lscp_params p;
    pyod_detector **dets;
    size_t n_dets;
    pyod_matrix train;
    pyod_matrix train_scores; /* n x n_dets */
} lscp_impl;

static double pearson(const double *a, const double *b, size_t n) {
    double ma = vec_mean(a, n), mb = vec_mean(b, n), cov = 0.0, va = 0.0, vb = 0.0;
    for (size_t i = 0; i < n; i++) {
        cov += (a[i] - ma) * (b[i] - mb);
        va += (a[i] - ma) * (a[i] - ma);
        vb += (b[i] - mb) * (b[i] - mb);
    }
    return cov / sqrt(va * vb);
}

/* Detectors in the most populated histogram bin of correlations. */
static size_t competent(const lscp_impl *m, double *corr, size_t k, size_t *out) {
    for (size_t i = 0; i < k; i++)
        if (isnan(corr[i])) corr[i] = 0.0;
    size_t nb = m->p.n_bins < k ? m->p.n_bins : k;
    if (nb < 1) nb = 1;
    double lo = vec_min(corr, k), hi = vec_max(corr, k);
    if (lo == hi) lo -= 0.5, hi += 0.5;
    size_t hist[64] = {0}, best = 0;
    if (nb > 64) nb = 64;
    for (size_t i = 0; i < k; i++) {
        size_t b = (size_t)((corr[i] - lo) / (hi - lo) * (double)nb);
        hist[b < nb ? b : nb - 1]++;
    }
    for (size_t b = 1; b < nb; b++)
        if (hist[b] > hist[best]) best = b;
    double e0 = lo + (hi - lo) * (double)best / (double)nb, e1 = lo + (hi - lo) * (double)(best + 1) / (double)nb;
    size_t c = 0;
    for (size_t i = 0; i < k; i++)
        if (corr[i] >= e0 && corr[i] <= e1) out[c++] = i;
    return c;
}

/* Scores for x. Local regions: training points among the nearest neighbours
 * in more than half of 20 random feature subspaces. PyOD redraws subspaces on
 * every call; we reseed from random_state so scores are reproducible. */
static pyod_status lscp_scores(const lscp_impl *m, const pyod_matrix *x, double *out) {
    size_t n = x->rows, nt = m->train.rows, d = m->train.cols, k = m->n_dets, lr = m->p.local_region_size;
    if (lr > nt) lr = nt;
    double maxf = fmin(m->p.local_max_features, 1.0), minf = (double)d * 0.5 < 1.0 ? 1.0 : 0.5;
    uint8_t *counts = NEW(uint8_t, n * nt);
    size_t *feat = NEW(size_t, d), *nbr = NEW(size_t, lr), *chosen = NEW(size_t, k), *region = NEW(size_t, nt);
    double *test = NEW(double, n * k), *buf = NEW(double, n), *pseudo = NEW(double, nt);
    double *truth = NEW(double, nt), *col = NEW(double, nt), *corr = NEW(double, k), *q = NEW(double, d);
    std_scaler sc = {0};
    pyod_matrix proj = {0}, tn = {0};
    pyod_status st = PYOD_ERR_NOMEM;
    if (!counts || !feat || !nbr || !chosen || !region || !test || !buf || !pseudo || !truth || !col || !corr || !q) goto done;
    pyod_rng r;
    rng_seed(&r, m->p.random_state);
    for (int it = 0; it < LR_ITERATIONS; it++) {
        size_t nf = d;
        if (maxf == minf) {
            for (size_t j = 0; j < d; j++) feat[j] = j;
        } else {
            nf = bagging_indices(&r, 0, d, (size_t)((double)d * minf), (size_t)((double)d * maxf), feat);
        }
        pyod_matrix_free(&proj);
        if ((st = mat_cols(&m->train, feat, nf, &proj)) != PYOD_OK) goto done;
        st = PYOD_ERR_NOMEM;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < nf; j++) q[j] = PYOD_AT(x, i, feat[j]);
            size_t got = k_nearest(&proj, q, lr, SIZE_MAX, nbr, NULL);
            for (size_t t = 0; t < got; t++) counts[i * nt + nbr[t]]++;
        }
    }
    for (size_t e = 0; e < k; e++) {
        if ((st = pyod_decision_function(m->dets[e], x, buf)) != PYOD_OK) goto done;
        for (size_t i = 0; i < n; i++) test[i * k + e] = buf[i];
    }
    st = PYOD_ERR_NOMEM;
    if (scaler_fit(&sc, &m->train_scores) != PYOD_OK || pyod_matrix_alloc(&tn, nt, k) != PYOD_OK) goto done;
    scaler_transform(&sc, &m->train_scores, tn.data);
    pyod_matrix tm = pyod_matrix_view(test, n, k);
    scaler_transform(&sc, &tm, test);
    for (size_t t = 0; t < nt; t++) pseudo[t] = vec_max(&PYOD_AT(&tn, t, 0), k);
    for (size_t i = 0; i < n; i++) {
        size_t rn = 0;
        for (int th = LR_THRESHOLD; th >= 0 && rn < 2; th--) {
            rn = 0;
            for (size_t t = 0; t < nt; t++)
                if (counts[i * nt + t] > th) region[rn++] = t;
        }
        for (size_t t = 0; t < rn; t++) truth[t] = pseudo[region[t]];
        for (size_t e = 0; e < k; e++) {
            for (size_t t = 0; t < rn; t++) col[t] = PYOD_AT(&tn, region[t], e);
            corr[e] = pearson(truth, col, rn);
        }
        size_t nc = competent(m, corr, k, chosen);
        double acc = 0.0;
        for (size_t c = 0; c < nc; c++) acc += test[i * k + chosen[c]];
        out[i] = acc / (double)nc;
    }
    st = PYOD_OK;
done:
    scaler_free(&sc);
    pyod_matrix_free(&proj);
    pyod_matrix_free(&tn);
    free(counts), free(feat), free(nbr), free(chosen), free(region), free(test), free(buf), free(pseudo);
    free(truth), free(col), free(corr), free(q);
    return st;
}

static pyod_status lscp_fit(pyod_detector *dt, const pyod_matrix *x) {
    lscp_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    if (m->n_dets < 2) return PYOD_ERR_PARAM;
    pyod_matrix_free(&m->train);
    pyod_matrix_free(&m->train_scores);
    CHECK(mat_copy(x, &m->train));
    CHECK(pyod_matrix_alloc(&m->train_scores, x->rows, m->n_dets));
    for (size_t e = 0; e < m->n_dets; e++) {
        CHECK(pyod_fit(m->dets[e], x));
        const double *s = pyod_decision_scores(m->dets[e], NULL);
        for (size_t i = 0; i < x->rows; i++) PYOD_AT(&m->train_scores, i, e) = s[i];
    }
    double *s = NEW(double, x->rows);
    if (!s) return PYOD_ERR_NOMEM;
    pyod_status st = lscp_scores(m, x, s);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(dt, s, x->rows, m->p.contamination);
}

static pyod_status lscp_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return lscp_scores(d->impl, x, out); }

static void lscp_destroy(void *p) {
    lscp_impl *m = p;
    for (size_t e = 0; e < m->n_dets; e++) pyod_free(m->dets[e]);
    free(m->dets);
    pyod_matrix_free(&m->train);
    pyod_matrix_free(&m->train_scores);
    free(m);
}

static const pyod_vtable LSCP_VT = {"LSCP", lscp_fit, NULL, lscp_score, lscp_destroy};

pyod_lscp_params pyod_lscp_defaults(void) { return (pyod_lscp_params){30, 10, 1.0, 0.1, 0}; }

pyod_detector *pyod_lscp_create(pyod_detector **detectors, size_t n, const pyod_lscp_params *p) {
    lscp_impl *m = NEW(lscp_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_lscp_defaults();
    m->dets = NEW(pyod_detector *, n);
    if (!m->dets) {
        free(m);
        return NULL;
    }
    memcpy(m->dets, detectors, n * sizeof(pyod_detector *));
    m->n_dets = n;
    return detector_new(&LSCP_VT, m);
}
