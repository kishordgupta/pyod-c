/* Matrix Profile discord detection (Yeh et al., ICDM 2016; STOMP). Port of
 * pyod.models.ts_matrix_profile. Transductive: only training scores exist. */
#include "internal.h"

typedef struct {
    pyod_matrix_profile_params p;
} mp_impl;

/* z-normalised distance from each subsequence to its nearest non-trivial match. */
static pyod_status matrix_profile(const double *t, size_t n, size_t m, double *mp) {
    size_t ns = n - m + 1, ez = m / 4;
    double *mu = NEW(double, ns), *sig = NEW(double, ns), *qt = NEW(double, ns);
    uint8_t *cst = NEW(uint8_t, ns);
    if (!mu || !sig || !qt || !cst) {
        free(mu), free(sig), free(qt), free(cst);
        return PYOD_ERR_NOMEM;
    }
    for (size_t i = 0; i < ns; i++) {
        double s = 0.0, s2 = 0.0;
        for (size_t k = 0; k < m; k++) s += t[i + k], s2 += t[i + k] * t[i + k];
        mu[i] = s / (double)m;
        sig[i] = sqrt(fmax(s2 / (double)m - mu[i] * mu[i], 0.0));
        cst[i] = sig[i] < 1e-10;
        mp[i] = INFINITY;
        qt[i] = 0.0;
        for (size_t k = 0; k < m; k++) qt[i] += t[k] * t[i + k];
    }
    for (size_t j = 0; j < ns; j++) {
        if (j > 0) {
            for (size_t i = ns - 1; i >= 1; i--) qt[i] = qt[i - 1] - t[i - 1] * t[j - 1] + t[i + m - 1] * t[j + m - 1];
            qt[0] = 0.0;
            for (size_t k = 0; k < m; k++) qt[0] += t[k] * t[j + k];
        }
        if (cst[j]) continue;
        size_t lo = j > ez ? j - ez : 0, hi = j + ez + 1 < ns ? j + ez + 1 : ns;
        for (size_t i = 0; i < ns; i++) {
            if (cst[i] || (i >= lo && i < hi)) continue;
            double denom = (double)m * sig[i] * sig[j];
            double corr = denom > 0.0 ? (qt[i] - (double)m * mu[i] * mu[j]) / denom : 0.0;
            double dd = sqrt(fmax(2.0 * (double)m * (1.0 - corr), 0.0));
            if (dd < mp[i]) mp[i] = dd;
        }
    }
    free(mu), free(sig), free(qt), free(cst);
    return PYOD_OK;
}

static pyod_status mp_fit(pyod_detector *d, const pyod_matrix *x) {
    mp_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t w = m->p.window_size, n = x->rows, c = x->cols;
    if (w < 2 || n < w + 1) return PYOD_ERR_PARAM;
    double *col = NEW(double, n), *prof = NEW(double, n), **pc = NEW(double *, c), *s = NEW(double, n);
    pyod_status st = col && prof && pc && s ? PYOD_OK : PYOD_ERR_NOMEM;
    for (size_t ch = 0; ch < c && st == PYOD_OK; ch++) {
        for (size_t i = 0; i < n; i++) col[i] = PYOD_AT(x, i, ch);
        if ((st = matrix_profile(col, n, w, prof)) != PYOD_OK) break;
        pc[ch] = NEW(double, n);
        if (!pc[ch]) {
            st = PYOD_ERR_NOMEM;
            break;
        }
        ts_map_scores(prof, n - w + 1, w, 1, n, m->p.channel_aggregation, pc[ch]);
    }
    if (st == PYOD_OK) ts_combine_channels(pc, c, n, m->p.channel_aggregation, s);
    for (size_t ch = 0; pc && ch < c; ch++) free(pc[ch]);
    free(pc), free(col), free(prof);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    return ts_fit_state(d, s, n, m->p.contamination);
}

static pyod_status mp_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    (void)d, (void)x, (void)out;
    return PYOD_ERR_UNSUPPORTED; /* transductive: use the fitted decision scores */
}

static const pyod_vtable MP_VT = {"MatrixProfile", mp_fit, NULL, mp_score, free};

pyod_matrix_profile_params pyod_matrix_profile_defaults(void) { return (pyod_matrix_profile_params){50, PYOD_AGG_MAX, 0.1}; }

pyod_detector *pyod_matrix_profile_create(const pyod_matrix_profile_params *p) {
    mp_impl *m = NEW(mp_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_matrix_profile_defaults();
    return detector_new(&MP_VT, m);
}
