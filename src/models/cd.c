/* CD: Cook's distance outlier detection (Cook, 1977). Port of pyod.models.cd:
 * each feature is regressed on the others (OLS with intercept); the score is
 * the min-max-normalised Cook's distance averaged over features. As in PyOD,
 * decision_function keeps the fitted regressions but measures leverage and
 * residual variance on the data it is given. */
#include "internal.h"

typedef struct {
    pyod_cd_params p;
    size_t d;
    double *coef;      /* d x (d - 1) */
    double *intercept; /* d */
} cd_impl;

/* x without column `skip`, as a new (n x d-1) array. */
static double *drop_col(const pyod_matrix *x, size_t skip) {
    size_t n = x->rows, d = x->cols;
    double *out = NEW(double, n * (d - 1));
    if (!out) return NULL;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0, k = 0; j < d; j++)
            if (j != skip) out[i * (d - 1) + k++] = PYOD_AT(x, i, j);
    return out;
}

/* Least squares with intercept (minimum-norm, like sklearn LinearRegression). */
static void ols(const double *xr, const double *y, size_t n, size_t p, double *coef, double *intercept) {
    double *xm = NEW(double, p), *xtx = NEW(double, p * p), *xty = NEW(double, p), *inv = NEW(double, p * p);
    if (!xm || !xtx || !xty || !inv) goto done;
    double ym = vec_mean(y, n);
    for (size_t j = 0; j < p; j++) {
        for (size_t i = 0; i < n; i++) xm[j] += xr[i * p + j];
        xm[j] /= (double)n;
    }
    for (size_t i = 0; i < n; i++)
        for (size_t a = 0; a < p; a++) {
            double ca = xr[i * p + a] - xm[a];
            xty[a] += ca * (y[i] - ym);
            for (size_t b = 0; b < p; b++) xtx[a * p + b] += ca * (xr[i * p + b] - xm[b]);
        }
    pinvh(xtx, p, inv);
    *intercept = ym;
    for (size_t a = 0; a < p; a++) {
        coef[a] = dot(inv + a * p, xty, p);
        *intercept -= coef[a] * xm[a];
    }
done:
    free(xm), free(xtx), free(xty), free(inv);
}

/* Port of _Cooks_dist; returns 0 when the residual variance is zero or undefined. */
static int cooks(const double *xr, const double *y, size_t n, size_t p, const double *coef, double intercept, double *out) {
    double *gram = NEW(double, p * p), *ginv = NEW(double, p * p), *vals = NEW(double, p);
    double *vecs = NEW(double, p * p), *resid = NEW(double, n), *lev = NEW(double, n);
    int ok = 0;
    if (!gram || !ginv || !vals || !vecs || !resid || !lev) goto done;
    for (size_t i = 0; i < n; i++)
        for (size_t a = 0; a < p; a++)
            for (size_t b = 0; b < p; b++) gram[a * p + b] += xr[i * p + a] * xr[i * p + b];
    pinvh(gram, p, ginv);
    sym_eigen(gram, p, vals, vecs);
    double smax = 0.0;
    for (size_t a = 0; a < p; a++) smax = fmax(smax, sqrt(fmax(vals[a], 0.0)));
    size_t rank = 0;
    for (size_t a = 0; a < p; a++) rank += sqrt(fmax(vals[a], 0.0)) > smax * (double)(n > p ? n : p) * PYOD_DBL_EPS;
    double sse = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double *r = xr + i * p;
        double h = 0.0;
        for (size_t a = 0; a < p; a++) h += dot(ginv + a * p, r, p) * r[a];
        lev[i] = h;
        resid[i] = y[i] - (intercept + dot(r, coef, p));
        sse += resid[i] * resid[i];
    }
    double mse = sse / ((double)n - (double)rank);
    if (mse == 0.0 || !isfinite(mse)) goto done;
    for (size_t i = 0; i < n; i++) {
        double stud = resid[i] / sqrt(mse) / sqrt(1.0 - lev[i]);
        out[i] = stud * stud / (double)p * (lev[i] / (1.0 - lev[i]));
    }
    double lo = vec_min(out, n), hi = vec_max(out, n);
    for (size_t i = 0; i < n; i++) out[i] = (out[i] - lo) / (hi - lo);
    ok = 1;
done:
    free(gram), free(ginv), free(vals), free(vecs), free(resid), free(lev);
    return ok;
}

static pyod_status cd_scores(const cd_impl *m, const pyod_matrix *x, double *out) {
    size_t n = x->rows, d = m->d;
    double *acc = NEW(double, n), *dist = NEW(double, n), *y = NEW(double, n);
    size_t *cnt = NEW(size_t, n);
    if (!acc || !dist || !y || !cnt) {
        free(acc), free(dist), free(y), free(cnt);
        return PYOD_ERR_NOMEM;
    }
    for (size_t f = 0; f < d; f++) {
        double *xr = drop_col(x, f);
        if (!xr) continue;
        for (size_t i = 0; i < n; i++) y[i] = PYOD_AT(x, i, f);
        if (cooks(xr, y, n, d - 1, m->coef + f * (d - 1), m->intercept[f], dist))
            for (size_t i = 0; i < n; i++)
                if (!isnan(dist[i])) acc[i] += dist[i], cnt[i]++;
        free(xr);
    }
    for (size_t i = 0; i < n; i++) out[i] = cnt[i] ? acc[i] / (double)cnt[i] : NAN;
    free(acc), free(dist), free(y), free(cnt);
    return PYOD_OK;
}

static pyod_status cd_fit(pyod_detector *dt, const pyod_matrix *x) {
    cd_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t n = x->rows, d = x->cols;
    if (d < 2) return PYOD_ERR_PARAM;
    free(m->coef), free(m->intercept);
    m->d = d;
    m->coef = NEW(double, d * (d - 1));
    m->intercept = NEW(double, d);
    double *y = NEW(double, n), *s = NEW(double, n);
    if (!m->coef || !m->intercept || !y || !s) {
        free(y), free(s);
        return PYOD_ERR_NOMEM;
    }
    for (size_t f = 0; f < d; f++) {
        double *xr = drop_col(x, f);
        if (!xr) {
            free(y), free(s);
            return PYOD_ERR_NOMEM;
        }
        for (size_t i = 0; i < n; i++) y[i] = PYOD_AT(x, i, f);
        ols(xr, y, n, d - 1, m->coef + f * (d - 1), &m->intercept[f]);
        free(xr);
    }
    free(y);
    pyod_status st = cd_scores(m, x, s);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(dt, s, n, m->p.contamination);
}

static pyod_status cd_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return cd_scores(d->impl, x, out); }

static void cd_destroy(void *p) {
    cd_impl *m = p;
    free(m->coef), free(m->intercept), free(m);
}

static const pyod_vtable CD_VT = {"CD", cd_fit, NULL, cd_score, cd_destroy};

pyod_cd_params pyod_cd_defaults(void) { return (pyod_cd_params){0.1}; }

pyod_detector *pyod_cd_create(const pyod_cd_params *p) {
    cd_impl *m = NEW(cd_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_cd_defaults();
    return detector_new(&CD_VT, m);
}
