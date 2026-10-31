/* Minimum Covariance Determinant (Rousseeuw & Van Driessen, 1999). Port of
 * pyod.models.mcd (sklearn MinCovDet / FastMCD); the score is the squared
 * robust Mahalanobis distance. */
#include "internal.h"

typedef struct {
    double *loc, *cov, *dist; /* d, d x d, n */
    size_t *support;          /* h */
    double det;
} estimate;

typedef struct {
    pyod_mcd_params p;
    double *location, *precision;
    size_t d;
} mcd_impl;

static void est_free(estimate *e) {
    free(e->loc), free(e->cov), free(e->dist), free(e->support);
    memset(e, 0, sizeof(*e));
}

static int est_alloc(estimate *e, size_t n, size_t d, size_t h) {
    e->loc = NEW(double, d);
    e->cov = NEW(double, d * d);
    e->dist = NEW(double, n);
    e->support = NEW(size_t, h);
    return e->loc && e->cov && e->dist && e->support;
}

/* Indices of the h smallest values. */
static void smallest(const double *v, size_t n, size_t h, size_t *out, size_t *tmp) {
    argsort(v, n, tmp);
    memcpy(out, tmp, h * sizeof(size_t));
}

static void support_stats(const pyod_matrix *x, const size_t *sup, size_t h, double *loc, double *cov) {
    pyod_matrix sub;
    if (mat_rows(x, sup, h, &sub) != PYOD_OK) return;
    col_mean(&sub, loc);
    covariance(&sub, loc, 0, cov);
    pyod_matrix_free(&sub);
}

/* sklearn _c_step: concentrate the support until the determinant stops shrinking. */
static void c_step(const pyod_matrix *x, size_t h, size_t iters, const double *init_loc, const double *init_cov,
                   pyod_rng *r, estimate *out) {
    size_t n = x->rows, d = x->cols;
    estimate cur = {0}, prev = {0};
    double *prec = NEW(double, d * d);
    size_t *tmp = NEW(size_t, n);
    if (!est_alloc(&cur, n, d, h) || !est_alloc(&prev, n, d, h) || !prec || !tmp) goto fail;
    if (!init_loc) {
        rng_sample(r, n, h, cur.support);
    } else {
        pinvh(init_cov, d, prec);
        mahalanobis_sq(x, init_loc, prec, cur.dist);
        smallest(cur.dist, n, h, cur.support, tmp);
    }
    support_stats(x, cur.support, h, cur.loc, cur.cov);
    cur.det = fast_logdet(cur.cov, d);
    pinvh(cur.cov, d, prec);
    for (size_t i = 0; i < n; i++) cur.dist[i] = INFINITY;
    int have_prev = 0;
    double prev_det = INFINITY;
    size_t remaining = iters;
    while (cur.det < prev_det && remaining > 0 && isfinite(cur.det)) {
        memcpy(prev.loc, cur.loc, d * sizeof(double));
        memcpy(prev.cov, cur.cov, d * d * sizeof(double));
        memcpy(prev.dist, cur.dist, n * sizeof(double));
        memcpy(prev.support, cur.support, h * sizeof(size_t));
        prev.det = cur.det;
        have_prev = 1;
        prev_det = cur.det;
        pinvh(cur.cov, d, prec);
        mahalanobis_sq(x, cur.loc, prec, cur.dist);
        smallest(cur.dist, n, h, cur.support, tmp);
        support_stats(x, cur.support, h, cur.loc, cur.cov);
        cur.det = fast_logdet(cur.cov, d);
        remaining--;
    }
    mahalanobis_sq(x, cur.loc, prec, cur.dist);
    int close = fabs(cur.det - prev_det) <= 1e-8 + 1e-5 * fabs(prev_det);
    if (!(isinf(cur.det) || close || remaining == 0) && cur.det > prev_det && have_prev) {
        *out = prev;
        est_free(&cur);
    } else {
        *out = cur;
        est_free(&prev);
    }
    free(prec), free(tmp);
    return;
fail:
    est_free(&cur), est_free(&prev);
    free(prec), free(tmp);
    memset(out, 0, sizeof(*out));
    out->det = INFINITY;
}

/* sklearn select_candidates: C-steps from many starts, keep the `keep` best. */
static size_t select_best(const pyod_matrix *x, size_t h, size_t n_random, const estimate *starts, size_t n_starts,
                          size_t keep, size_t iters, pyod_rng *r, estimate *out) {
    size_t total = starts ? n_starts : n_random;
    estimate *all = NEW(estimate, total);
    double *dets = NEW(double, total);
    size_t *order = NEW(size_t, total);
    if (!all || !dets || !order) {
        free(all), free(dets), free(order);
        return 0;
    }
    for (size_t t = 0; t < total; t++) {
        c_step(x, h, iters, starts ? starts[t].loc : NULL, starts ? starts[t].cov : NULL, r, &all[t]);
        dets[t] = all[t].det;
    }
    argsort(dets, total, order);
    if (keep > total) keep = total;
    for (size_t t = 0; t < total; t++) {
        if (t < keep) out[t] = all[order[t]];
        else est_free(&all[order[t]]);
    }
    free(all), free(dets), free(order);
    return keep;
}

/* sklearn fast_mcd; returns raw location / covariance and raw distances. */
static pyod_status fast_mcd(const pyod_matrix *x, size_t h, pyod_rng *r, double *loc, double *cov, double *dist) {
    size_t n = x->rows, d = x->cols;
    if (d == 1) {
        double *s = NEW(double, n), *dev = NEW(double, n), *sup = NEW(double, n);
        size_t *idx = NEW(size_t, n);
        if (!s || !dev || !sup || !idx) {
            free(s), free(dev), free(sup), free(idx);
            return PYOD_ERR_NOMEM;
        }
        memcpy(s, x->data, n * sizeof(double));
        qsort(s, n, sizeof(double), cmp_double);
        double location;
        size_t m = 0;
        if (h < n) {
            double mn = INFINITY;
            for (size_t i = 0; i < n - h; i++) mn = fmin(mn, s[i + h] - s[i]);
            double acc = 0.0;
            size_t cnt = 0;
            for (size_t i = 0; i < n - h; i++)
                if (s[i + h] - s[i] == mn) acc += s[i + h] + s[i], cnt++;
            location = 0.5 * acc / (double)cnt;
            for (size_t i = 0; i < n; i++) dev[i] = fabs(x->data[i] - location);
            argsort(dev, n, idx);
            for (size_t i = 0; i < h; i++) sup[m++] = x->data[idx[i]];
        } else {
            location = vec_mean(x->data, n);
            memcpy(sup, x->data, n * sizeof(double));
            m = n;
        }
        double mm = vec_mean(sup, m), var = 0.0;
        for (size_t i = 0; i < m; i++) var += (sup[i] - mm) * (sup[i] - mm);
        var /= (double)m;
        loc[0] = location;
        cov[0] = var;
        double prec;
        pinvh(cov, 1, &prec);
        mahalanobis_sq(x, loc, &prec, dist);
        free(s), free(dev), free(sup), free(idx);
        return PYOD_OK;
    }
    estimate best = {0};
    if (n > 500) {
        size_t n_subsets = n / 300, per = n / n_subsets;
        size_t h_sub = (size_t)ceil((double)per * ((double)h / (double)n));
        size_t n_trials = 500 / n_subsets > 10 ? 500 / n_subsets : 10;
        size_t *shuffle = NEW(size_t, n);
        estimate *pooled = NEW(estimate, n_subsets * 10);
        if (!shuffle || !pooled) {
            free(shuffle), free(pooled);
            return PYOD_ERR_NOMEM;
        }
        rng_sample(r, n, n, shuffle);
        size_t np = 0;
        for (size_t i = 0; i < n_subsets; i++) {
            pyod_matrix sub;
            CHECK(mat_rows(x, shuffle + i * per, per, &sub));
            np += select_best(&sub, h_sub, n_trials, NULL, 0, 10, 2, r, pooled + np);
            pyod_matrix_free(&sub);
        }
        size_t n_merged = n < 1500 ? n : 1500;
        size_t h_merged = (size_t)ceil((double)n_merged * ((double)h / (double)n));
        size_t *selection = NEW(size_t, n_merged);
        pyod_matrix merged_x;
        rng_sample(r, n, n_merged, selection);
        CHECK(mat_rows(x, selection, n_merged, &merged_x));
        size_t keep = n > 1500 ? 10 : 1;
        estimate merged[10];
        size_t got = select_best(&merged_x, h_merged, 0, pooled, np, keep, 30, r, merged);
        for (size_t i = 0; i < np; i++) est_free(&pooled[i]);
        free(pooled);
        if (n < 1500) {
            memcpy(loc, merged[0].loc, d * sizeof(double));
            memcpy(cov, merged[0].cov, d * d * sizeof(double));
            memset(dist, 0, n * sizeof(double));
            for (size_t k = 0; k < n_merged; k++) dist[selection[k]] = merged[0].dist[k];
            for (size_t i = 0; i < got; i++) est_free(&merged[i]);
        } else {
            select_best(x, h, 0, merged, got, 1, 30, r, &best);
            for (size_t i = 0; i < got; i++) est_free(&merged[i]);
        }
        free(shuffle), free(selection);
        pyod_matrix_free(&merged_x);
        if (n < 1500) return PYOD_OK;
    } else {
        estimate cand[10];
        size_t got = select_best(x, h, 30, NULL, 0, 10, 2, r, cand);
        select_best(x, h, 0, cand, got, 1, 30, r, &best);
        for (size_t i = 0; i < got; i++) est_free(&cand[i]);
    }
    if (!best.loc) return PYOD_ERR_NOMEM;
    memcpy(loc, best.loc, d * sizeof(double));
    memcpy(cov, best.cov, d * d * sizeof(double));
    memcpy(dist, best.dist, n * sizeof(double));
    est_free(&best);
    return PYOD_OK;
}

static pyod_status mcd_fit(pyod_detector *dt, const pyod_matrix *x) {
    mcd_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t n = x->rows, d = x->cols;
    if (n < 2) return PYOD_ERR_PARAM;
    size_t h;
    if (m->p.support_fraction <= 0.0) h = (size_t)ceil(0.5 * (double)(n + d + 1));
    else if (m->p.support_fraction <= 1.0) h = (size_t)(m->p.support_fraction * (double)n);
    else return PYOD_ERR_PARAM;
    if (h < 1) h = 1;
    if (h > n) h = n;
    m->d = d;
    free(m->location), free(m->precision);
    m->location = NEW(double, d);
    m->precision = NEW(double, d * d);
    double *loc = NEW(double, d), *cov = NEW(double, d * d), *dist = NEW(double, n), *s = NEW(double, n);
    pyod_status st = PYOD_ERR_NOMEM;
    if (!m->location || !m->precision || !loc || !cov || !dist || !s) goto done;
    pyod_rng r;
    rng_seed(&r, m->p.random_state);
    if ((st = fast_mcd(x, h, &r, loc, cov, dist)) != PYOD_OK) goto done;
    /* Consistency correction (sklearn correct_covariance). */
    double correction = vec_median(dist, n) / chi2_ppf(0.5, (double)d);
    for (size_t i = 0; i < n; i++) dist[i] /= correction;
    /* Reweighting (sklearn reweight_covariance). */
    double cut = chi2_ppf(0.975, (double)d);
    size_t *keep = NEW(size_t, n), nk = 0;
    if (!keep) {
        st = PYOD_ERR_NOMEM;
        goto done;
    }
    for (size_t i = 0; i < n; i++)
        if (dist[i] < cut) keep[nk++] = i;
    pyod_matrix kept;
    if ((st = mat_rows(x, keep, nk, &kept)) != PYOD_OK) {
        free(keep);
        goto done;
    }
    col_mean(&kept, m->location);
    covariance(&kept, m->location, 0, cov);
    pinvh(cov, d, m->precision);
    pyod_matrix_free(&kept);
    free(keep);
    mahalanobis_sq(x, m->location, m->precision, s);
    st = fit_state_contamination(dt, s, n, m->p.contamination);
    s = NULL;
done:
    free(loc), free(cov), free(dist), free(s);
    return st;
}

static pyod_status mcd_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    const mcd_impl *m = d->impl;
    mahalanobis_sq(x, m->location, m->precision, out);
    return PYOD_OK;
}

static void mcd_destroy(void *p) {
    mcd_impl *m = p;
    free(m->location), free(m->precision), free(m);
}

static const pyod_vtable MCD_VT = {"MCD", mcd_fit, NULL, mcd_score, mcd_destroy};

pyod_mcd_params pyod_mcd_defaults(void) { return (pyod_mcd_params){0.0, 0.1, 0}; }

pyod_detector *pyod_mcd_create(const pyod_mcd_params *p) {
    mcd_impl *m = NEW(mcd_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_mcd_defaults();
    return detector_new(&MCD_VT, m);
}
