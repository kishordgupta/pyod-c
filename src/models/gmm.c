/* GMM outlier detection (Aggarwal, 2015). Port of pyod.models.gmm (sklearn
 * GaussianMixture, full covariances, k-means init); score = -log likelihood. */
#include "internal.h"

typedef struct {
    pyod_gmm_params p;
    size_t k, d;
    double *weights, *means, *chol; /* k, k x d, k x d x d (lower Cholesky factors) */
} gmm_impl;

static void wlp(const gmm_impl *m, const double *r, double *out, double *tmp, double *z) {
    size_t d = m->d;
    for (size_t c = 0; c < m->k; c++) {
        const double *l = m->chol + c * d * d;
        for (size_t j = 0; j < d; j++) tmp[j] = r[j] - m->means[c * d + j];
        forward_solve(l, tmp, z, d);
        double maha = dot(z, z, d), logdet = 0.0;
        for (size_t j = 0; j < d; j++) logdet += log(l[j * d + j]);
        out[c] = log(m->weights[c]) - 0.5 * ((double)d * log(2.0 * PYOD_PI) + maha) - logdet;
    }
}

static pyod_status m_step(gmm_impl *m, const pyod_matrix *x, const double *resp) {
    size_t n = x->rows, d = m->d, k = m->k;
    double *nk = NEW(double, k), *cov = NEW(double, d * d);
    if (!nk || !cov) {
        free(nk), free(cov);
        return PYOD_ERR_NOMEM;
    }
    for (size_t c = 0; c < k; c++) {
        for (size_t i = 0; i < n; i++) nk[c] += resp[i * k + c];
        nk[c] += 10.0 * PYOD_DBL_EPS;
        m->weights[c] = nk[c] / (double)n;
        for (size_t j = 0; j < d; j++) {
            double s = 0.0;
            for (size_t i = 0; i < n; i++) s += resp[i * k + c] * PYOD_AT(x, i, j);
            m->means[c * d + j] = s / nk[c];
        }
        memset(cov, 0, d * d * sizeof(double));
        for (size_t i = 0; i < n; i++) {
            double w = resp[i * k + c];
            for (size_t a = 0; a < d; a++) {
                double da = PYOD_AT(x, i, a) - m->means[c * d + a];
                for (size_t b = a; b < d; b++) cov[a * d + b] += w * da * (PYOD_AT(x, i, b) - m->means[c * d + b]);
            }
        }
        for (size_t a = 0; a < d; a++) {
            for (size_t b = a; b < d; b++) {
                cov[a * d + b] /= nk[c];
                cov[b * d + a] = cov[a * d + b];
            }
            cov[a * d + a] += m->p.reg_covar;
        }
        if (!cholesky(cov, d, m->chol + c * d * d)) {
            free(nk), free(cov);
            return PYOD_ERR_NUMERIC;
        }
    }
    free(nk), free(cov);
    return PYOD_OK;
}

static void gmm_scores(const gmm_impl *m, const pyod_matrix *x, double *out) {
    double *w = NEW(double, m->k), *tmp = NEW(double, m->d), *z = NEW(double, m->d);
    if (w && tmp && z)
        for (size_t i = 0; i < x->rows; i++) {
            wlp(m, &PYOD_AT(x, i, 0), w, tmp, z);
            out[i] = -logsumexp(w, m->k);
        }
    free(w), free(tmp), free(z);
}

static pyod_status gmm_fit(pyod_detector *dt, const pyod_matrix *x) {
    gmm_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t n = x->rows, d = x->cols, k = m->p.n_components;
    if (k == 0 || k > n) return PYOD_ERR_PARAM;
    m->k = k;
    m->d = d;
    free(m->weights), free(m->means), free(m->chol);
    m->weights = NEW(double, k);
    m->means = NEW(double, k * d);
    m->chol = NEW(double, k * d * d);
    double *resp = NEW(double, n * k), *centers = NEW(double, k * d), *w = NEW(double, k);
    double *tmp = NEW(double, d), *z = NEW(double, d);
    double *bw = NEW(double, k), *bm = NEW(double, k * d), *bc = NEW(double, k * d * d);
    size_t *labels = NEW(size_t, n);
    pyod_status st = PYOD_ERR_NOMEM;
    if (!m->weights || !m->means || !m->chol || !resp || !centers || !w || !tmp || !z || !bw || !bm || !bc || !labels)
        goto done;
    pyod_rng r;
    rng_seed(&r, m->p.random_state);
    double best = -INFINITY;
    for (size_t init = 0; init < (m->p.n_init ? m->p.n_init : 1); init++) {
        kmeans(x, k, 1, 300, &r, centers, labels);
        memset(resp, 0, n * k * sizeof(double));
        for (size_t i = 0; i < n; i++) resp[i * k + labels[i]] = 1.0;
        if ((st = m_step(m, x, resp)) != PYOD_OK) goto done;
        double lower = -INFINITY;
        for (size_t it = 0; it < m->p.max_iter; it++) {
            double prev = lower, total = 0.0;
            for (size_t i = 0; i < n; i++) {
                wlp(m, &PYOD_AT(x, i, 0), w, tmp, z);
                double norm = logsumexp(w, k);
                total += norm;
                for (size_t c = 0; c < k; c++) resp[i * k + c] = exp(w[c] - norm);
            }
            lower = total / (double)n;
            if ((st = m_step(m, x, resp)) != PYOD_OK) goto done;
            if (fabs(lower - prev) < m->p.tol) break;
        }
        if (lower > best) {
            best = lower;
            memcpy(bw, m->weights, k * sizeof(double));
            memcpy(bm, m->means, k * d * sizeof(double));
            memcpy(bc, m->chol, k * d * d * sizeof(double));
        }
    }
    memcpy(m->weights, bw, k * sizeof(double));
    memcpy(m->means, bm, k * d * sizeof(double));
    memcpy(m->chol, bc, k * d * d * sizeof(double));
    double *s = NEW(double, n);
    if (!s) {
        st = PYOD_ERR_NOMEM;
        goto done;
    }
    gmm_scores(m, x, s);
    st = fit_state_contamination(dt, s, n, m->p.contamination);
done:
    free(resp), free(centers), free(w), free(tmp), free(z), free(bw), free(bm), free(bc), free(labels);
    return st;
}

static pyod_status gmm_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    gmm_scores(d->impl, x, out);
    return PYOD_OK;
}

static void gmm_destroy(void *p) {
    gmm_impl *m = p;
    free(m->weights), free(m->means), free(m->chol), free(m);
}

static const pyod_vtable GMM_VT = {"GMM", gmm_fit, NULL, gmm_score, gmm_destroy};

pyod_gmm_params pyod_gmm_defaults(void) { return (pyod_gmm_params){1, 100, 1, 1e-3, 1e-6, 0.1, 0}; }

pyod_detector *pyod_gmm_create(const pyod_gmm_params *p) {
    gmm_impl *m = NEW(gmm_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_gmm_defaults();
    return detector_new(&GMM_VT, m);
}
