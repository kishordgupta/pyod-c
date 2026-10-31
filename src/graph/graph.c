/* Graph outlier detection without neural networks: SCAN, ANOMALOUS and Radar.
 * Ports of pyod.models.pyg_scan, pyg_anomalous and pyg_radar. Graph detectors
 * are transductive, as in PyOD: they score the nodes of the fitted graph. */
#include "internal.h"

pyod_graph_detector *graph_detector_new(const char *name,
                                        pyod_status (*fit)(pyod_graph_detector *, const pyod_matrix *, const size_t (*)[2], size_t),
                                        void (*destroy)(void *), void *impl, double contamination) {
    if (!impl) return NULL;
    pyod_graph_detector *g = NEW(pyod_graph_detector, 1);
    if (!g) {
        destroy(impl);
        return NULL;
    }
    g->name = name;
    g->fit = fit;
    g->destroy = destroy;
    g->impl = impl;
    g->contamination = contamination;
    g->state.state.threshold = NAN;
    return g;
}

pyod_status pyod_graph_fit(pyod_graph_detector *g, const pyod_matrix *x, const size_t (*edges)[2], size_t n_edges) {
    if (!g || (n_edges && !edges)) return PYOD_ERR_PARAM;
    CHECK(check_matrix(x));
    CHECK(check_contamination(g->contamination));
    for (size_t e = 0; e < n_edges; e++)
        if (edges[e][0] >= x->rows || edges[e][1] >= x->rows) return PYOD_ERR_PARAM;
    g->state.fitted = 0;
    pyod_status st = g->fit(g, x, edges, n_edges);
    if (st != PYOD_OK) fit_state_clear(&g->state.state);
    return st;
}

const double *pyod_graph_scores(const pyod_graph_detector *g, size_t *n) { return g ? pyod_decision_scores(&g->state, n) : NULL; }
const uint8_t *pyod_graph_labels(const pyod_graph_detector *g, size_t *n) { return g ? pyod_labels(&g->state, n) : NULL; }
double pyod_graph_threshold(const pyod_graph_detector *g) { return g ? pyod_threshold(&g->state) : NAN; }
const char *pyod_graph_name(const pyod_graph_detector *g) { return g ? g->name : NULL; }

void pyod_graph_free(pyod_graph_detector *g) {
    if (!g) return;
    if (g->destroy) g->destroy(g->impl);
    fit_state_clear(&g->state.state);
    free(g);
}

/* Dense adjacency with duplicate edges summed (scipy csr semantics). */
static double *summed_adjacency(size_t n, const size_t (*e)[2], size_t ne) {
    double *a = NEW(double, n * n);
    if (a)
        for (size_t k = 0; k < ne; k++) a[e[k][0] * n + e[k][1]] += 1.0;
    return a;
}

/* ---------------- SCAN ---------------- */

typedef struct {
    pyod_scan_params p;
} scan_impl;

static pyod_status scan_fit(pyod_graph_detector *g, const pyod_matrix *x, const size_t (*edges)[2], size_t ne) {
    const scan_impl *m = g->impl;
    size_t n = x->rows;
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    if (ne == 0) {
        for (size_t i = 0; i < n; i++) s[i] = 1.0;
        return fit_state_contamination(&g->state, s, n, g->contamination);
    }
    double *adj = summed_adjacency(n, edges, ne), *aself = NEW(double, n * n), *sim = NEW(double, n * n);
    double *inv = NEW(double, n);
    size_t *neps = NEW(size_t, n), *comp = NEW(size_t, n), *stack = NEW(size_t, n), *seen = NEW(size_t, n);
    uint8_t *core = NEW(uint8_t, n);
    pyod_status st = PYOD_ERR_NOMEM;
    if (!adj || !aself || !sim || !inv || !neps || !comp || !stack || !seen || !core) goto done;
    for (size_t i = 0; i < n; i++) {
        double deg = 0.0;
        for (size_t j = 0; j < n; j++) deg += aself[i * n + j] = (adj[i * n + j] + (double)(i == j)) > 0.0;
        inv[i] = deg > 0.0 ? 1.0 / sqrt(deg) : 0.0;
    }
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            if (adj[i * n + j] == 0.0) continue;
            double inter = dot(aself + i * n, aself + j * n, n);
            sim[i * n + j] = inv[i] * inter * inv[j] * adj[i * n + j];
        }
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) neps[i] += sim[i * n + j] >= m->p.epsilon && sim[i * n + j] > 0.0;
        core[i] = neps[i] >= m->p.mu;
        comp[i] = SIZE_MAX;
    }
    /* Connected components of the core-to-core epsilon graph (undirected). */
    size_t next = 0;
    for (size_t s0 = 0; s0 < n; s0++) {
        if (!core[s0] || comp[s0] != SIZE_MAX) continue;
        size_t sp = 0;
        stack[sp++] = s0;
        comp[s0] = next;
        while (sp) {
            size_t u = stack[--sp];
            for (size_t v = 0; v < n; v++) {
                int linked = (sim[u * n + v] >= m->p.epsilon && sim[u * n + v] > 0.0) || (sim[v * n + u] >= m->p.epsilon && sim[v * n + u] > 0.0);
                if (core[v] && comp[v] == SIZE_MAX && linked) comp[v] = next, stack[sp++] = v;
            }
        }
        next++;
    }
    for (size_t u = 0; u < n; u++) {
        double deg = 0.0, ssum = 0.0;
        for (size_t v = 0; v < n; v++) deg += adj[u * n + v], ssum += sim[u * n + v];
        double mean_sim = deg > 0.0 ? ssum / deg : 0.0;
        if (core[u]) {
            size_t denom = m->p.mu * 2 > 1 ? m->p.mu * 2 : 1;
            s[u] = fmax(0.0, 1.0 - (double)neps[u] / (double)denom);
            continue;
        }
        size_t nc = 0; /* distinct clusters among neighbours */
        for (size_t v = 0; v < n; v++) {
            if (adj[u * n + v] == 0.0 || comp[v] == SIZE_MAX) continue;
            size_t k = 0;
            while (k < nc && seen[k] != comp[v]) k++;
            if (k == nc) seen[nc++] = comp[v];
        }
        s[u] = nc >= 2 ? 0.5 + 0.5 * (1.0 - mean_sim) : 1.0 + (1.0 - mean_sim);
    }
    st = fit_state_contamination(&g->state, s, n, g->contamination);
    s = NULL;
done:
    free(s), free(adj), free(aself), free(sim), free(inv), free(neps), free(comp), free(stack), free(seen), free(core);
    return st;
}

pyod_scan_params pyod_scan_defaults(void) { return (pyod_scan_params){0.5, 2, 0.1}; }

pyod_graph_detector *pyod_scan_create(const pyod_scan_params *p) {
    scan_impl *m = NEW(scan_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_scan_defaults();
    return graph_detector_new("SCAN", scan_fit, free, m, m->p.contamination);
}

/* ---------------- shared residual analysis ---------------- */

/* R = row-shrink(solve(I + gamma L, P)): Laplacian-smoothed residual with
 * row-wise group-lasso shrinkage. lap is n x n, p is n x d; result in r. */
static pyod_status smoothed_residual(const double *lap, const double *p, size_t n, size_t d, double gamma, double lambda, double *r) {
    double *sys = NEW(double, n * n);
    if (!sys) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) sys[i * n + j] = (double)(i == j) + gamma * lap[i * n + j];
    int ok = lin_solve(sys, p, r, n, d);
    free(sys);
    if (!ok) return PYOD_ERR_NUMERIC;
    for (size_t i = 0; i < n; i++) {
        double norm = fmax(sqrt(dot(r + i * d, r + i * d, d)), 1e-10);
        double shrink = fmax(1.0 - lambda / (2.0 * norm), 0.0);
        for (size_t j = 0; j < d; j++) r[i * d + j] *= shrink;
    }
    return PYOD_OK;
}

static double *laplacian(size_t n, const size_t (*edges)[2], size_t ne) {
    double *a = summed_adjacency(n, edges, ne);
    if (!a) return NULL;
    for (size_t i = 0; i < n; i++) {
        double deg = 0.0;
        for (size_t j = 0; j < n; j++) deg += a[i * n + j];
        for (size_t j = 0; j < n; j++) a[i * n + j] = (i == j ? deg : 0.0) - a[i * n + j];
    }
    return a;
}

static pyod_status residual_scores(pyod_graph_detector *g, const double *r, size_t n, size_t d) {
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < n; i++) s[i] = sqrt(dot(r + i * d, r + i * d, d));
    return fit_state_contamination(&g->state, s, n, g->contamination);
}

/* ---------------- ANOMALOUS ---------------- */

typedef struct {
    pyod_anomalous_params p;
} anomalous_impl;

static pyod_status anomalous_fit(pyod_graph_detector *g, const pyod_matrix *x, const size_t (*edges)[2], size_t ne) {
    const anomalous_impl *m = g->impl;
    size_t n = x->rows, d = x->cols;
    double *lap = laplacian(n, edges, ne), *xn = NEW(double, n * d), *xt = NEW(double, d * n), *w = NEW(double, d * n);
    double *r = NEW(double, n * d), *t1 = NEW(double, n * n), *xwx = NEW(double, n * d), *res = NEW(double, n * d);
    double *t2 = NEW(double, d * d), *grad = NEW(double, d * n);
    pyod_status st = PYOD_ERR_NOMEM;
    if (!lap || !xn || !xt || !w || !r || !t1 || !xwx || !res || !t2 || !grad) goto done;
    for (size_t j = 0; j < d; j++) {
        double nm = 0.0;
        for (size_t i = 0; i < n; i++) nm += PYOD_AT(x, i, j) * PYOD_AT(x, i, j);
        nm = fmax(sqrt(nm), 1e-10);
        for (size_t i = 0; i < n; i++) xn[i * d + j] = PYOD_AT(x, i, j) / nm;
    }
    mat_transpose(xn, xt, n, d);
    for (size_t it = 0; it < m->p.max_iter; it++) {
        mat_mul(xn, w, t1, n, d, n);   /* X W  (n x n) */
        mat_mul(t1, xn, xwx, n, n, d); /* X W X (n x d) */
        for (size_t i = 0; i < n * d; i++) res[i] = xn[i] - xwx[i] - r[i];
        mat_mul(xt, res, t2, d, n, d);  /* X^T res (d x d) */
        mat_mul(t2, xt, grad, d, d, n); /* (X^T res) X^T (d x n) */
        double gn = 0.0;
        for (size_t i = 0; i < d * n; i++) {
            grad[i] = -2.0 * grad[i] + 2.0 * m->p.alpha * w[i];
            gn += grad[i] * grad[i];
        }
        gn = sqrt(gn);
        double scale = gn > 1.0 ? 1.0 / gn : 1.0;
        for (size_t i = 0; i < d * n; i++) w[i] -= 1e-3 * grad[i] * scale;
        mat_mul(xn, w, t1, n, d, n);
        mat_mul(t1, xn, xwx, n, n, d);
        for (size_t i = 0; i < n * d; i++) res[i] = xn[i] - xwx[i];
        if ((st = smoothed_residual(lap, res, n, d, m->p.gamma, m->p.lambda_r, r)) != PYOD_OK) goto done;
    }
    st = residual_scores(g, r, n, d);
done:
    free(lap), free(xn), free(xt), free(w), free(r), free(t1), free(xwx), free(res), free(t2), free(grad);
    return st;
}

pyod_anomalous_params pyod_anomalous_defaults(void) { return (pyod_anomalous_params){1.0, 1.0, 0.01, 100, 0.1}; }

pyod_graph_detector *pyod_anomalous_create(const pyod_anomalous_params *p) {
    anomalous_impl *m = NEW(anomalous_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_anomalous_defaults();
    return graph_detector_new("ANOMALOUS", anomalous_fit, free, m, m->p.contamination);
}

/* ---------------- Radar ---------------- */

typedef struct {
    pyod_radar_params p;
} radar_impl;

static pyod_status radar_fit(pyod_graph_detector *g, const pyod_matrix *x, const size_t (*edges)[2], size_t ne) {
    const radar_impl *m = g->impl;
    size_t n = x->rows, d = x->cols;
    double *lap = laplacian(n, edges, ne), *xt = NEW(double, d * n), *gram = NEW(double, n * n), *eye = NEW(double, n * n);
    double *ginv = NEW(double, n * n), *r = NEW(double, n * d), *xr = NEW(double, n * d), *t = NEW(double, n * n);
    double *w = NEW(double, n * n), *p = NEW(double, n * d);
    pyod_status st = PYOD_ERR_NOMEM;
    if (!lap || !xt || !gram || !eye || !ginv || !r || !xr || !t || !w || !p) goto done;
    mat_transpose(x->data, xt, n, d);
    mat_mul(x->data, xt, gram, n, d, n);
    for (size_t i = 0; i < n; i++) gram[i * n + i] += m->p.alpha, eye[i * n + i] = 1.0;
    st = PYOD_ERR_NUMERIC;
    if (!lin_solve(gram, eye, ginv, n, n)) goto done;
    for (size_t it = 0; it < m->p.max_iter; it++) {
        /* W = (X - R) X^T (X X^T + alpha I)^-1 */
        for (size_t i = 0; i < n * d; i++) xr[i] = x->data[i] - r[i];
        mat_mul(xr, xt, t, n, d, n);
        mat_mul(t, ginv, w, n, n, n);
        mat_mul(w, x->data, p, n, n, d);
        for (size_t i = 0; i < n * d; i++) p[i] = x->data[i] - p[i];
        if ((st = smoothed_residual(lap, p, n, d, m->p.gamma, m->p.gamma, r)) != PYOD_OK) goto done;
    }
    st = residual_scores(g, r, n, d);
done:
    free(lap), free(xt), free(gram), free(eye), free(ginv), free(r), free(xr), free(t), free(w), free(p);
    return st;
}

pyod_radar_params pyod_radar_defaults(void) { return (pyod_radar_params){1.0, 0.01, 100, 0.1}; }

pyod_graph_detector *pyod_radar_create(const pyod_radar_params *p) {
    radar_impl *m = NEW(radar_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_radar_defaults();
    return graph_detector_new("Radar", radar_fit, free, m, m->p.contamination);
}
