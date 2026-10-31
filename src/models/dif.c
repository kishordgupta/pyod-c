/* DIF: deep isolation forest (Xu et al., TKDE 2023). Port of pyod.models.dif.
 * Each ensemble member projects the min-max-scaled data through a random,
 * untrained bias-free tanh network, standardises and squashes the result,
 * and fits a small isolation forest; the score combines isolation depth with
 * distance from split thresholds. As in PyOD, new data's representation is
 * standardised with its own statistics. */
#include "internal.h"

#define DIF_MAX_LAYERS 9

typedef struct {
    double *w[DIF_MAX_LAYERS]; /* layer weights, in x out */
    size_t in[DIF_MAX_LAYERS], out[DIF_MAX_LAYERS];
    itree_node **trees;
} dif_member;

typedef struct {
    pyod_dif_params p;
    size_t d, n_layers, psi;
    double *min, *range;
    dif_member *members;
    size_t n_members;
} dif_impl;

static void free_members(dif_impl *m) {
    for (size_t e = 0; e < m->n_members; e++) {
        for (size_t l = 0; l < m->n_layers; l++) free(m->members[e].w[l]);
        if (m->members[e].trees)
            for (size_t t = 0; t < m->p.n_estimators; t++) itree_free(m->members[e].trees[t]);
        free(m->members[e].trees);
    }
    free(m->members);
    m->members = NULL;
    m->n_members = 0;
}

/* Representation of min-max-scaled rows xs (n x d) -> tanh(standardise(net(xs))). */
static pyod_status represent(const dif_impl *m, const dif_member *mem, const double *xs, size_t n, pyod_matrix *z) {
    size_t width = m->d;
    double *h = NEW(double, n * width);
    if (!h) return PYOD_ERR_NOMEM;
    memcpy(h, xs, n * width * sizeof(double));
    for (size_t l = 0; l < m->n_layers; l++) {
        size_t o = mem->out[l], last = l + 1 == m->n_layers;
        double *nx = NEW(double, n * o);
        if (!nx) {
            free(h);
            return PYOD_ERR_NOMEM;
        }
        mat_mul(h, mem->w[l], nx, n, width, o);
        if (!last) {
            for (size_t i = 0; i < n * o; i++) nx[i] = tanh(nx[i]);
            if (m->p.skip_connection) {
                double *cat = NEW(double, n * (width + o));
                if (!cat) {
                    free(h), free(nx);
                    return PYOD_ERR_NOMEM;
                }
                for (size_t i = 0; i < n; i++) {
                    memcpy(cat + i * (width + o), h + i * width, width * sizeof(double));
                    memcpy(cat + i * (width + o) + width, nx + i * o, o * sizeof(double));
                }
                free(nx);
                nx = cat;
                o += width;
            }
        }
        free(h);
        h = nx;
        width = o;
    }
    *z = pyod_matrix_view(h, n, width);
    pyod_standardize(z);
    for (size_t i = 0; i < n * width; i++) h[i] = tanh(h[i]);
    return PYOD_OK;
}

/* Depth (path length + c(leaf size)) and mean |x_f - threshold| along the path. */
static void path_stats(const itree_node *t, const double *r, double *depth, double *dev) {
    double dsum = 0.0, dn = 0.0, dp = 0.0;
    while (!t->leaf) {
        double diff = fabs(r[t->feature] - t->value);
        if (diff != 0.0) dsum += diff, dn += 1.0;
        dp += 1.0;
        t = r[t->feature] < t->value ? t->left : t->right;
    }
    *depth = dp + itree_c(t->size);
    *dev = dsum / (dn + 1e-6);
}

static pyod_status dif_scores(const dif_impl *m, const pyod_matrix *x, double *out) {
    size_t n = x->rows;
    double *xs = NEW(double, n * m->d);
    if (!xs) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m->d; j++) xs[i * m->d + j] = (PYOD_AT(x, i, j) - m->min[j]) / m->range[j];
    for (size_t i = 0; i < n; i++) out[i] = 0.0;
    double cn = itree_c(m->psi), k = (double)m->p.n_estimators;
    for (size_t e = 0; e < m->n_members; e++) {
        pyod_matrix z;
        pyod_status st = represent(m, &m->members[e], xs, n, &z);
        if (st != PYOD_OK) {
            free(xs);
            return st;
        }
        for (size_t i = 0; i < n; i++) {
            double ds = 0.0, dv = 0.0;
            for (size_t t = 0; t < m->p.n_estimators; t++) {
                double dd, dvv;
                path_stats(m->members[e].trees[t], &PYOD_AT(&z, i, 0), &dd, &dvv);
                ds += dd, dv += dvv;
            }
            out[i] += pow(2.0, -ds / (k * cn)) * (dv / k);
        }
        free(z.data);
    }
    for (size_t i = 0; i < n; i++) out[i] /= (double)m->n_members;
    free(xs);
    return PYOD_OK;
}

static pyod_status dif_fit(pyod_detector *dt, const pyod_matrix *x) {
    dif_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t n = x->rows, d = x->cols;
    if (m->p.n_ensemble == 0 || m->p.n_estimators == 0 || m->p.representation_dim == 0) return PYOD_ERR_PARAM;
    free_members(m);
    free(m->min), free(m->range);
    m->d = d;
    m->min = NEW(double, d);
    m->range = NEW(double, d);
    m->members = NEW(dif_member, m->p.n_ensemble);
    double *xs = NEW(double, n * d);
    size_t *idx = NEW(size_t, n);
    if (!m->min || !m->range || !m->members || !xs || !idx) {
        free(xs), free(idx);
        return PYOD_ERR_NOMEM;
    }
    for (size_t j = 0; j < d; j++) {
        double lo = INFINITY, hi = -INFINITY;
        for (size_t i = 0; i < n; i++) lo = fmin(lo, PYOD_AT(x, i, j)), hi = fmax(hi, PYOD_AT(x, i, j));
        m->min[j] = lo;
        m->range[j] = hi - lo == 0.0 ? 1.0 : hi - lo;
    }
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < d; j++) xs[i * d + j] = (PYOD_AT(x, i, j) - m->min[j]) / m->range[j];
    m->n_layers = 0;
    size_t widths[DIF_MAX_LAYERS];
    for (size_t l = 0; l < 8 && m->p.hidden_neurons[l]; l++) widths[m->n_layers++] = m->p.hidden_neurons[l];
    widths[m->n_layers++] = m->p.representation_dim;
    m->psi = m->p.max_samples < n ? m->p.max_samples : n;
    if (m->psi < 2) m->psi = 2 < n ? 2 : n;
    size_t max_depth = (size_t)ceil(log2((double)m->psi));
    pyod_rng r;
    rng_seed(&r, m->p.random_state);
    pyod_status st = PYOD_OK;
    for (size_t e = 0; e < m->p.n_ensemble && st == PYOD_OK; e++) {
        dif_member *mem = &m->members[e];
        m->n_members++;
        size_t in_dim = d;
        for (size_t l = 0; l < m->n_layers; l++) {
            mem->in[l] = in_dim;
            mem->out[l] = widths[l];
            mem->w[l] = NEW(double, in_dim * widths[l]);
            if (!mem->w[l]) {
                st = PYOD_ERR_NOMEM;
                break;
            }
            for (size_t i = 0; i < in_dim * widths[l]; i++) mem->w[l][i] = rng_normal(&r);
            in_dim = m->p.skip_connection ? in_dim + widths[l] : widths[l];
        }
        if (st != PYOD_OK) break;
        pyod_matrix z;
        if ((st = represent(m, mem, xs, n, &z)) != PYOD_OK) break;
        mem->trees = NEW(itree_node *, m->p.n_estimators);
        if (!mem->trees) st = PYOD_ERR_NOMEM;
        for (size_t t = 0; st == PYOD_OK && t < m->p.n_estimators; t++) {
            rng_sample(&r, n, m->psi, idx);
            mem->trees[t] = itree_build(&z, idx, m->psi, 0, max_depth, &r);
        }
        free(z.data);
    }
    free(xs), free(idx);
    CHECK(st);
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    if ((st = dif_scores(m, x, s)) != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(dt, s, n, m->p.contamination);
}

static pyod_status dif_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return dif_scores(d->impl, x, out); }

static void dif_destroy(void *p) {
    dif_impl *m = p;
    free_members(m);
    free(m->min), free(m->range), free(m);
}

static const pyod_vtable DIF_VT = {"DIF", dif_fit, NULL, dif_score, dif_destroy};

pyod_dif_params pyod_dif_defaults(void) {
    pyod_dif_params p;
    memset(&p, 0, sizeof p);
    p.representation_dim = 20;
    p.hidden_neurons[0] = 500;
    p.hidden_neurons[1] = 100;
    p.n_ensemble = 50;
    p.n_estimators = 6;
    p.max_samples = 256;
    p.contamination = 0.1;
    return p;
}

pyod_detector *pyod_dif_create(const pyod_dif_params *p) {
    dif_impl *m = NEW(dif_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_dif_defaults();
    return detector_new(&DIF_VT, m);
}
