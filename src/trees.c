/* Decision-tree learners used by the ensembles: a random-forest regressor
 * (SUOD's default approximator) and an XGBoost-style gradient-boosted
 * classifier (XGBOD's default supervised model). Trees split greedily on
 * per-sample gradient statistics; with g = -y, h = 1 and lambda = 0 that is
 * a CART regression tree on squared error. */
#include "internal.h"

typedef struct {
    int leaf;
    size_t feature, left, right;
    double threshold, value;
} tnode;

typedef struct {
    tnode *nodes;
    size_t n, cap;
} tree;

typedef struct {
    size_t max_depth, min_samples_split, max_features;
    double lambda, min_child_weight;
} tree_params;

typedef struct {
    const pyod_matrix *x;
    const double *g, *h;
    const tree_params *p;
    pyod_rng *r;
    size_t *feat;
} grow_ctx;

static size_t push_node(tree *t) {
    if (t->n == t->cap) {
        size_t cap = t->cap ? t->cap * 2 : 64;
        tnode *grown = realloc(t->nodes, cap * sizeof(tnode));
        if (!grown) return SIZE_MAX;
        t->nodes = grown;
        t->cap = cap;
    }
    memset(&t->nodes[t->n], 0, sizeof(tnode));
    return t->n++;
}

typedef struct {
    double v;
    size_t i;
} val_idx;

static int cmp_val_idx(const void *a, const void *b) {
    const val_idx *x = a, *y = b;
    if (x->v != y->v) return (x->v > y->v) - (x->v < y->v);
    return (x->i > y->i) - (x->i < y->i);
}

static size_t grow(tree *t, grow_ctx *c, size_t *idx, size_t n, size_t depth) {
    double gs = 0.0, hs = 0.0;
    for (size_t i = 0; i < n; i++) gs += c->g[idx[i]], hs += c->h[idx[i]];
    size_t me = push_node(t);
    if (me == SIZE_MAX) return SIZE_MAX;
    t->nodes[me].leaf = 1;
    t->nodes[me].value = -gs / (hs + c->p->lambda);
    if (depth >= c->p->max_depth || n < c->p->min_samples_split) return me;
    size_t d = c->x->cols, nf = c->p->max_features < 1 ? 1 : c->p->max_features > d ? d : c->p->max_features;
    rng_sample(c->r, d, nf, c->feat);
    double parent = gs * gs / (hs + c->p->lambda), best_gain = 0.0, best_thr = 0.0;
    size_t best_f = SIZE_MAX;
    val_idx *vi = NEW(val_idx, n);
    if (!vi) return me;
    for (size_t fi = 0; fi < nf; fi++) {
        size_t f = c->feat[fi];
        for (size_t k = 0; k < n; k++) vi[k].v = PYOD_AT(c->x, idx[k], f), vi[k].i = idx[k];
        qsort(vi, n, sizeof(val_idx), cmp_val_idx);
        double gl = 0.0, hl = 0.0;
        for (size_t k = 0; k + 1 < n; k++) {
            gl += c->g[vi[k].i];
            hl += c->h[vi[k].i];
            double v = vi[k].v, nv = vi[k + 1].v;
            if (v == nv) continue;
            double gr = gs - gl, hr = hs - hl;
            if (hl < c->p->min_child_weight || hr < c->p->min_child_weight) continue;
            double gain = gl * gl / (hl + c->p->lambda) + gr * gr / (hr + c->p->lambda) - parent;
            if (gain > 1e-12 && gain > best_gain) best_gain = gain, best_f = f, best_thr = 0.5 * (v + nv);
        }
    }
    free(vi);
    if (best_f == SIZE_MAX) return me;
    size_t mid = 0;
    for (size_t k = 0; k < n; k++)
        if (PYOD_AT(c->x, idx[k], best_f) <= best_thr) {
            size_t tmp = idx[k];
            idx[k] = idx[mid];
            idx[mid++] = tmp;
        }
    size_t l = grow(t, c, idx, mid, depth + 1);
    size_t r = grow(t, c, idx + mid, n - mid, depth + 1);
    if (l == SIZE_MAX || r == SIZE_MAX) return SIZE_MAX;
    tnode *nd = &t->nodes[me];
    nd->leaf = 0;
    nd->feature = best_f;
    nd->threshold = best_thr;
    nd->left = l;
    nd->right = r;
    return me;
}

static double tree_predict(const tree *t, const double *row) {
    size_t i = 0;
    while (!t->nodes[i].leaf) i = row[t->nodes[i].feature] <= t->nodes[i].threshold ? t->nodes[i].left : t->nodes[i].right;
    return t->nodes[i].value;
}

static int tree_fit(tree *t, const pyod_matrix *x, const double *g, const double *h, size_t *idx, size_t n,
                    const tree_params *p, pyod_rng *r) {
    memset(t, 0, sizeof(*t));
    size_t *feat = NEW(size_t, x->cols);
    if (!feat) return 0;
    grow_ctx c = {x, g, h, p, r, feat};
    size_t root = grow(t, &c, idx, n, 0);
    free(feat);
    return root != SIZE_MAX;
}

/* ---------------- random forest regressor ---------------- */

struct rf_regressor {
    tree *trees;
    size_t n;
};

rf_regressor *rf_fit(const pyod_matrix *x, const double *y, size_t n_estimators, uint64_t seed) {
    size_t n = x->rows;
    rf_regressor *f = NEW(rf_regressor, 1);
    double *g = NEW(double, n), *h = NEW(double, n);
    size_t *idx = NEW(size_t, n);
    if (!f || !g || !h || !idx) goto fail;
    f->trees = NEW(tree, n_estimators);
    if (!f->trees) goto fail;
    for (size_t i = 0; i < n; i++) g[i] = -y[i], h[i] = 1.0;
    tree_params p = {SIZE_MAX, 2, x->cols, 0.0, 1.0};
    pyod_rng r;
    rng_seed(&r, seed);
    for (size_t t = 0; t < n_estimators; t++) {
        for (size_t i = 0; i < n; i++) idx[i] = rng_below(&r, n); /* bootstrap */
        if (!tree_fit(&f->trees[t], x, g, h, idx, n, &p, &r)) goto fail;
        f->n++;
    }
    free(g), free(h), free(idx);
    return f;
fail:
    free(g), free(h), free(idx);
    rf_free(f);
    return NULL;
}

void rf_predict(const rf_regressor *f, const pyod_matrix *x, double *out) {
    for (size_t i = 0; i < x->rows; i++) {
        double s = 0.0;
        for (size_t t = 0; t < f->n; t++) s += tree_predict(&f->trees[t], &PYOD_AT(x, i, 0));
        out[i] = s / (double)f->n;
    }
}

void rf_free(rf_regressor *f) {
    if (!f) return;
    for (size_t t = 0; t < f->n; t++) free(f->trees[t].nodes);
    free(f->trees);
    free(f);
}

/* ---------------- gradient-boosted classifier ---------------- */

typedef struct {
    pyod_gbdt_params p;
    double base;
    tree *trees;
    size_t n;
} gbdt;

static double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

static void gbdt_reset(gbdt *m) {
    for (size_t t = 0; t < m->n; t++) free(m->trees[t].nodes);
    free(m->trees);
    m->trees = NULL;
    m->n = 0;
}

static pyod_status gbdt_fit(void *self, const pyod_matrix *x, const uint8_t *y) {
    gbdt *m = self;
    size_t n = x->rows;
    gbdt_reset(m);
    double b = fmin(fmax(m->p.base_score, 1e-6), 1.0 - 1e-6);
    m->base = log(b / (1.0 - b));
    m->trees = NEW(tree, m->p.n_estimators);
    double *margin = NEW(double, n), *g = NEW(double, n), *h = NEW(double, n);
    size_t *idx = NEW(size_t, n);
    pyod_status st = PYOD_ERR_NOMEM;
    if (!m->trees || !margin || !g || !h || !idx) goto done;
    for (size_t i = 0; i < n; i++) margin[i] = m->base;
    tree_params tp = {m->p.max_depth, 2, x->cols, m->p.lambda, m->p.min_child_weight};
    pyod_rng r;
    rng_seed(&r, m->p.random_state);
    for (size_t t = 0; t < m->p.n_estimators; t++) {
        for (size_t i = 0; i < n; i++) {
            double pr = sigmoid(margin[i]);
            g[i] = pr - (double)y[i];
            h[i] = fmax(pr * (1.0 - pr), 1e-16);
            idx[i] = i;
        }
        if (!tree_fit(&m->trees[t], x, g, h, idx, n, &tp, &r)) goto done;
        m->n++;
        for (size_t i = 0; i < n; i++) margin[i] += m->p.learning_rate * tree_predict(&m->trees[t], &PYOD_AT(x, i, 0));
    }
    st = PYOD_OK;
done:
    free(margin), free(g), free(h), free(idx);
    return st;
}

static pyod_status gbdt_proba(const void *self, const pyod_matrix *x, double *out) {
    const gbdt *m = self;
    for (size_t i = 0; i < x->rows; i++) {
        double z = m->base;
        for (size_t t = 0; t < m->n; t++) z += m->p.learning_rate * tree_predict(&m->trees[t], &PYOD_AT(x, i, 0));
        out[i] = sigmoid(z);
    }
    return PYOD_OK;
}

static void gbdt_destroy(void *self) {
    if (!self) return;
    gbdt_reset(self);
    free(self);
}

pyod_gbdt_params pyod_gbdt_defaults(void) { return (pyod_gbdt_params){100, 6, 0.3, 1.0, 1.0, 0.5, 0}; }

pyod_classifier pyod_gbdt_classifier(const pyod_gbdt_params *p) {
    pyod_classifier c = {NULL, gbdt_fit, gbdt_proba, gbdt_destroy};
    gbdt *m = NEW(gbdt, 1);
    if (m) m->p = p ? *p : pyod_gbdt_defaults();
    c.self = m;
    return c;
}
