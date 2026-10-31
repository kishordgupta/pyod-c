/* HDBSCAN-based outlier detection (Campello et al., 2013). Port of
 * pyod.models.hdbscan, including the parts of sklearn's HDBSCAN it relies on:
 * Prim's MST over mutual reachability, condensed tree, excess-of-mass cluster
 * selection and membership probabilities. Training scores are
 * 1 - probability; new samples get the inverse-distance-weighted score of
 * their nearest training points. */
#include "internal.h"

typedef struct {
    pyod_hdbscan_params p;
    pyod_matrix train;
} hdb_impl;

typedef struct {
    size_t a, b;
    double w;
} edge;

typedef struct {
    size_t left, right, size;
    double dist;
} merge;

typedef struct {
    size_t parent, child, size;
    double lambda;
} cnode;

static int cmp_edge(const void *x, const void *y) {
    double a = ((const edge *)x)->w, b = ((const edge *)y)->w;
    return (a > b) - (a < b);
}

/* sklearn mst_from_data_matrix (Prim's algorithm, alpha = 1). */
static void mst(const pyod_matrix *x, const double *core, edge *out) {
    size_t n = x->rows, current = 0;
    uint8_t *in_tree = NEW(uint8_t, n);
    double *min_reach = NEW(double, n);
    size_t *src = NEW(size_t, n);
    if (!in_tree || !min_reach || !src) goto done;
    for (size_t i = 0; i < n; i++) min_reach[i] = INFINITY;
    for (size_t e = 0; e + 1 < n; e++) {
        in_tree[current] = 1;
        double best = 1.7976931348623157e308;
        size_t s = 0, nn = 0;
        for (size_t j = 0; j < n; j++) {
            if (in_tree[j]) continue;
            double mr = fmax(fmax(core[current], core[j]), euclidean(&PYOD_AT(x, current, 0), &PYOD_AT(x, j, 0), x->cols));
            if (mr < min_reach[j]) {
                min_reach[j] = mr;
                src[j] = current;
                if (mr < best) best = mr, s = current, nn = j;
            } else if (min_reach[j] < best) {
                best = min_reach[j], s = src[j], nn = j;
            }
        }
        out[e] = (edge){s, nn, best};
        current = nn;
    }
done:
    free(in_tree), free(min_reach), free(src);
}

static size_t uf_find(size_t *parent, size_t x) {
    size_t root = x;
    while (parent[root] != SIZE_MAX) root = parent[root];
    while (parent[x] != SIZE_MAX && parent[x] != root) {
        size_t up = parent[x];
        parent[x] = root;
        x = up;
    }
    return root;
}

/* Breadth-first list of hierarchy nodes below `root` (points < n are leaves). */
static size_t bfs(const merge *h, size_t n, size_t root, size_t *out) {
    size_t len = 0, head = 0;
    out[len++] = root;
    while (head < len) {
        size_t x = out[head++];
        if (x >= n) {
            out[len++] = h[x - n].left;
            out[len++] = h[x - n].right;
        }
    }
    return len;
}

static pyod_status hdbscan_probabilities(const pyod_matrix *x, size_t mcs, size_t ms, double *prob) {
    size_t n = x->rows, root = 2 * (n - 1);
    double *core = NEW(double, n), *dist = NEW(double, ms);
    edge *edges = NEW(edge, n - 1);
    merge *h = NEW(merge, n - 1);
    size_t *parent = NEW(size_t, 2 * n), *usize = NEW(size_t, 2 * n), *relabel = NEW(size_t, root + 1);
    size_t *nodes = NEW(size_t, 2 * n), *sub = NEW(size_t, 2 * n);
    uint8_t *ignore = NEW(uint8_t, root + 1);
    cnode *tree = NEW(cnode, 4 * n);
    double *births = NEW(double, 2 * n + 2), *stability = NEW(double, 2 * n + 2), *deaths = NEW(double, 2 * n + 2);
    uint8_t *is_cluster = NEW(uint8_t, 2 * n + 2), *selected_parent = NEW(uint8_t, 2 * n + 2);
    size_t *up = NEW(size_t, 2 * n + 2), *queue = NEW(size_t, 2 * n + 2);
    pyod_status st = PYOD_ERR_NOMEM;
    if (!core || !dist || !edges || !h || !parent || !usize || !relabel || !nodes || !sub || !ignore || !tree ||
        !births || !stability || !deaths || !is_cluster || !selected_parent || !up || !queue)
        goto done;
    /* Core distance: distance to the ms-th neighbour, counting the point itself. */
    for (size_t i = 0; i < n; i++) {
        k_nearest(x, &PYOD_AT(x, i, 0), ms, SIZE_MAX, NULL, dist);
        core[i] = dist[ms - 1];
    }
    mst(x, core, edges);
    qsort(edges, n - 1, sizeof(edge), cmp_edge);
    /* make_single_linkage */
    for (size_t i = 0; i < 2 * n; i++) parent[i] = SIZE_MAX, usize[i] = i < n;
    for (size_t e = 0, next = n; e + 1 < n; e++, next++) {
        size_t ra = uf_find(parent, edges[e].a), rb = uf_find(parent, edges[e].b);
        parent[ra] = parent[rb] = next;
        usize[next] = usize[ra] + usize[rb];
        h[e] = (merge){ra, rb, usize[next], edges[e].w};
    }
    /* _condense_tree */
    size_t nt = 0, next_label = n + 1, len = bfs(h, n, root, nodes);
    relabel[root] = n;
    for (size_t t = 0; t < len; t++) {
        size_t node = nodes[t];
        if (ignore[node] || node < n) continue;
        merge m = h[node - n];
        double lambda = m.dist > 0.0 ? 1.0 / m.dist : INFINITY;
        size_t lc = m.left >= n ? h[m.left - n].size : 1, rc = m.right >= n ? h[m.right - n].size : 1;
        size_t p = relabel[node];
        int big_l = lc >= mcs, big_r = rc >= mcs;
        if (big_l && big_r) {
            relabel[m.left] = next_label++;
            tree[nt++] = (cnode){p, relabel[m.left], lc, lambda};
            relabel[m.right] = next_label++;
            tree[nt++] = (cnode){p, relabel[m.right], rc, lambda};
            continue;
        }
        if (!big_l && big_r) relabel[m.right] = p;
        if (big_l && !big_r) relabel[m.left] = p;
        size_t drop[2] = {big_l ? SIZE_MAX : m.left, big_r ? SIZE_MAX : m.right};
        for (int k = 0; k < 2; k++) {
            if (drop[k] == SIZE_MAX) continue;
            size_t sl = bfs(h, n, drop[k], sub);
            for (size_t q = 0; q < sl; q++) {
                if (sub[q] < n) tree[nt++] = (cnode){p, sub[q], 1, lambda};
                ignore[sub[q]] = 1;
            }
        }
    }
    /* _compute_stability; cluster labels live in [n, next_label). */
    size_t nl = next_label;
    for (size_t c = 0; c < nl; c++) births[c] = NAN, stability[c] = 0.0, deaths[c] = 0.0, up[c] = SIZE_MAX;
    for (size_t t = 0; t < nt; t++) births[tree[t].child] = tree[t].lambda;
    births[n] = 0.0;
    uint8_t *has = NEW(uint8_t, nl);
    if (!has) goto done;
    for (size_t t = 0; t < nt; t++) {
        stability[tree[t].parent] += (tree[t].lambda - births[tree[t].parent]) * (double)tree[t].size;
        has[tree[t].parent] = 1;
    }
    /* Excess of mass, from the leaves (largest ids) up, excluding the root. */
    for (size_t c = nl; c-- > n + 1;) {
        if (!has[c]) continue;
        double subst = 0.0;
        for (size_t t = 0; t < nt; t++)
            if (tree[t].size > 1 && tree[t].parent == c && has[tree[t].child]) subst += stability[tree[t].child];
        is_cluster[c] = 1;
        if (subst > stability[c]) {
            is_cluster[c] = 0;
            stability[c] = subst;
        } else {
            size_t qh = 0, ql = 0;
            queue[ql++] = c;
            while (qh < ql) {
                size_t q = queue[qh++];
                for (size_t t = 0; t < nt; t++)
                    if (tree[t].size > 1 && tree[t].parent == q) {
                        is_cluster[tree[t].child] = 0;
                        queue[ql++] = tree[t].child;
                    }
            }
        }
    }
    free(has);
    /* Label points with their selected ancestor cluster (or noise), then
     * probabilities = lambda / max lambda of the cluster. */
    for (size_t t = 0; t < nt; t++)
        if (!(tree[t].child >= n && is_cluster[tree[t].child])) up[tree[t].child] = tree[t].parent;
    for (size_t t = 0; t < nt; t++) deaths[tree[t].parent] = fmax(deaths[tree[t].parent], tree[t].lambda);
    for (size_t i = 0; i < n; i++) prob[i] = 0.0;
    for (size_t t = 0; t < nt; t++) {
        size_t pt = tree[t].child;
        if (pt >= n) continue;
        size_t c = pt;
        while (up[c] != SIZE_MAX) c = up[c];
        if (c == n) continue; /* noise */
        double ml = deaths[c];
        prob[pt] = ml == 0.0 || isinf(tree[t].lambda) ? 1.0 : fmin(tree[t].lambda, ml) / ml;
    }
    st = PYOD_OK;
done:
    free(core), free(dist), free(edges), free(h), free(parent), free(usize), free(relabel), free(nodes), free(sub);
    free(ignore), free(tree), free(births), free(stability), free(deaths), free(is_cluster), free(selected_parent);
    free(up), free(queue);
    return st;
}

static pyod_status hdb_fit(pyod_detector *d, const pyod_matrix *x) {
    hdb_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t n = x->rows, ms = m->p.min_samples ? m->p.min_samples : m->p.min_cluster_size;
    if (n < 2 || ms == 0 || ms > n || m->p.min_cluster_size < 2) return PYOD_ERR_PARAM;
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    pyod_status st = hdbscan_probabilities(x, m->p.min_cluster_size, ms, s);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    for (size_t i = 0; i < n; i++) s[i] = 1.0 - s[i];
    pyod_matrix_free(&m->train);
    if ((st = mat_copy(x, &m->train)) != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(d, s, n, m->p.contamination);
}

static pyod_status hdb_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    const hdb_impl *m = d->impl;
    size_t k = m->p.min_cluster_size < m->train.rows ? m->p.min_cluster_size : m->train.rows;
    size_t *idx = NEW(size_t, k);
    double *dist = NEW(double, k);
    if (!idx || !dist) {
        free(idx), free(dist);
        return PYOD_ERR_NOMEM;
    }
    for (size_t i = 0; i < x->rows; i++) {
        k_nearest(&m->train, &PYOD_AT(x, i, 0), k, SIZE_MAX, idx, dist);
        double wsum = 0.0, acc = 0.0;
        for (size_t t = 0; t < k; t++) wsum += 1.0 / (dist[t] + 1e-10);
        for (size_t t = 0; t < k; t++) acc += (1.0 / (dist[t] + 1e-10)) / wsum * d->state.scores[idx[t]];
        out[i] = acc;
    }
    free(idx), free(dist);
    return PYOD_OK;
}

static void hdb_destroy(void *p) {
    hdb_impl *m = p;
    pyod_matrix_free(&m->train);
    free(m);
}

static const pyod_vtable HDB_VT = {"HDBSCAN", hdb_fit, NULL, hdb_score, hdb_destroy};

pyod_hdbscan_params pyod_hdbscan_defaults(void) { return (pyod_hdbscan_params){5, 0, 0.1}; }

pyod_detector *pyod_hdbscan_create(const pyod_hdbscan_params *p) {
    hdb_impl *m = NEW(hdb_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_hdbscan_defaults();
    return detector_new(&HDB_VT, m);
}
