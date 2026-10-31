/* Neural graph detectors: DOMINANT, AnomalyDAE, CoLA, CONAD and GUIDE. Ports of
 * pyod.models.pyg_* (PyTorch Geometric); convolutions are computed densely,
 * which suits graphs up to a few thousand nodes. Transductive like PyOD. */
#include "deep/common.h"

typedef const size_t (*edge_list)[2];

/* PyG gcn_norm with self loops: S[t][s] = 1 / sqrt(deg(s) deg(t)) per edge
 * s -> t (existing self loops replaced), deg = in-degree incl. the self loop. */
static ad_tensor *gcn_matrix(size_t n, edge_list e, size_t ne) {
    ad_tensor *s = ad_const(2, AD_SHAPE(n, n), NULL);
    double *deg = NEW(double, n);
    if (!deg) abort();
    for (size_t i = 0; i < n; i++) deg[i] = 1.0;
    for (size_t k = 0; k < ne; k++)
        if (e[k][0] != e[k][1]) deg[e[k][1]] += 1.0;
    for (size_t i = 0; i < n; i++) s->data[i * n + i] += 1.0 / deg[i];
    for (size_t k = 0; k < ne; k++)
        if (e[k][0] != e[k][1]) s->data[e[k][1] * n + e[k][0]] += 1.0 / sqrt(deg[e[k][0]] * deg[e[k][1]]);
    free(deg);
    return s;
}

/* Binary adjacency A[s][t] = 1 per edge. */
static ad_tensor *adjacency(size_t n, edge_list e, size_t ne) {
    ad_tensor *a = ad_const(2, AD_SHAPE(n, n), NULL);
    for (size_t k = 0; k < ne; k++) a->data[e[k][0] * n + e[k][1]] = 1.0;
    return a;
}

typedef struct {
    nn_linear lin; /* Glorot weights, no bias */
    ad_tensor *bias;
} gcn_layer;

static void gcn_init(gcn_layer *g, nn_params *ps, size_t in, size_t out, pyod_rng *r) {
    nn_linear_init(&g->lin, ps, in, out, 0, NN_INIT_XAVIER_UNIFORM, 1.0, r);
    g->bias = nn_new_param(ps, 1, AD_SHAPE(out));
}

typedef struct {
    gcn_layer *convs;
    size_t n;
    double dropout;
} gcn_encoder;

static pyod_status encoder_init(gcn_encoder *e, nn_params *ps, size_t in, size_t hid, size_t layers, double dropout, pyod_rng *r) {
    e->n = layers ? layers : 1;
    e->dropout = dropout;
    e->convs = NEW(gcn_layer, e->n);
    if (!e->convs) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < e->n; i++) {
        gcn_init(&e->convs[i], ps, i ? hid : in, hid, r);
        if (!e->convs[i].lin.w || !e->convs[i].bias) return PYOD_ERR_NOMEM;
    }
    return PYOD_OK;
}

/* ReLU and dropout between layers, none after the last. */
static ad_tensor *encoder_fwd(const gcn_encoder *e, ad_tensor *s, ad_tensor *x, int train, pyod_rng *r) {
    for (size_t i = 0; i < e->n; i++) {
        x = ad_add(ad_matmul(s, nn_linear_fwd(&e->convs[i].lin, x)), e->convs[i].bias);
        if (i + 1 < e->n) x = ad_dropout(ad_relu(x), e->dropout, train, r);
    }
    return x;
}

/* Per-node alpha * ||A - sigmoid(Z Z^T)||^2 + (1 - alpha) * ||X - X_hat||^2, as [N, 1]. */
static ad_tensor *recon_errors(ad_tensor *adj, ad_tensor *z, ad_tensor *x, ad_tensor *xhat, double alpha) {
    ad_tensor *ahat = ad_sigmoid(ad_matmul(z, ad_transpose(z)));
    ad_tensor *se = ad_sum_dim(ad_pow_s(ad_sub(adj, ahat), 2.0), 1), *ae = ad_sum_dim(ad_pow_s(ad_sub(x, xhat), 2.0), 1);
    return ad_add(ad_mul_s(se, alpha), ad_mul_s(ae, 1.0 - alpha));
}

static pyod_status finish(pyod_graph_detector *g, ad_tensor *scores, size_t n) {
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    memcpy(s, scores->data, n * sizeof(double));
    ad_tape_clear();
    return fit_state_contamination(&g->state, s, n, g->contamination);
}

typedef struct {
    pyod_graph_nn_params p;
} gnn_impl;

/* ---------------- DOMINANT ---------------- */

static pyod_status dominant_fit(pyod_graph_detector *g, const pyod_matrix *x, edge_list e, size_t ne) {
    const pyod_graph_nn_params *p = &((gnn_impl *)g->impl)->p;
    size_t n = x->rows, d = x->cols;
    pyod_rng r;
    rng_seed(&r, p->random_state);
    nn_params ps = {0};
    gcn_encoder enc;
    nn_linear dec;
    pyod_status st = encoder_init(&enc, &ps, d, p->hidden_dim, p->num_layers, p->dropout, &r);
    nn_linear_init(&dec, &ps, p->hidden_dim, d, 1, NN_INIT_TORCH, 1.0, &r);
    ad_optimizer opt = {0};
    if (st == PYOD_OK) st = dec.w ? ad_opt_init(&opt, AD_ADAM, ps.p, ps.n, p->lr, 0, 0) : PYOD_ERR_NOMEM;
    for (size_t ep = 0; st == PYOD_OK && ep < p->epochs; ep++) {
        ad_tensor *xt = ad_from_rows(x, NULL, n), *z = encoder_fwd(&enc, gcn_matrix(n, e, ne), xt, 1, &r);
        ad_zero_grad(ps.p, ps.n);
        ad_backward(ad_mean(recon_errors(adjacency(n, e, ne), z, xt, nn_linear_fwd(&dec, z), p->alpha)));
        ad_tape_clear();
        ad_opt_step(&opt);
    }
    if (st == PYOD_OK) {
        ad_opt_free(&opt);
        ad_tensor *xt = ad_from_rows(x, NULL, n), *z = encoder_fwd(&enc, gcn_matrix(n, e, ne), xt, 0, &r);
        st = finish(g, recon_errors(adjacency(n, e, ne), z, xt, nn_linear_fwd(&dec, z), p->alpha), n);
    }
    free(enc.convs);
    nn_params_free(&ps);
    return st;
}

/* ---------------- AnomalyDAE ---------------- */

typedef struct {
    nn_linear lin; /* in -> heads * out, Glorot, no bias */
    ad_tensor *att_src, *att_dst, *bias;
    size_t heads, out;
    double dropout;
} gat_layer;

/* Multi-head graph attention with heads averaged (GATConv(concat=False)).
 * mask[i][j] = 0 where j -> i is an edge (or i == j), -inf otherwise. */
static ad_tensor *gat_fwd(const gat_layer *g, ad_tensor *mask, ad_tensor *x, int train, pyod_rng *r) {
    size_t n = x->shape[0], h = g->heads, c = g->out;
    ad_tensor *xw = ad_swap(ad_reshape(nn_linear_fwd(&g->lin, x), 3, AD_SHAPE(n, h, c)), 0, 1); /* [H, N, C] */
    ad_tensor *a_src = ad_sum_dim(ad_mul(xw, ad_reshape(g->att_src, 3, AD_SHAPE(h, 1, c))), 2);  /* [H, N, 1] */
    ad_tensor *a_dst = ad_sum_dim(ad_mul(xw, ad_reshape(g->att_dst, 3, AD_SHAPE(h, 1, c))), 2);
    ad_tensor *e = ad_add(ad_leaky_relu(ad_add(a_dst, ad_transpose(a_src)), 0.2), mask); /* [H, i, j] */
    ad_tensor *alpha = ad_dropout(ad_softmax(e), g->dropout, train, r);
    return ad_add(ad_reshape(ad_mean_dim(ad_matmul(alpha, xw), 0), 2, AD_SHAPE(n, c)), g->bias);
}

static ad_tensor *attention_mask(size_t n, edge_list e, size_t ne) {
    ad_tensor *m = ad_const(3, AD_SHAPE(1, n, n), NULL);
    for (size_t i = 0; i < n * n; i++) m->data[i] = -INFINITY;
    for (size_t i = 0; i < n; i++) m->data[i * n + i] = 0.0;
    for (size_t k = 0; k < ne; k++) m->data[e[k][1] * n + e[k][0]] = 0.0;
    return m;
}

static pyod_status anomalydae_fit(pyod_graph_detector *g, const pyod_matrix *x, edge_list e, size_t ne) {
    const pyod_graph_nn_params *p = &((gnn_impl *)g->impl)->p;
    size_t n = x->rows, d = x->cols, emb = p->hidden_dim, h = p->num_heads ? p->num_heads : 1;
    pyod_rng r;
    rng_seed(&r, p->random_state);
    nn_params ps = {0};
    gat_layer gat = {.heads = h, .out = emb, .dropout = p->dropout};
    nn_linear_init(&gat.lin, &ps, d, h * emb, 0, NN_INIT_XAVIER_UNIFORM, 1.0, &r);
    gat.att_src = nn_new_param(&ps, 2, AD_SHAPE(h, emb));
    gat.att_dst = nn_new_param(&ps, 2, AD_SHAPE(h, emb));
    gat.bias = nn_new_param(&ps, 1, AD_SHAPE(emb));
    nn_linear enc, dec;
    nn_linear_init(&enc, &ps, d, emb, 1, NN_INIT_TORCH, 1.0, &r);
    nn_linear_init(&dec, &ps, emb, d, 1, NN_INIT_TORCH, 1.0, &r);
    pyod_status st = gat.lin.w && gat.att_src && gat.att_dst && gat.bias && enc.w && dec.w ? PYOD_OK : PYOD_ERR_NOMEM;
    if (st == PYOD_OK) {
        double a = sqrt(6.0 / (double)(emb + h)); /* glorot on [heads, out] */
        for (size_t i = 0; i < h * emb; i++) gat.att_src->data[i] = a * (2.0 * rng_f64(&r) - 1.0), gat.att_dst->data[i] = a * (2.0 * rng_f64(&r) - 1.0);
    }
    ad_optimizer opt = {0};
    if (st == PYOD_OK) st = ad_opt_init(&opt, AD_ADAM, ps.p, ps.n, p->lr, 0, 0);
    for (int pass = 0; st == PYOD_OK && pass <= 1; pass++) {
        size_t epochs = pass == 0 ? p->epochs : 1;
        for (size_t ep = 0; ep < epochs; ep++) {
            int train = pass == 0;
            ad_tensor *xt = ad_from_rows(x, NULL, n);
            ad_tensor *z = ad_mul_s(ad_add(gat_fwd(&gat, attention_mask(n, e, ne), xt, train, &r), ad_relu(nn_linear_fwd(&enc, xt))), 0.5);
            ad_tensor *err = recon_errors(adjacency(n, e, ne), z, xt, nn_linear_fwd(&dec, z), p->alpha);
            if (!train) {
                st = finish(g, err, n);
                break;
            }
            ad_zero_grad(ps.p, ps.n);
            ad_backward(ad_mean(err));
            ad_tape_clear();
            ad_opt_step(&opt);
        }
    }
    if (opt.params) ad_opt_free(&opt);
    nn_params_free(&ps);
    return st;
}

/* ---------------- CoLA ---------------- */

/* Row-normalised adjacency: each node's context is the mean of its out-neighbours. */
static ad_tensor *context_matrix(size_t n, edge_list e, size_t ne) {
    ad_tensor *m = ad_const(2, AD_SHAPE(n, n), NULL);
    double *deg = NEW(double, n);
    if (!deg) abort();
    for (size_t k = 0; k < ne; k++) deg[e[k][0]] += 1.0;
    for (size_t k = 0; k < ne; k++) m->data[e[k][0] * n + e[k][1]] += 1.0 / fmax(deg[e[k][0]], 1.0);
    free(deg);
    return m;
}

static ad_tensor *ones_col(size_t n, double v) {
    ad_tensor *t = ad_const(2, AD_SHAPE(n, 1), NULL);
    for (size_t i = 0; i < n; i++) t->data[i] = v;
    return t;
}

static pyod_status cola_fit(pyod_graph_detector *g, const pyod_matrix *x, edge_list e, size_t ne) {
    const pyod_graph_nn_params *p = &((gnn_impl *)g->impl)->p;
    size_t n = x->rows, d = x->cols, hid = p->hidden_dim;
    pyod_rng r;
    rng_seed(&r, p->random_state);
    nn_params ps = {0};
    gcn_encoder enc;
    nn_linear bil; /* bilinear discriminator z^T W c + b */
    pyod_status st = encoder_init(&enc, &ps, d, hid, p->num_layers, 0.3, &r);
    nn_linear_init(&bil, &ps, hid, hid, 1, NN_INIT_TORCH, 1.0, &r);
    size_t *perm = NEW(size_t, n);
    ad_optimizer opt = {0};
    if (st == PYOD_OK) st = bil.w && perm ? ad_opt_init(&opt, AD_ADAM, ps.p, ps.n, p->lr, 0, 0) : PYOD_ERR_NOMEM;
    for (size_t ep = 0; st == PYOD_OK && ep < p->epochs; ep++) {
        ad_tensor *s = gcn_matrix(n, e, ne), *ctx_m = context_matrix(n, e, ne);
        ad_tensor *z = encoder_fwd(&enc, s, ad_from_rows(x, NULL, n), 1, &r);
        ad_tensor *ctx = ad_matmul(ctx_m, z);
        rng_sample(&r, n, n, perm);
        ad_tensor *zneg = encoder_fwd(&enc, s, ad_from_rows(x, perm, n), 1, &r);
        ad_tensor *pos = ad_sum_dim(ad_mul(nn_linear_fwd(&bil, z), ctx), 1), *neg = ad_sum_dim(ad_mul(nn_linear_fwd(&bil, zneg), ctx), 1);
        ad_zero_grad(ps.p, ps.n);
        ad_backward(ad_add(ad_bce_logits(pos, ones_col(n, 1.0)), ad_bce_logits(neg, ones_col(n, 0.0))));
        ad_tape_clear();
        ad_opt_step(&opt);
    }
    if (st == PYOD_OK) {
        ad_opt_free(&opt);
        /* As in PyOD, score with dropout active and average 5 passes. */
        double *acc = NEW(double, n);
        if (!acc) st = PYOD_ERR_NOMEM;
        for (int pass = 0; st == PYOD_OK && pass < 5; pass++) {
            ad_tensor *z = encoder_fwd(&enc, gcn_matrix(n, e, ne), ad_from_rows(x, NULL, n), 1, &r);
            ad_tensor *sc = ad_sum_dim(ad_mul(nn_linear_fwd(&bil, z), ad_matmul(context_matrix(n, e, ne), z)), 1);
            for (size_t i = 0; i < n; i++) acc[i] -= sc->data[i] / 5.0;
            ad_tape_clear();
        }
        if (st == PYOD_OK) st = fit_state_contamination(&g->state, acc, n, g->contamination);
    }
    free(enc.convs), free(perm);
    nn_params_free(&ps);
    return st;
}

/* ---------------- CONAD ---------------- */


/* Row-wise cosine similarity as [N, 1]. */
static ad_tensor *cosine_rows(ad_tensor *a, ad_tensor *b) {
    ad_tensor *na = ad_div(a, ad_clamp(ad_sqrt(ad_sum_dim(ad_pow_s(a, 2.0), 1)), 1e-12, INFINITY));
    ad_tensor *nb = ad_div(b, ad_clamp(ad_sqrt(ad_sum_dim(ad_pow_s(b, 2.0), 1)), 1e-12, INFINITY));
    return ad_sum_dim(ad_mul(na, nb), 1);
}

/* Port of _create_anomalous_view: swap some nodes' features and add random edges. */
static pyod_status anomalous_view(const pyod_matrix *x, edge_list e, size_t ne, double ratio, pyod_rng *r,
                                  pyod_matrix *xa, size_t (**ea)[2], size_t *nea) {
    size_t n = x->rows, k = (size_t)((double)n * ratio);
    if (k < 1) k = 1;
    if (k > n) k = n;
    CHECK(mat_copy(x, xa));
    size_t *tg = NEW(size_t, k), *src = NEW(size_t, k), add = (size_t)((double)ne * ratio);
    if (add < 1) add = 1;
    *ea = malloc((ne + add) * sizeof **ea);
    if (!tg || !src || !*ea) {
        free(tg), free(src), free(*ea);
        return PYOD_ERR_NOMEM;
    }
    rng_sample(r, n, k, tg);
    rng_sample(r, n, k, src);
    for (size_t i = 0; i < k; i++) memcpy(&PYOD_AT(xa, tg[i], 0), &PYOD_AT(x, src[i], 0), x->cols * sizeof(double));
    memcpy(*ea, e, ne * sizeof **ea);
    for (size_t i = 0; i < add; i++) (*ea)[ne + i][0] = rng_below(r, n), (*ea)[ne + i][1] = rng_below(r, n);
    *nea = ne + add;
    free(tg), free(src);
    return PYOD_OK;
}

static pyod_status conad_fit(pyod_graph_detector *g, const pyod_matrix *x, edge_list e, size_t ne) {
    const pyod_graph_nn_params *p = &((gnn_impl *)g->impl)->p;
    size_t n = x->rows, d = x->cols;
    pyod_rng r;
    rng_seed(&r, p->random_state);
    nn_params ps = {0};
    gcn_encoder enc;
    nn_linear dec;
    pyod_status st = encoder_init(&enc, &ps, d, p->hidden_dim, p->num_layers, p->dropout, &r);
    nn_linear_init(&dec, &ps, p->hidden_dim, d, 1, NN_INIT_TORCH, 1.0, &r);
    ad_optimizer opt = {0};
    if (st == PYOD_OK) st = dec.w ? ad_opt_init(&opt, AD_ADAM, ps.p, ps.n, p->lr, 0, 0) : PYOD_ERR_NOMEM;
    for (size_t ep = 0; st == PYOD_OK && ep <= p->epochs; ep++) {
        int train = ep < p->epochs;
        pyod_matrix xa;
        size_t (*ea)[2], nea;
        if ((st = anomalous_view(x, e, ne, p->aug_ratio, &r, &xa, &ea, &nea)) != PYOD_OK) break;
        ad_tensor *xt = ad_from_rows(x, NULL, n);
        ad_tensor *z = encoder_fwd(&enc, gcn_matrix(n, e, ne), xt, train, &r);
        ad_tensor *za = encoder_fwd(&enc, gcn_matrix(n, (edge_list)ea, nea), ad_from_rows(&xa, NULL, n), train, &r);
        /* Structure and attribute errors are each averaged, then added. */
        ad_tensor *recon = ad_mul_s(recon_errors(adjacency(n, e, ne), z, xt, nn_linear_fwd(&dec, z), 0.5), 2.0);
        ad_tensor *cos = cosine_rows(z, za);
        if (train) {
            ad_zero_grad(ps.p, ps.n);
            ad_backward(ad_add(ad_neg(ad_mean(cos)), ad_mul_s(ad_mean(recon), p->alpha)));
            ad_tape_clear();
            ad_opt_step(&opt);
        } else {
            st = finish(g, ad_add(ad_add_s(ad_neg(cos), 1.0), ad_mul_s(recon, p->alpha)), n);
        }
        pyod_matrix_free(&xa);
        free(ea);
    }
    if (opt.params) ad_opt_free(&opt);
    free(enc.convs);
    nn_params_free(&ps);
    return st;
}

/* ---------------- GUIDE ---------------- */

static pyod_status guide_fit(pyod_graph_detector *g, const pyod_matrix *x, edge_list e, size_t ne) {
    const pyod_graph_nn_params *p = &((gnn_impl *)g->impl)->p;
    size_t n = x->rows, d = x->cols, nm = 0;
    /* Motif edges: edges that close a triangle, ((A A) * A) != 0. */
    uint8_t *a = NEW(uint8_t, n * n);
    size_t (*motif)[2] = malloc((ne ? ne : 1) * sizeof *motif);
    if (!a || !motif) {
        free(a), free(motif);
        return PYOD_ERR_NOMEM;
    }
    for (size_t k = 0; k < ne; k++) a[e[k][0] * n + e[k][1]] = 1;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            if (!a[i * n + j]) continue;
            for (size_t k = 0; k < n; k++)
                if (a[i * n + k] && a[k * n + j]) {
                    motif[nm][0] = i, motif[nm++][1] = j;
                    break;
                }
        }
    free(a);
    if (nm == 0) { /* GUIDE needs triangles; use DOMINANT or CoLA instead */
        free(motif);
        return PYOD_ERR_PARAM;
    }
    pyod_rng r;
    rng_seed(&r, p->random_state);
    nn_params ps = {0};
    gcn_encoder eo, em;
    nn_linear dop, dmo;
    pyod_status st = encoder_init(&eo, &ps, d, p->hidden_dim, p->num_layers, p->dropout, &r);
    nn_linear_init(&dop, &ps, p->hidden_dim, d, 1, NN_INIT_TORCH, 1.0, &r);
    if (st == PYOD_OK) st = encoder_init(&em, &ps, d, p->hidden_dim, p->num_layers, p->dropout, &r);
    nn_linear_init(&dmo, &ps, p->hidden_dim, d, 1, NN_INIT_TORCH, 1.0, &r);
    ad_optimizer opt = {0};
    if (st == PYOD_OK) st = dop.w && dmo.w ? ad_opt_init(&opt, AD_ADAM, ps.p, ps.n, p->lr, 0, 0) : PYOD_ERR_NOMEM;
    for (size_t ep = 0; st == PYOD_OK && ep <= p->epochs; ep++) {
        int train = ep < p->epochs;
        ad_tensor *xt = ad_from_rows(x, NULL, n);
        ad_tensor *zo = encoder_fwd(&eo, gcn_matrix(n, e, ne), xt, train, &r);
        ad_tensor *zm = encoder_fwd(&em, gcn_matrix(n, (edge_list)motif, nm), xt, train, &r);
        ad_tensor *eo_err = ad_mul_s(recon_errors(adjacency(n, e, ne), zo, xt, nn_linear_fwd(&dop, zo), 0.5), 2.0);
        ad_tensor *em_err = ad_mul_s(recon_errors(adjacency(n, (edge_list)motif, nm), zm, xt, nn_linear_fwd(&dmo, zm), 0.5), 2.0);
        ad_tensor *score = ad_add(ad_mul_s(eo_err, p->alpha), ad_mul_s(em_err, 1.0 - p->alpha));
        if (train) {
            ad_zero_grad(ps.p, ps.n);
            ad_backward(ad_mean(score));
            ad_tape_clear();
            ad_opt_step(&opt);
        } else {
            st = finish(g, score, n);
        }
    }
    if (opt.params) ad_opt_free(&opt);
    free(eo.convs), free(em.convs), free(motif);
    nn_params_free(&ps);
    return st;
}

/* ---------------- constructors ---------------- */

static pyod_graph_nn_params base_params(double lr) {
    pyod_graph_nn_params p;
    memset(&p, 0, sizeof p);
    p.hidden_dim = 64;
    p.num_layers = 2;
    p.epochs = 100;
    p.dropout = 0.3;
    p.lr = lr;
    p.alpha = 0.5;
    p.contamination = 0.1;
    p.aug_ratio = 0.2;
    p.num_heads = 4;
    p.random_state = 42;
    return p;
}

pyod_graph_nn_params pyod_dominant_defaults(void) { return base_params(5e-3); }
pyod_graph_nn_params pyod_anomalydae_defaults(void) { return base_params(5e-3); }
pyod_graph_nn_params pyod_cola_defaults(void) { return base_params(1e-3); }
pyod_graph_nn_params pyod_conad_defaults(void) { return base_params(1e-3); }
pyod_graph_nn_params pyod_guide_defaults(void) { return base_params(5e-3); }

static pyod_graph_detector *make(const char *name, pyod_status (*fit)(pyod_graph_detector *, const pyod_matrix *, edge_list, size_t),
                                 const pyod_graph_nn_params *p, pyod_graph_nn_params defaults) {
    gnn_impl *m = NEW(gnn_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : defaults;
    return graph_detector_new(name, fit, free, m, m->p.contamination);
}

pyod_graph_detector *pyod_dominant_create(const pyod_graph_nn_params *p) { return make("DOMINANT", dominant_fit, p, pyod_dominant_defaults()); }
pyod_graph_detector *pyod_anomalydae_create(const pyod_graph_nn_params *p) { return make("AnomalyDAE", anomalydae_fit, p, pyod_anomalydae_defaults()); }
pyod_graph_detector *pyod_cola_create(const pyod_graph_nn_params *p) { return make("CoLA", cola_fit, p, pyod_cola_defaults()); }
pyod_graph_detector *pyod_conad_create(const pyod_graph_nn_params *p) { return make("CONAD", conad_fit, p, pyod_conad_defaults()); }
pyod_graph_detector *pyod_guide_create(const pyod_graph_nn_params *p) { return make("GUIDE", guide_fit, p, pyod_guide_defaults()); }
