/* Neural-network layers (see nn.h). */
#include "deep/nn.h"

/* ---------------- parameters ---------------- */

ad_tensor *nn_new_param(nn_params *ps, int ndim, const size_t *shape) {
    if (ps->n == ps->cap) {
        size_t cap = ps->cap ? ps->cap * 2 : 16;
        ad_tensor **grown = realloc(ps->p, cap * sizeof(ad_tensor *));
        if (!grown) return NULL;
        ps->p = grown;
        ps->cap = cap;
    }
    ad_tensor *t = ad_param(ndim, shape);
    if (t) ps->p[ps->n++] = t;
    return t;
}

void nn_params_free(nn_params *ps) {
    for (size_t i = 0; i < ps->n; i++) ad_param_free(ps->p[i]);
    free(ps->p);
    memset(ps, 0, sizeof(*ps));
}

size_t nn_params_size(const nn_params *ps) {
    size_t s = 0;
    for (size_t i = 0; i < ps->n; i++) s += ps->p[i]->size;
    return s;
}

void nn_snapshot(const nn_params *ps, double *buf) {
    for (size_t i = 0; i < ps->n; i++) {
        memcpy(buf, ps->p[i]->data, ps->p[i]->size * sizeof(double));
        buf += ps->p[i]->size;
    }
}

void nn_restore(nn_params *ps, const double *buf) {
    for (size_t i = 0; i < ps->n; i++) {
        memcpy(ps->p[i]->data, buf, ps->p[i]->size * sizeof(double));
        buf += ps->p[i]->size;
    }
}

void nn_set_requires_grad(nn_params *ps, int on) {
    for (size_t i = 0; i < ps->n; i++) ps->p[i]->requires_grad = on;
}

ad_tensor *nn_activate(nn_act a, ad_tensor *x) {
    switch (a) {
    case NN_RELU: return ad_relu(x);
    case NN_LEAKY_RELU: return ad_leaky_relu(x, 0.01);
    case NN_TANH: return ad_tanh(x);
    case NN_SIGMOID: return ad_sigmoid(x);
    case NN_ELU: return ad_elu(x);
    case NN_SOFTPLUS: return ad_softplus(x);
    default: return x;
    }
}

/* ---------------- linear ---------------- */

void nn_linear_init(nn_linear *l, nn_params *ps, size_t in, size_t out, int bias, nn_init init, double gain, pyod_rng *r) {
    l->in = in;
    l->out = out;
    l->w = nn_new_param(ps, 2, AD_SHAPE(in, out));
    l->b = bias ? nn_new_param(ps, 1, AD_SHAPE(out)) : NULL;
    if (!l->w || (bias && !l->b)) return;
    double fan_in = (double)in, fan_out = (double)out;
    for (size_t i = 0; i < l->w->size; i++) {
        double u = 2.0 * rng_f64(r) - 1.0, v;
        switch (init) {
        case NN_INIT_KAIMING_UNIFORM: v = u * gain * sqrt(3.0 / fan_in); break;
        case NN_INIT_KAIMING_NORMAL: v = rng_normal(r) * gain / sqrt(fan_in); break;
        case NN_INIT_XAVIER_UNIFORM: v = u * gain * sqrt(6.0 / (fan_in + fan_out)); break;
        default: v = u / sqrt(fan_in); /* kaiming_uniform(a = sqrt(5)) */
        }
        l->w->data[i] = v;
    }
    if (l->b)
        for (size_t i = 0; i < out; i++) l->b->data[i] = (2.0 * rng_f64(r) - 1.0) / sqrt(fan_in);
}

ad_tensor *nn_linear_fwd(const nn_linear *l, ad_tensor *x) {
    ad_tensor *x2 = x;
    if (x->ndim != 2) x2 = ad_reshape(x, 2, AD_SHAPE(x->size / l->in, l->in));
    ad_tensor *y = ad_matmul(x2, l->w);
    if (l->b) y = ad_add(y, l->b);
    if (x->ndim != 2) {
        size_t shape[AD_MAXDIM];
        memcpy(shape, x->shape, sizeof shape);
        shape[x->ndim - 1] = l->out;
        y = ad_reshape(y, x->ndim, shape);
    }
    return y;
}

/* ---------------- normalisation ---------------- */

pyod_status nn_batchnorm_init(nn_batchnorm *bn, nn_params *ps, size_t c) {
    bn->c = c;
    bn->momentum = 0.1;
    bn->eps = 1e-5;
    bn->gamma = nn_new_param(ps, 1, AD_SHAPE(c));
    bn->beta = nn_new_param(ps, 1, AD_SHAPE(c));
    bn->running_mean = NEW(double, c);
    bn->running_var = NEW(double, c);
    if (!bn->gamma || !bn->beta || !bn->running_mean || !bn->running_var) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < c; i++) bn->gamma->data[i] = 1.0, bn->running_var[i] = 1.0;
    return PYOD_OK;
}

ad_tensor *nn_batchnorm_fwd(nn_batchnorm *bn, ad_tensor *x, int train) {
    size_t b = x->shape[0], c = bn->c;
    ad_tensor *mean, *var;
    if (train && b > 1) {
        mean = ad_mean_dim(x, 0);
        ad_tensor *centered = ad_sub(x, mean);
        var = ad_mean_dim(ad_mul(centered, centered), 0);
        for (size_t i = 0; i < c; i++) { /* running stats use the unbiased variance */
            bn->running_mean[i] = (1.0 - bn->momentum) * bn->running_mean[i] + bn->momentum * mean->data[i];
            bn->running_var[i] = (1.0 - bn->momentum) * bn->running_var[i] + bn->momentum * var->data[i] * (double)b / (double)(b - 1);
        }
        return ad_add(ad_mul(ad_div(centered, ad_sqrt(ad_add_s(var, bn->eps))), bn->gamma), bn->beta);
    }
    mean = ad_const(2, AD_SHAPE(1, c), bn->running_mean);
    var = ad_const(2, AD_SHAPE(1, c), bn->running_var);
    return ad_add(ad_mul(ad_div(ad_sub(x, mean), ad_sqrt(ad_add_s(var, bn->eps))), bn->gamma), bn->beta);
}

void nn_batchnorm_free(nn_batchnorm *bn) {
    free(bn->running_mean), free(bn->running_var);
    bn->running_mean = bn->running_var = NULL;
}

void nn_layernorm_init(nn_layernorm *ln, nn_params *ps, size_t d) {
    ln->eps = 1e-5;
    ln->gamma = nn_new_param(ps, 1, AD_SHAPE(d));
    ln->beta = nn_new_param(ps, 1, AD_SHAPE(d));
    if (ln->gamma)
        for (size_t i = 0; i < d; i++) ln->gamma->data[i] = 1.0;
}

ad_tensor *nn_layernorm_fwd(const nn_layernorm *ln, ad_tensor *x) {
    int last = x->ndim - 1;
    ad_tensor *centered = ad_sub(x, ad_mean_dim(x, last));
    ad_tensor *var = ad_mean_dim(ad_mul(centered, centered), last);
    return ad_add(ad_mul(ad_div(centered, ad_sqrt(ad_add_s(var, ln->eps))), ln->gamma), ln->beta);
}

/* ---------------- blocks and MLPs ---------------- */

pyod_status nn_mlp_push(nn_mlp *m, nn_params *ps, size_t in, size_t out, int has_act, nn_act act, int bn, double dropout,
                        int bias, int kaiming, pyod_rng *r) {
    nn_block *grown = realloc(m->blocks, (m->n + 1) * sizeof(nn_block));
    if (!grown) return PYOD_ERR_NOMEM;
    m->blocks = grown;
    nn_block *b = &m->blocks[m->n++];
    memset(b, 0, sizeof(*b));
    /* PyOD's kaiming_uniform with a = 0 and the leaky_relu nonlinearity: gain sqrt(2). */
    nn_linear_init(&b->lin, ps, in, out, bias, kaiming ? NN_INIT_KAIMING_UNIFORM : NN_INIT_TORCH, sqrt(2.0), r);
    if (!b->lin.w) return PYOD_ERR_NOMEM;
    b->has_bn = bn;
    if (bn) CHECK(nn_batchnorm_init(&b->bn, ps, out));
    b->has_act = has_act;
    b->act = act;
    b->dropout = dropout;
    return PYOD_OK;
}

ad_tensor *nn_mlp_fwd(nn_mlp *m, ad_tensor *x, int train, pyod_rng *r) {
    for (size_t i = 0; i < m->n; i++) {
        nn_block *b = &m->blocks[i];
        x = nn_linear_fwd(&b->lin, x);
        if (b->has_bn) x = nn_batchnorm_fwd(&b->bn, x, train);
        if (b->has_act) x = nn_activate(b->act, x);
        x = ad_dropout(x, b->dropout, train, r);
    }
    return x;
}

void nn_mlp_free(nn_mlp *m) {
    for (size_t i = 0; i < m->n; i++)
        if (m->blocks[i].has_bn) nn_batchnorm_free(&m->blocks[i].bn);
    free(m->blocks);
    m->blocks = NULL;
    m->n = 0;
}

pyod_status nn_dense_init(nn_dense *d, nn_params *ps, const size_t *sizes, size_t n_sizes, nn_act act, int has_out_act,
                          nn_act out_act, double dropout, nn_drop_at at, pyod_rng *r) {
    memset(d, 0, sizeof(*d));
    if (n_sizes < 2) return PYOD_ERR_PARAM;
    d->n_hidden = n_sizes - 2;
    d->hidden = NEW(nn_linear, d->n_hidden);
    if (!d->hidden) return PYOD_ERR_NOMEM;
    for (size_t i = 0; i < d->n_hidden; i++) {
        nn_linear_init(&d->hidden[i], ps, sizes[i], sizes[i + 1], 1, NN_INIT_TORCH, 1.0, r);
        if (!d->hidden[i].w) return PYOD_ERR_NOMEM;
    }
    nn_linear_init(&d->out, ps, sizes[n_sizes - 2], sizes[n_sizes - 1], 1, NN_INIT_TORCH, 1.0, r);
    if (!d->out.w) return PYOD_ERR_NOMEM;
    d->act = act;
    d->has_out_act = has_out_act;
    d->out_act = out_act;
    d->dropout = dropout;
    d->at = at;
    return PYOD_OK;
}

ad_tensor *nn_dense_fwd(const nn_dense *d, ad_tensor *x, int train, pyod_rng *r) {
    for (size_t i = 0; i < d->n_hidden; i++) {
        if (d->at == NN_DROP_INPUT) x = nn_activate(d->act, nn_linear_fwd(&d->hidden[i], ad_dropout(x, d->dropout, train, r)));
        else x = nn_activate(d->act, ad_dropout(nn_linear_fwd(&d->hidden[i], x), d->dropout, train, r));
    }
    x = nn_linear_fwd(&d->out, x);
    return d->has_out_act ? nn_activate(d->out_act, x) : x;
}

void nn_dense_free(nn_dense *d) {
    free(d->hidden);
    d->hidden = NULL;
}

/* ---------------- LSTM ---------------- */

void nn_lstm_init(nn_lstm *l, nn_params *ps, size_t in, size_t h, pyod_rng *r) {
    l->in = in;
    l->h = h;
    l->w_ih = nn_new_param(ps, 2, AD_SHAPE(in, 4 * h));
    l->w_hh = nn_new_param(ps, 2, AD_SHAPE(h, 4 * h));
    l->b_ih = nn_new_param(ps, 1, AD_SHAPE(4 * h));
    l->b_hh = nn_new_param(ps, 1, AD_SHAPE(4 * h));
    if (!l->w_ih || !l->w_hh || !l->b_ih || !l->b_hh) return;
    ad_tensor *all[4] = {l->w_ih, l->w_hh, l->b_ih, l->b_hh};
    double k = 1.0 / sqrt((double)h); /* PyTorch: U(-1/sqrt(h), 1/sqrt(h)) */
    for (int q = 0; q < 4; q++)
        for (size_t i = 0; i < all[q]->size; i++) all[q]->data[i] = (2.0 * rng_f64(r) - 1.0) * k;
}

ad_tensor *nn_lstm_fwd(const nn_lstm *l, ad_tensor *x) {
    size_t B = x->shape[0], T = x->shape[1], H = l->h;
    ad_tensor *h = ad_const(2, AD_SHAPE(B, H), NULL), *c = ad_const(2, AD_SHAPE(B, H), NULL), *seq = NULL;
    for (size_t t = 0; t < T; t++) {
        ad_tensor *xt = ad_reshape(ad_narrow(x, 1, t, 1), 2, AD_SHAPE(B, l->in));
        ad_tensor *g = ad_add(ad_add(ad_matmul(xt, l->w_ih), ad_matmul(h, l->w_hh)), ad_add(l->b_ih, l->b_hh));
        ad_tensor *i = ad_sigmoid(ad_narrow(g, 1, 0, H)), *f = ad_sigmoid(ad_narrow(g, 1, H, H));
        ad_tensor *gg = ad_tanh(ad_narrow(g, 1, 2 * H, H)), *o = ad_sigmoid(ad_narrow(g, 1, 3 * H, H));
        c = ad_add(ad_mul(f, c), ad_mul(i, gg));
        h = ad_mul(o, ad_tanh(c));
        ad_tensor *h3 = ad_reshape(h, 3, AD_SHAPE(B, 1, H));
        seq = seq ? ad_concat(seq, h3, 1) : h3;
    }
    return seq;
}

/* ---------------- training helpers ---------------- */

pyod_status nn_normalizer_fit(nn_normalizer *z, const pyod_matrix *x, int enabled) {
    memset(z, 0, sizeof(*z));
    if (!enabled) return PYOD_OK;
    z->d = x->cols;
    z->mean = NEW(double, z->d);
    z->std = NEW(double, z->d);
    if (!z->mean || !z->std) return PYOD_ERR_NOMEM;
    col_mean(x, z->mean);
    for (size_t j = 0; j < z->d; j++) {
        double v = 0.0;
        for (size_t i = 0; i < x->rows; i++) v += (PYOD_AT(x, i, j) - z->mean[j]) * (PYOD_AT(x, i, j) - z->mean[j]);
        z->std[j] = sqrt(v / (double)x->rows);
    }
    return PYOD_OK;
}

pyod_status nn_normalizer_apply(const nn_normalizer *z, const pyod_matrix *x, pyod_matrix *out) {
    CHECK(mat_copy(x, out));
    if (z->mean)
        for (size_t i = 0; i < x->rows; i++)
            for (size_t j = 0; j < x->cols; j++) PYOD_AT(out, i, j) = (PYOD_AT(x, i, j) - z->mean[j]) / (z->std[j] + 1e-8);
    return PYOD_OK;
}

void nn_normalizer_free(nn_normalizer *z) {
    free(z->mean), free(z->std);
    memset(z, 0, sizeof(*z));
}

size_t nn_batches(size_t n, size_t bs, int drop_last, pyod_rng *r, size_t *perm) {
    if (bs == 0) bs = 1;
    rng_sample(r, n, n, perm);
    size_t nb = (n + bs - 1) / bs;
    if (drop_last && nb > 1 && n % bs) nb--;
    return nb;
}

size_t nn_batch_len(size_t n, size_t bs, size_t b) {
    if (bs == 0) bs = 1;
    size_t start = b * bs;
    return n - start < bs ? n - start : bs;
}

void nn_row_distances(const ad_tensor *a, const ad_tensor *b, double *out) {
    size_t n = a->shape[0], d = a->size / n;
    for (size_t i = 0; i < n; i++) out[i] = euclidean(a->data + i * d, b->data + i * d, d);
}
