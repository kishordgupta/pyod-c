/* Reverse-mode automatic differentiation (see ad.h). */
#include "deep/ad.h"

enum {
    OP_LEAF = 0, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_ADD_S, OP_MUL_S, OP_POW_S, OP_EXP, OP_LOG, OP_SQRT,
    OP_ABS, OP_COS, OP_TANH, OP_SIGMOID, OP_RELU, OP_LEAKY, OP_SOFTPLUS, OP_GELU, OP_ELU, OP_CLAMP,
    OP_MATMUL, OP_RESHAPE, OP_SWAP, OP_CONCAT, OP_NARROW, OP_UNFOLD3, OP_SUM, OP_SUM_DIM, OP_SOFTMAX,
    OP_DROPOUT
};

/* ---------------- tape ---------------- */

typedef struct {
    ad_tensor **items;
    size_t n, cap;
} tape;

static _Thread_local tape g_tape;

static void tensor_free(ad_tensor *t) {
    if (!t) return;
    free(t->data), free(t->grad), free(t->aux), free(t);
}

void ad_tape_clear(void) {
    for (size_t i = 0; i < g_tape.n; i++) tensor_free(g_tape.items[i]);
    g_tape.n = 0;
}

size_t ad_tape_size(void) { return g_tape.n; }

static ad_tensor *alloc_tensor(int ndim, const size_t *shape) {
    ad_tensor *t = NEW(ad_tensor, 1);
    if (!t) return NULL;
    t->ndim = ndim;
    t->size = 1;
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i], t->size *= shape[i];
    t->data = NEW(double, t->size);
    if (!t->data) {
        free(t);
        return NULL;
    }
    return t;
}

/* Allocate a tape tensor; aborts on OOM (the training loops have no recovery path). */
static ad_tensor *node(int ndim, const size_t *shape, int op, ad_tensor *a, ad_tensor *b) {
    ad_tensor *t = alloc_tensor(ndim, shape);
    if (t && g_tape.n == g_tape.cap) {
        size_t cap = g_tape.cap ? g_tape.cap * 2 : 1024;
        ad_tensor **grown = realloc(g_tape.items, cap * sizeof(ad_tensor *));
        if (!grown) {
            tensor_free(t);
            t = NULL;
        } else {
            g_tape.items = grown;
            g_tape.cap = cap;
        }
    }
    if (!t) {
        fprintf(stderr, "pyod: out of memory in autodiff\n");
        abort();
    }
    g_tape.items[g_tape.n++] = t;
    t->op = op;
    t->a = a;
    t->b = b;
    t->requires_grad = (a && a->requires_grad) || (b && b->requires_grad);
    return t;
}

ad_tensor *ad_param(int ndim, const size_t *shape) {
    ad_tensor *t = alloc_tensor(ndim, shape);
    if (!t) return NULL;
    t->grad = NEW(double, t->size);
    if (!t->grad) {
        tensor_free(t);
        return NULL;
    }
    t->requires_grad = t->is_param = 1;
    return t;
}

void ad_param_free(ad_tensor *t) { tensor_free(t); }

ad_tensor *ad_const(int ndim, const size_t *shape, const double *data) {
    ad_tensor *t = node(ndim, shape, OP_LEAF, NULL, NULL);
    if (data) memcpy(t->data, data, t->size * sizeof(double));
    return t;
}

ad_tensor *ad_from_rows(const pyod_matrix *x, const size_t *idx, size_t n) {
    ad_tensor *t = node(2, AD_SHAPE(n, x->cols), OP_LEAF, NULL, NULL);
    for (size_t i = 0; i < n; i++) memcpy(t->data + i * x->cols, &PYOD_AT(x, idx ? idx[i] : i, 0), x->cols * sizeof(double));
    return t;
}

ad_tensor *ad_detach(ad_tensor *x) { return ad_const(x->ndim, x->shape, x->data); }

double ad_item(const ad_tensor *t) { return t->data[0]; }

static void ensure_grad(ad_tensor *t) {
    if (!t->grad) {
        t->grad = NEW(double, t->size);
        if (!t->grad) {
            fprintf(stderr, "pyod: out of memory in autodiff\n");
            abort();
        }
    }
}

void ad_zero_grad(ad_tensor **params, size_t n) {
    for (size_t i = 0; i < n; i++) memset(params[i]->grad, 0, params[i]->size * sizeof(double));
}

/* ---------------- broadcasting ---------------- */

typedef struct {
    size_t shape[4], sa[4], sb[4];
} bcast;

/* Align shapes right to 4 dims; fill output shape and per-operand strides
 * (0 along broadcast dims). Returns 0 if incompatible. */
static int bcast_setup(const ad_tensor *a, const ad_tensor *b, bcast *bc) {
    size_t da[4], db[4];
    for (int i = 0; i < 4; i++) {
        int ia = i - (4 - a->ndim), ib = i - (4 - b->ndim);
        da[i] = ia >= 0 ? a->shape[ia] : 1;
        db[i] = ib >= 0 ? b->shape[ib] : 1;
        if (da[i] != db[i] && da[i] != 1 && db[i] != 1) return 0;
        bc->shape[i] = da[i] > db[i] ? da[i] : db[i];
    }
    size_t ra = 1, rb = 1;
    for (int i = 3; i >= 0; i--) {
        bc->sa[i] = da[i] == 1 ? 0 : ra;
        bc->sb[i] = db[i] == 1 ? 0 : rb;
        ra *= da[i];
        rb *= db[i];
    }
    return 1;
}

static ad_tensor *binary(ad_tensor *a, ad_tensor *b, int op) {
    bcast bc;
    if (!bcast_setup(a, b, &bc)) {
        fprintf(stderr, "pyod: incompatible shapes in autodiff op %d\n", op);
        abort();
    }
    int nd = a->ndim > b->ndim ? a->ndim : b->ndim;
    ad_tensor *t = node(nd, bc.shape + (4 - nd), op, a, b);
    size_t k = 0;
    for (size_t i0 = 0; i0 < bc.shape[0]; i0++)
        for (size_t i1 = 0; i1 < bc.shape[1]; i1++)
            for (size_t i2 = 0; i2 < bc.shape[2]; i2++)
                for (size_t i3 = 0; i3 < bc.shape[3]; i3++, k++) {
                    double x = a->data[i0 * bc.sa[0] + i1 * bc.sa[1] + i2 * bc.sa[2] + i3 * bc.sa[3]];
                    double y = b->data[i0 * bc.sb[0] + i1 * bc.sb[1] + i2 * bc.sb[2] + i3 * bc.sb[3]];
                    t->data[k] = op == OP_ADD ? x + y : op == OP_SUB ? x - y : op == OP_MUL ? x * y : x / y;
                }
    return t;
}

static void binary_backward(ad_tensor *t) {
    ad_tensor *a = t->a, *b = t->b;
    bcast bc;
    bcast_setup(a, b, &bc);
    if (a->requires_grad) ensure_grad(a);
    if (b->requires_grad) ensure_grad(b);
    size_t k = 0;
    for (size_t i0 = 0; i0 < bc.shape[0]; i0++)
        for (size_t i1 = 0; i1 < bc.shape[1]; i1++)
            for (size_t i2 = 0; i2 < bc.shape[2]; i2++)
                for (size_t i3 = 0; i3 < bc.shape[3]; i3++, k++) {
                    size_t oa = i0 * bc.sa[0] + i1 * bc.sa[1] + i2 * bc.sa[2] + i3 * bc.sa[3];
                    size_t ob = i0 * bc.sb[0] + i1 * bc.sb[1] + i2 * bc.sb[2] + i3 * bc.sb[3];
                    double g = t->grad[k], x = a->data[oa], y = b->data[ob];
                    double gx, gy;
                    switch (t->op) {
                    case OP_ADD: gx = g, gy = g; break;
                    case OP_SUB: gx = g, gy = -g; break;
                    case OP_MUL: gx = g * y, gy = g * x; break;
                    default: gx = g / y, gy = -g * x / (y * y);
                    }
                    if (a->requires_grad) a->grad[oa] += gx;
                    if (b->requires_grad) b->grad[ob] += gy;
                }
}

ad_tensor *ad_add(ad_tensor *a, ad_tensor *b) { return binary(a, b, OP_ADD); }
ad_tensor *ad_sub(ad_tensor *a, ad_tensor *b) { return binary(a, b, OP_SUB); }
ad_tensor *ad_mul(ad_tensor *a, ad_tensor *b) { return binary(a, b, OP_MUL); }
ad_tensor *ad_div(ad_tensor *a, ad_tensor *b) { return binary(a, b, OP_DIV); }

/* ---------------- unary ---------------- */

static double gelu_f(double x) { return 0.5 * x * (1.0 + erf(x / sqrt(2.0))); }

static ad_tensor *unary(ad_tensor *a, int op, double s0, double s1) {
    ad_tensor *t = node(a->ndim, a->shape, op, a, NULL);
    t->s0 = s0;
    t->s1 = s1;
    for (size_t i = 0; i < a->size; i++) {
        double x = a->data[i], y;
        switch (op) {
        case OP_ADD_S: y = x + s0; break;
        case OP_MUL_S: y = x * s0; break;
        case OP_POW_S: y = pow(x, s0); break;
        case OP_EXP: y = exp(x); break;
        case OP_LOG: y = log(x); break;
        case OP_SQRT: y = sqrt(x); break;
        case OP_ABS: y = fabs(x); break;
        case OP_COS: y = cos(x); break;
        case OP_TANH: y = tanh(x); break;
        case OP_SIGMOID: y = 1.0 / (1.0 + exp(-x)); break;
        case OP_RELU: y = x > 0.0 ? x : 0.0; break;
        case OP_LEAKY: y = x > 0.0 ? x : s0 * x; break;
        case OP_SOFTPLUS: y = x > 20.0 ? x : log1p(exp(x)); break;
        case OP_GELU: y = gelu_f(x); break;
        case OP_ELU: y = x > 0.0 ? x : expm1(x); break;
        default: y = fmin(fmax(x, s0), s1); /* OP_CLAMP */
        }
        t->data[i] = y;
    }
    return t;
}

static void unary_backward(ad_tensor *t) {
    ad_tensor *a = t->a;
    ensure_grad(a);
    for (size_t i = 0; i < a->size; i++) {
        double x = a->data[i], y = t->data[i], d;
        switch (t->op) {
        case OP_ADD_S: d = 1.0; break;
        case OP_MUL_S: d = t->s0; break;
        case OP_POW_S: d = t->s0 * pow(x, t->s0 - 1.0); break;
        case OP_EXP: d = y; break;
        case OP_LOG: d = 1.0 / x; break;
        case OP_SQRT: d = 0.5 / y; break;
        case OP_ABS: d = sign(x); break;
        case OP_COS: d = -sin(x); break;
        case OP_TANH: d = 1.0 - y * y; break;
        case OP_SIGMOID: d = y * (1.0 - y); break;
        case OP_RELU: d = x > 0.0; break;
        case OP_LEAKY: d = x > 0.0 ? 1.0 : t->s0; break;
        case OP_SOFTPLUS: d = x > 20.0 ? 1.0 : 1.0 / (1.0 + exp(-x)); break;
        case OP_GELU: d = 0.5 * (1.0 + erf(x / sqrt(2.0))) + x * exp(-0.5 * x * x) / sqrt(2.0 * PYOD_PI); break;
        case OP_ELU: d = x > 0.0 ? 1.0 : exp(x); break;
        default: d = (x >= t->s0 && x <= t->s1); /* OP_CLAMP */
        }
        a->grad[i] += t->grad[i] * d;
    }
}

ad_tensor *ad_add_s(ad_tensor *a, double s) { return unary(a, OP_ADD_S, s, 0); }
ad_tensor *ad_mul_s(ad_tensor *a, double s) { return unary(a, OP_MUL_S, s, 0); }
ad_tensor *ad_pow_s(ad_tensor *a, double p) { return unary(a, OP_POW_S, p, 0); }
ad_tensor *ad_neg(ad_tensor *a) { return unary(a, OP_MUL_S, -1.0, 0); }
ad_tensor *ad_exp(ad_tensor *a) { return unary(a, OP_EXP, 0, 0); }
ad_tensor *ad_log(ad_tensor *a) { return unary(a, OP_LOG, 0, 0); }
ad_tensor *ad_sqrt(ad_tensor *a) { return unary(a, OP_SQRT, 0, 0); }
ad_tensor *ad_abs(ad_tensor *a) { return unary(a, OP_ABS, 0, 0); }
ad_tensor *ad_cos(ad_tensor *a) { return unary(a, OP_COS, 0, 0); }
ad_tensor *ad_tanh(ad_tensor *a) { return unary(a, OP_TANH, 0, 0); }
ad_tensor *ad_sigmoid(ad_tensor *a) { return unary(a, OP_SIGMOID, 0, 0); }
ad_tensor *ad_relu(ad_tensor *a) { return unary(a, OP_RELU, 0, 0); }
ad_tensor *ad_leaky_relu(ad_tensor *a, double slope) { return unary(a, OP_LEAKY, slope, 0); }
ad_tensor *ad_softplus(ad_tensor *a) { return unary(a, OP_SOFTPLUS, 0, 0); }
ad_tensor *ad_gelu(ad_tensor *a) { return unary(a, OP_GELU, 0, 0); }
ad_tensor *ad_elu(ad_tensor *a) { return unary(a, OP_ELU, 0, 0); }
ad_tensor *ad_clamp(ad_tensor *a, double lo, double hi) { return unary(a, OP_CLAMP, lo, hi); }

/* ---------------- matmul ---------------- */

ad_tensor *ad_matmul(ad_tensor *a, ad_tensor *b) {
    int nd = a->ndim;
    if (nd < 2 || b->ndim != nd || a->shape[nd - 1] != b->shape[nd - 2]) {
        fprintf(stderr, "pyod: matmul shape mismatch\n");
        abort();
    }
    size_t n = a->shape[nd - 2], k = a->shape[nd - 1], m = b->shape[nd - 1], batch = 1, shape[4];
    for (int i = 0; i < nd - 2; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "pyod: matmul batch mismatch\n");
            abort();
        }
        batch *= a->shape[i];
        shape[i] = a->shape[i];
    }
    shape[nd - 2] = n;
    shape[nd - 1] = m;
    ad_tensor *t = node(nd, shape, OP_MATMUL, a, b);
    for (size_t q = 0; q < batch; q++) mat_mul(a->data + q * n * k, b->data + q * k * m, t->data + q * n * m, n, k, m);
    t->i0 = n, t->i1 = k, t->i2 = m;
    return t;
}

static void matmul_backward(ad_tensor *t) {
    ad_tensor *a = t->a, *b = t->b;
    size_t n = t->i0, k = t->i1, m = t->i2, batch = t->size / (n * m);
    if (a->requires_grad) ensure_grad(a);
    if (b->requires_grad) ensure_grad(b);
    for (size_t q = 0; q < batch; q++) {
        const double *g = t->grad + q * n * m, *ad = a->data + q * n * k, *bd = b->data + q * k * m;
        if (a->requires_grad) { /* dA = G B^T */
            double *ga = a->grad + q * n * k;
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < m; j++) {
                    double gij = g[i * m + j];
                    if (gij == 0.0) continue;
                    for (size_t p = 0; p < k; p++) ga[i * k + p] += gij * bd[p * m + j];
                }
        }
        if (b->requires_grad) { /* dB = A^T G */
            double *gb = b->grad + q * k * m;
            for (size_t i = 0; i < n; i++)
                for (size_t p = 0; p < k; p++) {
                    double aip = ad[i * k + p];
                    if (aip == 0.0) continue;
                    for (size_t j = 0; j < m; j++) gb[p * m + j] += aip * g[i * m + j];
                }
        }
    }
}

/* ---------------- shape ops ---------------- */

ad_tensor *ad_reshape(ad_tensor *a, int ndim, const size_t *shape) {
    ad_tensor *t = node(ndim, shape, OP_RESHAPE, a, NULL);
    if (t->size != a->size) {
        fprintf(stderr, "pyod: reshape size mismatch\n");
        abort();
    }
    memcpy(t->data, a->data, a->size * sizeof(double));
    return t;
}

/* Strides of a tensor aligned right to 4 dims. */
static void strides4(const ad_tensor *t, size_t *shape4, size_t *stride4) {
    size_t s = 1;
    for (int i = 3; i >= 0; i--) {
        int j = i - (4 - t->ndim);
        shape4[i] = j >= 0 ? t->shape[j] : 1;
        stride4[i] = s;
        s *= shape4[i];
    }
}

ad_tensor *ad_swap(ad_tensor *a, int d0, int d1) {
    size_t shape[4];
    memcpy(shape, a->shape, sizeof shape);
    shape[d0] = a->shape[d1];
    shape[d1] = a->shape[d0];
    ad_tensor *t = node(a->ndim, shape, OP_SWAP, a, NULL);
    t->i0 = (size_t)d0;
    t->i1 = (size_t)d1;
    size_t ash[4], ast[4], tsh[4], tst[4];
    strides4(a, ash, ast);
    strides4(t, tsh, tst);
    int off = 4 - a->ndim, e0 = d0 + off, e1 = d1 + off;
    size_t idx[4];
    for (idx[0] = 0; idx[0] < tsh[0]; idx[0]++)
        for (idx[1] = 0; idx[1] < tsh[1]; idx[1]++)
            for (idx[2] = 0; idx[2] < tsh[2]; idx[2]++)
                for (idx[3] = 0; idx[3] < tsh[3]; idx[3]++) {
                    size_t src[4] = {idx[0], idx[1], idx[2], idx[3]};
                    src[e0] = idx[e1];
                    src[e1] = idx[e0];
                    size_t to = idx[0] * tst[0] + idx[1] * tst[1] + idx[2] * tst[2] + idx[3] * tst[3];
                    t->data[to] = a->data[src[0] * ast[0] + src[1] * ast[1] + src[2] * ast[2] + src[3] * ast[3]];
                }
    return t;
}

static void swap_backward(ad_tensor *t) {
    ad_tensor *a = t->a;
    ensure_grad(a);
    size_t ash[4], ast[4], tsh[4], tst[4];
    strides4(a, ash, ast);
    strides4(t, tsh, tst);
    int off = 4 - a->ndim, e0 = (int)t->i0 + off, e1 = (int)t->i1 + off;
    size_t idx[4];
    for (idx[0] = 0; idx[0] < tsh[0]; idx[0]++)
        for (idx[1] = 0; idx[1] < tsh[1]; idx[1]++)
            for (idx[2] = 0; idx[2] < tsh[2]; idx[2]++)
                for (idx[3] = 0; idx[3] < tsh[3]; idx[3]++) {
                    size_t src[4] = {idx[0], idx[1], idx[2], idx[3]};
                    src[e0] = idx[e1];
                    src[e1] = idx[e0];
                    a->grad[src[0] * ast[0] + src[1] * ast[1] + src[2] * ast[2] + src[3] * ast[3]] +=
                        t->grad[idx[0] * tst[0] + idx[1] * tst[1] + idx[2] * tst[2] + idx[3] * tst[3]];
                }
}

ad_tensor *ad_transpose(ad_tensor *a) { return ad_swap(a, a->ndim - 2, a->ndim - 1); }

/* outer = product of dims before `dim`, inner = product after. */
static void split_dims(const ad_tensor *t, int dim, size_t *outer, size_t *inner) {
    *outer = *inner = 1;
    for (int i = 0; i < dim; i++) *outer *= t->shape[i];
    for (int i = dim + 1; i < t->ndim; i++) *inner *= t->shape[i];
}

ad_tensor *ad_concat(ad_tensor *a, ad_tensor *b, int dim) {
    size_t shape[4];
    memcpy(shape, a->shape, sizeof shape);
    shape[dim] = a->shape[dim] + b->shape[dim];
    ad_tensor *t = node(a->ndim, shape, OP_CONCAT, a, b);
    t->i0 = (size_t)dim;
    size_t outer, inner;
    split_dims(a, dim, &outer, &inner);
    size_t la = a->shape[dim] * inner, lb = b->shape[dim] * inner;
    for (size_t o = 0; o < outer; o++) {
        memcpy(t->data + o * (la + lb), a->data + o * la, la * sizeof(double));
        memcpy(t->data + o * (la + lb) + la, b->data + o * lb, lb * sizeof(double));
    }
    return t;
}

static void concat_backward(ad_tensor *t) {
    ad_tensor *a = t->a, *b = t->b;
    size_t outer, inner;
    split_dims(a, (int)t->i0, &outer, &inner);
    size_t la = a->shape[t->i0] * inner, lb = b->shape[t->i0] * inner;
    if (a->requires_grad) ensure_grad(a);
    if (b->requires_grad) ensure_grad(b);
    for (size_t o = 0; o < outer; o++)
        for (size_t i = 0; i < la + lb; i++) {
            double g = t->grad[o * (la + lb) + i];
            if (i < la) {
                if (a->requires_grad) a->grad[o * la + i] += g;
            } else if (b->requires_grad) {
                b->grad[o * lb + i - la] += g;
            }
        }
}

ad_tensor *ad_narrow(ad_tensor *a, int dim, size_t start, size_t len) {
    size_t shape[4];
    memcpy(shape, a->shape, sizeof shape);
    shape[dim] = len;
    ad_tensor *t = node(a->ndim, shape, OP_NARROW, a, NULL);
    t->i0 = (size_t)dim;
    t->i1 = start;
    size_t outer, inner;
    split_dims(a, dim, &outer, &inner);
    for (size_t o = 0; o < outer; o++)
        memcpy(t->data + o * len * inner, a->data + (o * a->shape[dim] + start) * inner, len * inner * sizeof(double));
    return t;
}

static void narrow_backward(ad_tensor *t) {
    ad_tensor *a = t->a;
    ensure_grad(a);
    int dim = (int)t->i0;
    size_t outer, inner, len = t->shape[dim];
    split_dims(a, dim, &outer, &inner);
    for (size_t o = 0; o < outer; o++)
        for (size_t i = 0; i < len * inner; i++) a->grad[(o * a->shape[dim] + t->i1) * inner + i] += t->grad[o * len * inner + i];
}

ad_tensor *ad_unfold3(ad_tensor *a) {
    size_t B = a->shape[0], L = a->shape[1], C = a->shape[2];
    ad_tensor *t = node(3, AD_SHAPE(B, L, 3 * C), OP_UNFOLD3, a, NULL);
    for (size_t b = 0; b < B; b++)
        for (size_t l = 0; l < L; l++)
            for (size_t k = 0; k < 3; k++) {
                long src = (long)l + (long)k - 1;
                if (src < 0 || src >= (long)L) continue;
                memcpy(t->data + (b * L + l) * 3 * C + k * C, a->data + (b * L + (size_t)src) * C, C * sizeof(double));
            }
    return t;
}

static void unfold3_backward(ad_tensor *t) {
    ad_tensor *a = t->a;
    ensure_grad(a);
    size_t B = a->shape[0], L = a->shape[1], C = a->shape[2];
    for (size_t b = 0; b < B; b++)
        for (size_t l = 0; l < L; l++)
            for (size_t k = 0; k < 3; k++) {
                long src = (long)l + (long)k - 1;
                if (src < 0 || src >= (long)L) continue;
                for (size_t c = 0; c < C; c++) a->grad[(b * L + (size_t)src) * C + c] += t->grad[(b * L + l) * 3 * C + k * C + c];
            }
}

/* ---------------- reductions ---------------- */

ad_tensor *ad_sum(ad_tensor *a) {
    ad_tensor *t = node(1, AD_SHAPE(1), OP_SUM, a, NULL);
    t->s0 = 1.0;
    for (size_t i = 0; i < a->size; i++) t->data[0] += a->data[i];
    return t;
}

ad_tensor *ad_mean(ad_tensor *a) {
    ad_tensor *t = ad_sum(a);
    t->s0 = 1.0 / (double)a->size;
    t->data[0] *= t->s0;
    return t;
}

static void sum_backward(ad_tensor *t) {
    ensure_grad(t->a);
    for (size_t i = 0; i < t->a->size; i++) t->a->grad[i] += t->grad[0] * t->s0;
}

ad_tensor *ad_sum_dim(ad_tensor *a, int dim) {
    size_t shape[4];
    memcpy(shape, a->shape, sizeof shape);
    shape[dim] = 1;
    ad_tensor *t = node(a->ndim, shape, OP_SUM_DIM, a, NULL);
    t->i0 = (size_t)dim;
    t->s0 = 1.0;
    size_t outer, inner, n = a->shape[dim];
    split_dims(a, dim, &outer, &inner);
    for (size_t o = 0; o < outer; o++)
        for (size_t k = 0; k < n; k++)
            for (size_t i = 0; i < inner; i++) t->data[o * inner + i] += a->data[(o * n + k) * inner + i];
    return t;
}

ad_tensor *ad_mean_dim(ad_tensor *a, int dim) {
    ad_tensor *t = ad_sum_dim(a, dim);
    t->s0 = 1.0 / (double)a->shape[dim];
    for (size_t i = 0; i < t->size; i++) t->data[i] *= t->s0;
    return t;
}

static void sum_dim_backward(ad_tensor *t) {
    ad_tensor *a = t->a;
    ensure_grad(a);
    size_t outer, inner, n = a->shape[t->i0];
    split_dims(a, (int)t->i0, &outer, &inner);
    for (size_t o = 0; o < outer; o++)
        for (size_t k = 0; k < n; k++)
            for (size_t i = 0; i < inner; i++) a->grad[(o * n + k) * inner + i] += t->grad[o * inner + i] * t->s0;
}

ad_tensor *ad_softmax(ad_tensor *a) {
    ad_tensor *t = node(a->ndim, a->shape, OP_SOFTMAX, a, NULL);
    size_t n = a->shape[a->ndim - 1], rows = a->size / n;
    for (size_t r = 0; r < rows; r++) {
        const double *x = a->data + r * n;
        double *y = t->data + r * n, mx = -INFINITY, s = 0.0;
        for (size_t i = 0; i < n; i++) mx = fmax(mx, x[i]);
        for (size_t i = 0; i < n; i++) s += y[i] = exp(x[i] - mx);
        for (size_t i = 0; i < n; i++) y[i] /= s;
    }
    return t;
}

static void softmax_backward(ad_tensor *t) {
    ad_tensor *a = t->a;
    ensure_grad(a);
    size_t n = a->shape[a->ndim - 1], rows = a->size / n;
    for (size_t r = 0; r < rows; r++) {
        const double *y = t->data + r * n, *g = t->grad + r * n;
        double s = 0.0;
        for (size_t i = 0; i < n; i++) s += g[i] * y[i];
        for (size_t i = 0; i < n; i++) a->grad[r * n + i] += y[i] * (g[i] - s);
    }
}

/* ---------------- stochastic ---------------- */

ad_tensor *ad_dropout(ad_tensor *a, double p, int train, pyod_rng *r) {
    if (!train || p <= 0.0) return a;
    ad_tensor *t = node(a->ndim, a->shape, OP_DROPOUT, a, NULL);
    t->aux = NEW(double, a->size);
    if (!t->aux) abort();
    double keep = 1.0 - p;
    for (size_t i = 0; i < a->size; i++) {
        t->aux[i] = rng_f64(r) < keep ? 1.0 / keep : 0.0;
        t->data[i] = a->data[i] * t->aux[i];
    }
    return t;
}

static void dropout_backward(ad_tensor *t) {
    ensure_grad(t->a);
    for (size_t i = 0; i < t->size; i++) t->a->grad[i] += t->grad[i] * t->aux[i];
}

ad_tensor *ad_randn(int ndim, const size_t *shape, double mean, double std, pyod_rng *r) {
    ad_tensor *t = node(ndim, shape, OP_LEAF, NULL, NULL);
    for (size_t i = 0; i < t->size; i++) t->data[i] = mean + std * rng_normal(r);
    return t;
}

ad_tensor *ad_rand(int ndim, const size_t *shape, pyod_rng *r) {
    ad_tensor *t = node(ndim, shape, OP_LEAF, NULL, NULL);
    for (size_t i = 0; i < t->size; i++) t->data[i] = rng_f64(r);
    return t;
}

ad_tensor *ad_bce_logits(ad_tensor *z, ad_tensor *target) {
    return ad_mean(ad_sub(ad_softplus(z), ad_mul(target, z)));
}

/* ---------------- backward ---------------- */

void ad_backward(ad_tensor *loss) {
    if (!loss->requires_grad) return;
    /* Drop gradients left on the tape by an earlier backward pass, so several
     * backward calls between tape clears stay independent. */
    for (size_t k = 0; k < g_tape.n; k++) {
        free(g_tape.items[k]->grad);
        g_tape.items[k]->grad = NULL;
    }
    ensure_grad(loss);
    loss->grad[0] = 1.0;
    for (size_t k = g_tape.n; k-- > 0;) {
        ad_tensor *t = g_tape.items[k];
        if (!t->grad || !t->requires_grad || t->op == OP_LEAF) continue;
        switch (t->op) {
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: binary_backward(t); break;
        case OP_MATMUL: matmul_backward(t); break;
        case OP_RESHAPE:
            ensure_grad(t->a);
            for (size_t i = 0; i < t->size; i++) t->a->grad[i] += t->grad[i];
            break;
        case OP_SWAP: swap_backward(t); break;
        case OP_CONCAT: concat_backward(t); break;
        case OP_NARROW: narrow_backward(t); break;
        case OP_UNFOLD3: unfold3_backward(t); break;
        case OP_SUM: sum_backward(t); break;
        case OP_SUM_DIM: sum_dim_backward(t); break;
        case OP_SOFTMAX: softmax_backward(t); break;
        case OP_DROPOUT: dropout_backward(t); break;
        default: unary_backward(t);
        }
    }
}

/* ---------------- optimizers ---------------- */

pyod_status ad_opt_init(ad_optimizer *o, ad_opt_kind kind, ad_tensor **params, size_t n, double lr, double weight_decay,
                        double momentum) {
    memset(o, 0, sizeof(*o));
    o->kind = kind;
    o->n = n;
    o->lr = lr;
    o->beta1 = 0.9, o->beta2 = 0.999, o->eps = 1e-8, o->alpha = 0.99;
    o->weight_decay = weight_decay;
    o->momentum = momentum;
    o->params = NEW(ad_tensor *, n);
    o->m = NEW(double *, n);
    o->v = NEW(double *, n);
    if (!o->params || !o->m || !o->v) return PYOD_ERR_NOMEM;
    memcpy(o->params, params, n * sizeof(ad_tensor *));
    for (size_t i = 0; i < n; i++) {
        o->m[i] = NEW(double, params[i]->size);
        o->v[i] = NEW(double, params[i]->size);
        if (!o->m[i] || !o->v[i]) return PYOD_ERR_NOMEM;
    }
    return PYOD_OK;
}

void ad_opt_step(ad_optimizer *o) {
    o->t++;
    double bc1 = 1.0 - pow(o->beta1, (double)o->t), bc2 = 1.0 - pow(o->beta2, (double)o->t);
    for (size_t q = 0; q < o->n; q++) {
        ad_tensor *p = o->params[q];
        if (!p->requires_grad) continue;
        double *m = o->m[q], *v = o->v[q];
        for (size_t i = 0; i < p->size; i++) {
            double g = p->grad[i] + o->weight_decay * p->data[i];
            if (o->kind == AD_ADAM) {
                m[i] = o->beta1 * m[i] + (1.0 - o->beta1) * g;
                v[i] = o->beta2 * v[i] + (1.0 - o->beta2) * g * g;
                p->data[i] -= o->lr * (m[i] / bc1) / (sqrt(v[i] / bc2) + o->eps);
            } else if (o->kind == AD_SGD) {
                if (o->momentum > 0.0) {
                    m[i] = o->t == 1 ? g : o->momentum * m[i] + g;
                    g = m[i];
                }
                p->data[i] -= o->lr * g;
            } else {
                v[i] = o->alpha * v[i] + (1.0 - o->alpha) * g * g;
                p->data[i] -= o->lr * g / (sqrt(v[i]) + o->eps);
            }
        }
    }
}

void ad_opt_free(ad_optimizer *o) {
    for (size_t i = 0; i < o->n; i++) {
        if (o->m) free(o->m[i]);
        if (o->v) free(o->v[i]);
    }
    free(o->params), free(o->m), free(o->v);
    memset(o, 0, sizeof(*o));
}
