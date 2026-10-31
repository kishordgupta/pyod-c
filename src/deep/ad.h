/* Minimal reverse-mode automatic differentiation for the neural detectors.
 *
 * Tensors have up to 4 dimensions (row-major doubles). Operations record
 * themselves on the calling thread's tape; ad_backward() walks the tape in
 * reverse, and ad_tape_clear() frees every non-parameter tensor. Parameters
 * (ad_param) persist and accumulate gradients until ad_zero_grad(). Binary
 * element-wise ops broadcast like NumPy. */
#ifndef PYOD_AD_H
#define PYOD_AD_H

#include "internal.h"

#define AD_MAXDIM 4

typedef struct ad_tensor ad_tensor;
struct ad_tensor {
    int ndim;
    size_t shape[AD_MAXDIM];
    size_t size;
    double *data, *grad;
    int requires_grad, is_param;
    int op;
    ad_tensor *a, *b;
    double s0, s1;
    size_t i0, i1, i2;
    double *aux; /* op-specific saved values (dropout mask, softmax output, ...) */
};

/* ---- lifetime ---- */
ad_tensor *ad_param(int ndim, const size_t *shape);     /* persistent, zero-initialised, requires grad */
void ad_param_free(ad_tensor *t);
ad_tensor *ad_const(int ndim, const size_t *shape, const double *data); /* tape, copies data (NULL = zeros) */
ad_tensor *ad_from_rows(const pyod_matrix *x, const size_t *idx, size_t n); /* tape, n x cols (idx NULL = first n) */
void ad_tape_clear(void);
void ad_backward(ad_tensor *loss); /* loss must have size 1 */
void ad_zero_grad(ad_tensor **params, size_t n);
/* Gradient-free copy on the tape. */
ad_tensor *ad_detach(ad_tensor *x);
/* Number of tensors currently on the tape (for tests). */
size_t ad_tape_size(void);

/* ---- element-wise (b broadcasts against a, NumPy rules) ---- */
ad_tensor *ad_add(ad_tensor *a, ad_tensor *b);
ad_tensor *ad_sub(ad_tensor *a, ad_tensor *b);
ad_tensor *ad_mul(ad_tensor *a, ad_tensor *b);
ad_tensor *ad_div(ad_tensor *a, ad_tensor *b);
ad_tensor *ad_add_s(ad_tensor *a, double s);
ad_tensor *ad_mul_s(ad_tensor *a, double s);
ad_tensor *ad_pow_s(ad_tensor *a, double p);
ad_tensor *ad_neg(ad_tensor *a);
ad_tensor *ad_exp(ad_tensor *a);
ad_tensor *ad_log(ad_tensor *a);
ad_tensor *ad_sqrt(ad_tensor *a);
ad_tensor *ad_abs(ad_tensor *a);
ad_tensor *ad_cos(ad_tensor *a);
ad_tensor *ad_tanh(ad_tensor *a);
ad_tensor *ad_sigmoid(ad_tensor *a);
ad_tensor *ad_relu(ad_tensor *a);
ad_tensor *ad_leaky_relu(ad_tensor *a, double slope);
ad_tensor *ad_softplus(ad_tensor *a);
ad_tensor *ad_gelu(ad_tensor *a); /* exact (erf) GELU, as torch.nn.GELU */
ad_tensor *ad_elu(ad_tensor *a);
ad_tensor *ad_clamp(ad_tensor *a, double lo, double hi); /* use +-INFINITY for one-sided */

/* ---- shape and linear algebra ---- */
/* 2-D, or batched with identical leading dims: [..., n, k] x [..., k, m]. */
ad_tensor *ad_matmul(ad_tensor *a, ad_tensor *b);
ad_tensor *ad_reshape(ad_tensor *a, int ndim, const size_t *shape);
ad_tensor *ad_swap(ad_tensor *a, int d0, int d1); /* swap two dimensions */
ad_tensor *ad_transpose(ad_tensor *a);            /* swap the last two dimensions */
ad_tensor *ad_concat(ad_tensor *a, ad_tensor *b, int dim);
ad_tensor *ad_narrow(ad_tensor *a, int dim, size_t start, size_t len);
/* [B, L, C] -> [B, L, 3C]: rows l-1, l, l+1 side by side, zero-padded (conv1d k=3). */
ad_tensor *ad_unfold3(ad_tensor *a);

/* ---- reductions ---- */
ad_tensor *ad_sum(ad_tensor *a);  /* -> [1] */
ad_tensor *ad_mean(ad_tensor *a); /* -> [1] */
ad_tensor *ad_sum_dim(ad_tensor *a, int dim);  /* keeps the dim with size 1 */
ad_tensor *ad_mean_dim(ad_tensor *a, int dim); /* keeps the dim with size 1 */
ad_tensor *ad_softmax(ad_tensor *a);           /* over the last dimension */

/* ---- stochastic ---- */
/* Inverted dropout with probability p (identity when !train or p == 0). */
ad_tensor *ad_dropout(ad_tensor *a, double p, int train, pyod_rng *r);
ad_tensor *ad_randn(int ndim, const size_t *shape, double mean, double std, pyod_rng *r); /* tape constant */
ad_tensor *ad_rand(int ndim, const size_t *shape, pyod_rng *r);                           /* U[0, 1) */

/* ---- convenience ---- */
static inline size_t *ad_shape2(size_t *buf, size_t a, size_t b) { buf[0] = a, buf[1] = b; return buf; }
#define AD_SHAPE(...) ((const size_t[]){__VA_ARGS__})
#define AD_NDIM(...) ((int)(sizeof((const size_t[]){__VA_ARGS__}) / sizeof(size_t)))
/* ad_new_param(rows, cols) etc. */
#define AD_PARAM(...) ad_param(AD_NDIM(__VA_ARGS__), AD_SHAPE(__VA_ARGS__))
#define AD_CONST(data, ...) ad_const(AD_NDIM(__VA_ARGS__), AD_SHAPE(__VA_ARGS__), data)
/* Mean binary cross-entropy of sigmoid(z) against target: softplus(z) - target * z. */
ad_tensor *ad_bce_logits(ad_tensor *z, ad_tensor *target);
double ad_item(const ad_tensor *t);

/* ---- optimizers ---- */
typedef enum { AD_ADAM, AD_SGD, AD_RMSPROP } ad_opt_kind;
typedef struct {
    ad_opt_kind kind;
    ad_tensor **params;
    size_t n;
    double lr, beta1, beta2, eps, weight_decay, momentum, alpha;
    double **m, **v; /* per-parameter state */
    long t;
} ad_optimizer;
/* PyTorch defaults: Adam(betas .9/.999, eps 1e-8), SGD, RMSprop(alpha .99, eps 1e-8).
 * weight_decay is an L2 term added to the gradient, as in PyTorch. */
pyod_status ad_opt_init(ad_optimizer *o, ad_opt_kind kind, ad_tensor **params, size_t n, double lr, double weight_decay,
                        double momentum);
void ad_opt_step(ad_optimizer *o);
void ad_opt_free(ad_optimizer *o);

#endif
