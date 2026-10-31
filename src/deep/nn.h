/* Neural-network layers on top of the autodiff engine, mirroring the PyTorch
 * modules PyOD uses (Linear, BatchNorm1d, LayerNorm, Dropout, LSTM) and PyOD's
 * own LinearBlock, plus dense graph layers (GCNConv, GATConv). */
#ifndef PYOD_NN_H
#define PYOD_NN_H

#include "deep/ad.h"

/* Parameter list (owns the tensors). */
typedef struct {
    ad_tensor **p;
    size_t n, cap;
} nn_params;
ad_tensor *nn_new_param(nn_params *ps, int ndim, const size_t *shape);
void nn_params_free(nn_params *ps);
size_t nn_params_size(const nn_params *ps); /* total number of scalars */
void nn_snapshot(const nn_params *ps, double *buf);
void nn_restore(nn_params *ps, const double *buf);
void nn_set_requires_grad(nn_params *ps, int on);

typedef enum {
    NN_INIT_TORCH = 0,      /* PyTorch nn.Linear default */
    NN_INIT_KAIMING_UNIFORM, /* gain-scaled, fan_in */
    NN_INIT_KAIMING_NORMAL,
    NN_INIT_XAVIER_UNIFORM
} nn_init;

typedef enum { NN_RELU = 0, NN_LEAKY_RELU, NN_TANH, NN_SIGMOID, NN_ELU, NN_SOFTPLUS, NN_IDENTITY } nn_act;
ad_tensor *nn_activate(nn_act a, ad_tensor *x);

typedef struct {
    ad_tensor *w, *b; /* w is in x out */
    size_t in, out;
} nn_linear;
void nn_linear_init(nn_linear *l, nn_params *ps, size_t in, size_t out, int bias, nn_init init, double gain, pyod_rng *r);
/* x is [..., in]; leading dims are flattened for the product. */
ad_tensor *nn_linear_fwd(const nn_linear *l, ad_tensor *x);

typedef struct {
    ad_tensor *gamma, *beta;
    double *running_mean, *running_var;
    size_t c;
    double momentum, eps;
} nn_batchnorm;
pyod_status nn_batchnorm_init(nn_batchnorm *bn, nn_params *ps, size_t c);
ad_tensor *nn_batchnorm_fwd(nn_batchnorm *bn, ad_tensor *x, int train); /* x [B, C] */
void nn_batchnorm_free(nn_batchnorm *bn);

typedef struct {
    ad_tensor *gamma, *beta;
    double eps;
} nn_layernorm;
void nn_layernorm_init(nn_layernorm *ln, nn_params *ps, size_t d);
ad_tensor *nn_layernorm_fwd(const nn_layernorm *ln, ad_tensor *x); /* over the last dim */

/* PyOD LinearBlock: linear -> [batch norm] -> [activation] -> [dropout]. */
typedef struct {
    nn_linear lin;
    nn_batchnorm bn;
    int has_bn, has_act;
    nn_act act;
    double dropout;
} nn_block;

typedef struct {
    nn_block *blocks;
    size_t n;
} nn_mlp;
/* Append a block. kaiming = PyOD's kaiming_uniform (a = 0); else PyTorch default. */
pyod_status nn_mlp_push(nn_mlp *m, nn_params *ps, size_t in, size_t out, int has_act, nn_act act, int bn, double dropout,
                        int bias, int kaiming, pyod_rng *r);
ad_tensor *nn_mlp_fwd(nn_mlp *m, ad_tensor *x, int train, pyod_rng *r);
void nn_mlp_free(nn_mlp *m);

/* Dropout placement for plain dense nets (AnoGAN: before linear; ALAD: after). */
typedef enum { NN_DROP_INPUT = 0, NN_DROP_AFTER_LINEAR = 1 } nn_drop_at;
typedef struct {
    nn_linear *hidden;
    size_t n_hidden;
    nn_linear out;
    nn_act act;
    int has_out_act;
    nn_act out_act;
    double dropout;
    nn_drop_at at;
} nn_dense;
/* sizes = [input, hidden..., output] (n_sizes >= 2), PyTorch default init. */
pyod_status nn_dense_init(nn_dense *d, nn_params *ps, const size_t *sizes, size_t n_sizes, nn_act act, int has_out_act,
                          nn_act out_act, double dropout, nn_drop_at at, pyod_rng *r);
ad_tensor *nn_dense_fwd(const nn_dense *d, ad_tensor *x, int train, pyod_rng *r);
void nn_dense_free(nn_dense *d);

/* Single-layer LSTM (PyTorch gate order i, f, g, o), batch first. */
typedef struct {
    ad_tensor *w_ih, *w_hh, *b_ih, *b_hh;
    size_t in, h;
} nn_lstm;
void nn_lstm_init(nn_lstm *l, nn_params *ps, size_t in, size_t h, pyod_rng *r);
/* x [B, T, in] -> all hidden states [B, T, h]. */
ad_tensor *nn_lstm_fwd(const nn_lstm *l, ad_tensor *x);

/* ---- training helpers ---- */

/* Column standardisation (x - mean) / (std + 1e-8), as PyOD's TorchDataset. */
typedef struct {
    double *mean, *std;
    size_t d;
} nn_normalizer;
pyod_status nn_normalizer_fit(nn_normalizer *z, const pyod_matrix *x, int enabled);
/* Apply into a newly allocated matrix (a copy when disabled). */
pyod_status nn_normalizer_apply(const nn_normalizer *z, const pyod_matrix *x, pyod_matrix *out);
void nn_normalizer_free(nn_normalizer *z);

/* Shuffled mini-batches: perm is a fresh permutation; with drop_last the final
 * partial batch is skipped unless it is the only one. Returns the batch count;
 * batch b covers perm[b * bs, ...) with length nn_batch_len. */
size_t nn_batches(size_t n, size_t bs, int drop_last, pyod_rng *r, size_t *perm);
size_t nn_batch_len(size_t n, size_t bs, size_t b);

/* Euclidean distance between matching rows of two [n, d] tensors. */
void nn_row_distances(const ad_tensor *a, const ad_tensor *b, double *out);

#endif
