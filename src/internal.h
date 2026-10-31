/* Internal definitions shared by the pyod-c implementation. Not installed. */
#ifndef PYOD_INTERNAL_H
#define PYOD_INTERNAL_H

#include "pyod.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PYOD_PI 3.14159265358979323846
#define PYOD_EULER_GAMMA 0.57721566490153286061
#define PYOD_DBL_EPS 2.220446049250313e-16

/* ------------------------------------------------------------------ */
/* Detector plumbing                                                   */
/* ------------------------------------------------------------------ */

/* Attributes PyOD sets in _process_decision_scores. */
typedef struct {
    double *scores;   /* decision_scores_ */
    uint8_t *labels;  /* labels_ */
    size_t n;
    double threshold; /* threshold_ */
    double mu, sigma;
    double contamination;
} pyod_fit_state;

typedef struct {
    const char *name;
    /* Fit on x and fill `state` (use fit_state_* helpers). */
    pyod_status (*fit)(pyod_detector *d, const pyod_matrix *x);
    /* Optional: supervised fit (XGBOD, DevNet). */
    pyod_status (*fit_supervised)(pyod_detector *d, const pyod_matrix *x, const uint8_t *y);
    /* Score new data; called only once fitted. */
    pyod_status (*score)(const pyod_detector *d, const pyod_matrix *x, double *out);
    void (*destroy)(void *impl);
} pyod_vtable;

struct pyod_detector {
    const pyod_vtable *vt;
    void *impl;
    int fitted;
    pyod_fit_state state;
    /* Number of features seen in fit (0 = not checked). */
    size_t n_features;
};

/* Allocate a detector around `impl` (taking ownership). Returns NULL on OOM
 * after destroying impl. */
pyod_detector *detector_new(const pyod_vtable *vt, void *impl);

/* Fill d->state: threshold at the (1 - contamination) percentile. Takes
 * ownership of `scores` (malloc'd, n entries). */
pyod_status fit_state_contamination(pyod_detector *d, double *scores, size_t n, double contamination);
/* Fill d->state with a fixed threshold. Takes ownership of `scores`. */
pyod_status fit_state_threshold(pyod_detector *d, double *scores, size_t n, double threshold);
void fit_state_clear(pyod_fit_state *s);

pyod_status check_matrix(const pyod_matrix *x);
pyod_status check_contamination(double c);

/* Convenience: score new data through the public API from inside a model. */
#define DET_IMPL(d, T) ((T *)(d)->impl)

/* ------------------------------------------------------------------ */
/* Random numbers (SplitMix64, deterministic per seed)                 */
/* ------------------------------------------------------------------ */

typedef struct {
    uint64_t s;
} pyod_rng;

void rng_seed(pyod_rng *r, uint64_t seed);
uint64_t rng_u64(pyod_rng *r);
double rng_f64(pyod_rng *r);           /* uniform [0, 1) */
size_t rng_below(pyod_rng *r, size_t n); /* uniform [0, n) */
double rng_normal(pyod_rng *r);
/* k distinct indices from [0, n) (partial Fisher-Yates); out has k entries. */
void rng_sample(pyod_rng *r, size_t n, size_t k, size_t *out);

/* ------------------------------------------------------------------ */
/* Statistics                                                          */
/* ------------------------------------------------------------------ */

double vec_mean(const double *v, size_t n);
double vec_std(const double *v, size_t n); /* population (ddof 0) */
/* np.percentile with linear interpolation; q in [0, 100]. */
double vec_percentile(const double *v, size_t n, double q);
double vec_median(const double *v, size_t n);
double vec_skew(const double *v, size_t n);     /* biased, 0 for constant input */
double vec_kurtosis(const double *v, size_t n); /* excess, biased */
double vec_min(const double *v, size_t n);
double vec_max(const double *v, size_t n);
double sign(double x); /* np.sign: 0 maps to 0 */
int cmp_double(const void *a, const void *b);
/* Argsort ascending (stable). */
void argsort(const double *v, size_t n, size_t *idx);
/* numba's default quicksort argsort on unsigned keys (PyOD's SOD tie order). */
void numba_argsort_u32(const uint32_t *v, size_t n, size_t *idx);

double sq_euclidean(const double *a, const double *b, size_t d);
double euclidean(const double *a, const double *b, size_t d);
double dot(const double *a, const double *b, size_t d);

/* Column ECDF P(X <= x) for each entry; out is n x d. */
void column_ecdf(const pyod_matrix *x, double *out);

/* Indices/distances of the k nearest rows of `train` to `q`, nearest first.
 * `skip` (or SIZE_MAX) excludes one training row. Returns the number found. */
size_t k_nearest(const pyod_matrix *train, const double *q, size_t k, size_t skip, size_t *idx, double *dist);

/* sklearn StandardScaler. */
typedef struct {
    double *mean, *scale;
    size_t d;
} std_scaler;
pyod_status scaler_fit(std_scaler *s, const pyod_matrix *x);
/* out may alias x->data. */
void scaler_transform(const std_scaler *s, const pyod_matrix *x, double *out);
void scaler_free(std_scaler *s);

/* Special functions. */
double erf_(double x);
double erfc_(double x);
double lgamma_(double x);
double gamma_p(double a, double x);
double gamma_q(double a, double x);
double chi2_ppf(double q, double df);
double norm_ppf(double p);
double logsumexp(const double *v, size_t n);

/* ------------------------------------------------------------------ */
/* Dense linear algebra (row-major n x m arrays)                       */
/* ------------------------------------------------------------------ */

/* C = A (n x k) * B (k x m). */
void mat_mul(const double *a, const double *b, double *c, size_t n, size_t k, size_t m);
void mat_transpose(const double *a, double *t, size_t n, size_t m);
void col_mean(const pyod_matrix *x, double *mean);
/* Covariance about `center`, divided by n - ddof; out is d x d. */
void covariance(const pyod_matrix *x, const double *center, size_t ddof, double *out);
/* Symmetric eigen-decomposition (tred2 + tql2). Eigenvalues descending;
 * vecs[i * n ...] is the i-th unit eigenvector. */
void sym_eigen(const double *a, size_t n, double *vals, double *vecs);
/* Lower Cholesky factor; returns 0 if not positive definite. */
int cholesky(const double *a, size_t n, double *l);
void forward_solve(const double *l, const double *b, double *y, size_t n);
/* Pseudo-inverse of a symmetric matrix (scipy.linalg.pinvh). */
void pinvh(const double *a, size_t n, double *out);
/* log|det| of a symmetric matrix or -INFINITY if not positive definite. */
double fast_logdet(const double *a, size_t n);
/* Solve A X = B (A n x n, B n x m) in place into x; returns 0 if singular. */
int lin_solve(const double *a, const double *b, double *x, size_t n, size_t m);
/* Squared Mahalanobis distance of each row from loc under precision. */
void mahalanobis_sq(const pyod_matrix *x, const double *loc, const double *prec, double *out);
/* k-means++ seeding + Lloyd. centers k x d, labels n. Returns inertia. */
double kmeans(const pyod_matrix *x, size_t k, size_t n_init, size_t max_iter, pyod_rng *r, double *centers, size_t *labels);
size_t nearest_center(const double *centers, size_t k, size_t d, const double *row, double *dist2);

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

#define NEW(T, n) ((T *)calloc((n) ? (size_t)(n) : 1u, sizeof(T)))
#define CHECK(expr)                     \
    do {                                \
        pyod_status st__ = (expr);      \
        if (st__ != PYOD_OK) return st__; \
    } while (0)

/* Copy of a matrix (allocated). */
pyod_status mat_copy(const pyod_matrix *x, pyod_matrix *out);
/* Rows `idx` of x (allocated). */
pyod_status mat_rows(const pyod_matrix *x, const size_t *idx, size_t k, pyod_matrix *out);
/* Columns `cols` of x (allocated). */
pyod_status mat_cols(const pyod_matrix *x, const size_t *cols, size_t k, pyod_matrix *out);

/* Isolation-tree pieces shared by IForest and DIF. */
typedef struct itree_node {
    int leaf;
    size_t size;
    size_t feature;
    double value;
    struct itree_node *left, *right;
} itree_node;
double itree_c(size_t n);
itree_node *itree_build(const pyod_matrix *x, size_t *idx, size_t n, size_t depth, size_t max_depth, pyod_rng *r);
void itree_free(itree_node *t);
double itree_path(const itree_node *t, const double *row, double depth);

/* Random-forest regressor (SUOD's approximator). */
typedef struct rf_regressor rf_regressor;
rf_regressor *rf_fit(const pyod_matrix *x, const double *y, size_t n_estimators, uint64_t seed);
void rf_predict(const rf_regressor *f, const pyod_matrix *x, double *out);
void rf_free(rf_regressor *f);

/* Port of generate_bagging_indices: draw a count in [min, max), then features.
 * Returns the count written to out (capacity n_features). */
size_t bagging_indices(pyod_rng *r, int bootstrap, size_t n_features, size_t min, size_t max, size_t *out);

/* Kernel functions shared by OCSVM and KPCA. */
double pyod__svm_kernel(pyod_svm_kernel k, int degree, double coef0, double gamma, const double *a, const double *b, size_t d);

/* Time-series helpers (rows = timestamps). Score arrays use NAN for
 * timestamps no window covers. */
size_t ts_windows(const pyod_matrix *x, size_t window, size_t step, pyod_matrix *out);
void ts_map_scores(const double *window_scores, size_t n_windows, size_t window, size_t step, size_t n,
                   pyod_aggregation agg, double *out);
/* Combine per-channel timestamp scores (c arrays of n) into out; a timestamp is
 * covered only if every channel covers it. */
void ts_combine_channels(double **per_channel, size_t c, size_t n, pyod_aggregation agg, double *out);
/* Threshold on covered timestamps, then fill uncovered ones with it. Takes
 * ownership of scores. */
pyod_status ts_fit_state(pyod_detector *d, double *scores, size_t n, double contamination);
void ts_fill(double *scores, size_t n, double value);
void znormalize(const double *x, size_t n, double *out);
/* Shape-based distance: 1 - max normalised cross-correlation, and its shift. */
double sbd(const double *x, const double *y, size_t m, long *shift);
/* DFT of any length (Bluestein for non powers of two), re/im interleaved. */
void dft(const double *in, size_t n, int inverse, double *out);

/* Graph detectors. */
struct pyod_graph_detector {
    const char *name;
    pyod_status (*fit)(pyod_graph_detector *g, const pyod_matrix *x, const size_t (*edges)[2], size_t n_edges);
    void (*destroy)(void *impl);
    void *impl;
    pyod_detector state; /* fit state holder */
    double contamination;
};
pyod_graph_detector *graph_detector_new(const char *name,
                                        pyod_status (*fit)(pyod_graph_detector *, const pyod_matrix *, const size_t (*)[2], size_t),
                                        void (*destroy)(void *), void *impl, double contamination);

#endif
