/*
 * pyod-c: a C11 port of PyOD (Python Outlier Detection).
 *
 * Every tabular and time-series detector is a `pyod_detector`, mirroring
 * PyOD's BaseDetector: fit, decision_function, predict, predict_proba,
 * predict_confidence, plus the fitted decision scores, threshold and labels.
 * Larger scores mean more outlying.
 *
 * Data is a row-major `pyod_matrix` (one row per sample, or per timestamp
 * for time-series detectors). Functions return a `pyod_status`; output
 * buffers are allocated by the caller unless documented otherwise.
 */
#ifndef PYOD_H
#define PYOD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Core types                                                          */
/* ------------------------------------------------------------------ */

typedef enum {
    PYOD_OK = 0,
    PYOD_ERR_NOT_FITTED,
    PYOD_ERR_EMPTY_INPUT,
    PYOD_ERR_DIMENSION,
    PYOD_ERR_PARAM,
    /* The operation is not available for this detector (e.g. scoring new
     * data with a transductive method, or `pyod_fit` on a supervised one). */
    PYOD_ERR_UNSUPPORTED,
    PYOD_ERR_NOMEM,
    PYOD_ERR_NUMERIC
} pyod_status;

const char *pyod_strerror(pyod_status s);

/* Row-major matrix. `data` holds rows * cols doubles. */
typedef struct {
    double *data;
    size_t rows, cols;
} pyod_matrix;

#define PYOD_AT(m, i, j) ((m)->data[(size_t)(i) * (m)->cols + (size_t)(j)])

/* Wrap existing memory (not copied, not freed by pyod_matrix_free). */
pyod_matrix pyod_matrix_view(double *data, size_t rows, size_t cols);
/* Allocate a zeroed matrix; free with pyod_matrix_free. */
pyod_status pyod_matrix_alloc(pyod_matrix *m, size_t rows, size_t cols);
void pyod_matrix_free(pyod_matrix *m);

typedef struct pyod_detector pyod_detector;

/* Builds a fresh, unfitted detector from a seed. Ensembles use factories to
 * create their members, so any detector (including user-defined ones) can be
 * plugged in. `ctx` is passed through unchanged. */
typedef pyod_detector *(*pyod_factory)(uint64_t seed, void *ctx);

/* How pyod_predict_proba maps raw scores into [0, 1]. */
typedef enum { PYOD_PROBA_LINEAR = 0, PYOD_PROBA_UNIFY = 1 } pyod_proba_method;

/* ------------------------------------------------------------------ */
/* Detector interface (PyOD BaseDetector)                              */
/* ------------------------------------------------------------------ */

pyod_status pyod_fit(pyod_detector *d, const pyod_matrix *x);
/* Supervised detectors (XGBOD, DevNet): y[i] == 1 marks an outlier. */
pyod_status pyod_fit_supervised(pyod_detector *d, const pyod_matrix *x, const uint8_t *y);
/* Raw outlier scores for x; `out` has x->rows entries. */
pyod_status pyod_decision_function(const pyod_detector *d, const pyod_matrix *x, double *out);
pyod_status pyod_predict(const pyod_detector *d, const pyod_matrix *x, uint8_t *labels);
pyod_status pyod_predict_proba(const pyod_detector *d, const pyod_matrix *x, pyod_proba_method m, double *out);
/* Confidence of each prediction (Perini et al., 2020), in [0, 1]. */
pyod_status pyod_predict_confidence(const pyod_detector *d, const pyod_matrix *x, double *out);
/* Fit, then copy the training labels into `labels` (x->rows entries). */
pyod_status pyod_fit_predict(pyod_detector *d, const pyod_matrix *x, uint8_t *labels);

/* Fitted attributes; pointers stay valid until the next fit or pyod_free.
 * Return NULL (or NaN for the threshold) before fitting. */
const double *pyod_decision_scores(const pyod_detector *d, size_t *n);
const uint8_t *pyod_labels(const pyod_detector *d, size_t *n);
double pyod_threshold(const pyod_detector *d);
const char *pyod_name(const pyod_detector *d);
void pyod_free(pyod_detector *d);

/* Re-threshold a fitted detector's training scores with a custom rule (the
 * role PyThresh plays in PyOD). Writes the new threshold and labels. */
typedef double (*pyod_thresholder)(const double *scores, size_t n, void *ctx);
pyod_status pyod_apply_thresholder(const pyod_detector *d, pyod_thresholder t, void *ctx,
                                   double *threshold, uint8_t *labels);

/* ------------------------------------------------------------------ */
/* Data utilities (pyod.utils.data / utility)                          */
/* ------------------------------------------------------------------ */

/* Gaussian inliers around (+-2, ..., +-2) plus uniform outliers in [-6, 6];
 * outliers come last. x is allocated (free with pyod_matrix_free); y must
 * hold n entries. */
pyod_status pyod_generate_data(size_t n, size_t n_features, double contamination, uint64_t seed,
                               pyod_matrix *x, uint8_t *y);
/* ROC AUC via the Mann-Whitney U statistic (ties count half). */
double pyod_roc_auc(const uint8_t *y, const double *scores, size_t n);
/* Precision among the top-n scores, n = number of true outliers. */
double pyod_precision_n(const uint8_t *y, const double *scores, size_t n);

/* Score combination (pyod.models.combination). `scores` is n x k (one column
 * per detector); `out` has n entries. */
void pyod_comb_average(const pyod_matrix *scores, const double *weights, double *out);
void pyod_comb_maximization(const pyod_matrix *scores, double *out);
void pyod_comb_median(const pyod_matrix *scores, double *out);
/* Average of Maximum / Maximum of Average over `n_buckets` consecutive groups. */
pyod_status pyod_comb_aom(const pyod_matrix *scores, size_t n_buckets, double *out);
pyod_status pyod_comb_moa(const pyod_matrix *scores, size_t n_buckets, double *out);
/* Majority vote over binary labels (n x k, ties go to inlier). */
void pyod_comb_majority_vote(const uint8_t *labels, size_t n, size_t k, uint8_t *out);

/* Column standardisation (sklearn StandardScaler, zero std -> 1), in place. */
void pyod_standardize(pyod_matrix *x);

/* ------------------------------------------------------------------ */
/* Classical detectors                                                 */
/* Each has a params struct, pyod_<name>_defaults() returning PyOD's   */
/* defaults, and pyod_<name>_create() (NULL on invalid params / OOM).  */
/* ------------------------------------------------------------------ */

/* ABOD, fast variant (Kriegel et al., KDD 2008). */
typedef struct { size_t n_neighbors; double contamination; } pyod_abod_params;
pyod_abod_params pyod_abod_defaults(void);
pyod_detector *pyod_abod_create(const pyod_abod_params *p);

/* COPOD: copula-based outlier detection (Li et al., ICDM 2020). */
typedef struct { double contamination; } pyod_copod_params;
pyod_copod_params pyod_copod_defaults(void);
pyod_detector *pyod_copod_create(const pyod_copod_params *p);

/* ECOD: empirical-CDF-based outlier detection (Li et al., TKDE 2022). */
typedef struct { double contamination; } pyod_ecod_params;
pyod_ecod_params pyod_ecod_defaults(void);
pyod_detector *pyod_ecod_create(const pyod_ecod_params *p);

/* HBOS: histogram-based outlier score, fixed bin count. */
typedef struct { size_t n_bins; double alpha, tol, contamination; } pyod_hbos_params;
pyod_hbos_params pyod_hbos_defaults(void);
pyod_detector *pyod_hbos_create(const pyod_hbos_params *p);

/* Isolation Forest. Scores are 2^(-E[h]/c(psi)); PyOD's differ by a constant.
 * max_samples = 0 means min(256, n). */
typedef struct { size_t n_estimators, max_samples; double contamination; uint64_t random_state; } pyod_iforest_params;
pyod_iforest_params pyod_iforest_defaults(void);
pyod_detector *pyod_iforest_create(const pyod_iforest_params *p);

/* k nearest neighbours: distance to the k-th neighbour, or mean / median. */
typedef enum { PYOD_KNN_LARGEST = 0, PYOD_KNN_MEAN, PYOD_KNN_MEDIAN } pyod_knn_method;
typedef struct { size_t n_neighbors; pyod_knn_method method; double contamination; } pyod_knn_params;
pyod_knn_params pyod_knn_defaults(void);
pyod_detector *pyod_knn_create(const pyod_knn_params *p);

/* LOF: local outlier factor. */
typedef struct { size_t n_neighbors; double contamination; } pyod_lof_params;
pyod_lof_params pyod_lof_defaults(void);
pyod_detector *pyod_lof_create(const pyod_lof_params *p);

/* MAD: modified z-score for one feature; labels use the fixed threshold. */
typedef struct { double threshold; } pyod_mad_params;
pyod_mad_params pyod_mad_defaults(void);
pyod_detector *pyod_mad_create(const pyod_mad_params *p);

/* PCA: weighted distance to the low-variance principal axes.
 * n_components / n_selected_components = 0 means "all kept". */
typedef struct {
    size_t n_components, n_selected_components;
    int weighted, standardization;
    double contamination;
} pyod_pca_params;
pyod_pca_params pyod_pca_defaults(void);
pyod_detector *pyod_pca_create(const pyod_pca_params *p);

/* KDE: negative log kernel density (sklearn KernelDensity, Euclidean). */
typedef enum { PYOD_KERNEL_GAUSSIAN = 0, PYOD_KERNEL_TOPHAT, PYOD_KERNEL_EPANECHNIKOV,
               PYOD_KERNEL_EXPONENTIAL, PYOD_KERNEL_LINEAR } pyod_kde_kernel;
typedef struct { double bandwidth; pyod_kde_kernel kernel; double contamination; } pyod_kde_params;
pyod_kde_params pyod_kde_defaults(void);
pyod_detector *pyod_kde_create(const pyod_kde_params *p);

/* GMM: negative log likelihood under a full-covariance Gaussian mixture
 * (k-means initialisation, EM). */
typedef struct {
    size_t n_components, max_iter, n_init;
    double tol, reg_covar, contamination;
    uint64_t random_state;
} pyod_gmm_params;
pyod_gmm_params pyod_gmm_defaults(void);
pyod_detector *pyod_gmm_create(const pyod_gmm_params *p);

/* MCD: squared robust Mahalanobis distance (FastMCD, consistency correction,
 * reweighting). support_fraction <= 0 means (n + d + 1) / 2. */
typedef struct { double support_fraction, contamination; uint64_t random_state; } pyod_mcd_params;
pyod_mcd_params pyod_mcd_defaults(void);
pyod_detector *pyod_mcd_create(const pyod_mcd_params *p);

/* OCSVM: one-class SVM solved with libsvm's SMO. gamma <= 0 means 1/n_features. */
typedef enum { PYOD_SVM_LINEAR = 0, PYOD_SVM_POLY, PYOD_SVM_RBF, PYOD_SVM_SIGMOID } pyod_svm_kernel;
typedef struct {
    pyod_svm_kernel kernel;
    int degree;
    double coef0, gamma, nu, tol, contamination;
    size_t max_iter; /* 0 = max(100 n, 1e7) */
} pyod_ocsvm_params;
pyod_ocsvm_params pyod_ocsvm_defaults(void);
pyod_detector *pyod_ocsvm_create(const pyod_ocsvm_params *p);

/* CBLOF: clustering-based local outlier factor (k-means). */
typedef struct {
    size_t n_clusters;
    double alpha, beta;
    int use_weights;
    double contamination;
    uint64_t random_state;
} pyod_cblof_params;
pyod_cblof_params pyod_cblof_defaults(void);
pyod_detector *pyod_cblof_create(const pyod_cblof_params *p);

/* COF: connectivity-based outlier factor. Transductive like PyOD:
 * decision_function scores the given samples against each other. */
typedef struct { size_t n_neighbors; double contamination; } pyod_cof_params;
pyod_cof_params pyod_cof_defaults(void);
pyod_detector *pyod_cof_create(const pyod_cof_params *p);

/* INNE: isolation using nearest-neighbour ensembles. max_samples 0 = min(8, n). */
typedef struct { size_t n_estimators, max_samples; double contamination; uint64_t random_state; } pyod_inne_params;
pyod_inne_params pyod_inne_defaults(void);
pyod_detector *pyod_inne_create(const pyod_inne_params *p);

/* LODA: lightweight on-line detector of anomalies (sparse random projections). */
typedef struct { size_t n_bins, n_random_cuts; double contamination; uint64_t random_state; } pyod_loda_params;
pyod_loda_params pyod_loda_defaults(void);
pyod_detector *pyod_loda_create(const pyod_loda_params *p);

/* SOD: subspace outlier detection. Transductive like PyOD. */
typedef struct { size_t n_neighbors, ref_set; double alpha, contamination; } pyod_sod_params;
pyod_sod_params pyod_sod_defaults(void);
pyod_detector *pyod_sod_create(const pyod_sod_params *p);

/* SOS: stochastic outlier selection. Transductive like PyOD. */
typedef struct { double perplexity, eps, contamination; } pyod_sos_params;
pyod_sos_params pyod_sos_defaults(void);
pyod_detector *pyod_sos_create(const pyod_sos_params *p);

/* LMDD: linear method for deviation detection. Transductive like PyOD. */
typedef enum { PYOD_LMDD_AAD = 0, PYOD_LMDD_VAR, PYOD_LMDD_IQR } pyod_lmdd_measure;
typedef struct { size_t n_iter; pyod_lmdd_measure dis_measure; double contamination; uint64_t random_state; } pyod_lmdd_params;
pyod_lmdd_params pyod_lmdd_defaults(void);
pyod_detector *pyod_lmdd_create(const pyod_lmdd_params *p);

/* LOCI: local correlation integral, O(n^3). Transductive like PyOD. */
typedef struct { double alpha, k, contamination; } pyod_loci_params;
pyod_loci_params pyod_loci_defaults(void);
pyod_detector *pyod_loci_create(const pyod_loci_params *p);

/* QMCD: quasi-Monte Carlo discrepancy. */
typedef struct { double contamination; } pyod_qmcd_params;
pyod_qmcd_params pyod_qmcd_defaults(void);
pyod_detector *pyod_qmcd_create(const pyod_qmcd_params *p);

/* ROD: rotation-based outlier detection (3-D subspaces). */
typedef struct { double contamination; } pyod_rod_params;
pyod_rod_params pyod_rod_defaults(void);
pyod_detector *pyod_rod_create(const pyod_rod_params *p);

/* KPCA: kernel PCA reconstruction error. n_components / n_selected_components
 * 0 = all positive eigenvalues; gamma <= 0 = 1/n_features. With default
 * settings training scores are rounding noise, as in PyOD. */
typedef struct {
    size_t n_components, n_selected_components;
    pyod_svm_kernel kernel;
    int degree;
    double coef0, gamma;
    int sampling;
    size_t subset_size;
    double contamination;
    uint64_t random_state;
} pyod_kpca_params;
pyod_kpca_params pyod_kpca_defaults(void);
pyod_detector *pyod_kpca_create(const pyod_kpca_params *p);

/* CD: Cook's distance from regressing each feature on the others. */
typedef struct { double contamination; } pyod_cd_params;
pyod_cd_params pyod_cd_defaults(void);
pyod_detector *pyod_cd_create(const pyod_cd_params *p);

/* Sampling: distance to the nearest of a random training subset. */
typedef struct { size_t subset_size; double contamination; uint64_t random_state; } pyod_sampling_params;
pyod_sampling_params pyod_sampling_defaults(void);
pyod_detector *pyod_sampling_create(const pyod_sampling_params *p);

/* HDBSCAN: 1 - cluster membership probability (EOM selection). New samples
 * get the inverse-distance-weighted score of their nearest training points.
 * min_samples 0 = min_cluster_size. */
typedef struct { size_t min_cluster_size, min_samples; double contamination; } pyod_hdbscan_params;
pyod_hdbscan_params pyod_hdbscan_defaults(void);
pyod_detector *pyod_hdbscan_create(const pyod_hdbscan_params *p);

/* R-Graph: random walks on an elastic-net (lasso) self-representation graph. */
typedef struct {
    size_t transition_steps, n_nonzero, support_size, maxiter, maxiter_lasso, blocksize_test_data;
    double gamma, contamination;
    int active_support, preprocessing;
} pyod_rgraph_params;
pyod_rgraph_params pyod_rgraph_defaults(void);
pyod_detector *pyod_rgraph_create(const pyod_rgraph_params *p);

/* DIF: deep isolation forest (random untrained tanh networks + iForests).
 * hidden_neurons: up to 8 entries, terminated by 0. */
typedef struct {
    size_t representation_dim, hidden_neurons[8];
    int skip_connection;
    size_t n_ensemble, n_estimators, max_samples;
    double contamination;
    uint64_t random_state;
} pyod_dif_params;
pyod_dif_params pyod_dif_defaults(void);
pyod_detector *pyod_dif_create(const pyod_dif_params *p);

/* ------------------------------------------------------------------ */
/* Ensembles and supervised / embedding detectors                      */
/* ------------------------------------------------------------------ */

typedef enum { PYOD_COMBINE_AVERAGE = 0, PYOD_COMBINE_MAX = 1 } pyod_combination;

/* Feature Bagging (Lazarevic & Kumar, KDD 2005). factory NULL = LOF. Member
 * scores are combined without normalisation, as in PyOD. */
typedef struct {
    pyod_factory base_estimator;
    void *factory_ctx;
    size_t n_estimators;
    double max_features;
    int bootstrap_features;
    pyod_combination combination;
    double contamination;
    uint64_t random_state;
} pyod_feature_bagging_params;
pyod_feature_bagging_params pyod_feature_bagging_defaults(void);
pyod_detector *pyod_feature_bagging_create(const pyod_feature_bagging_params *p);

/* LSCP: locally selective combination of parallel outlier ensembles.
 * Takes ownership of `detectors` (array of n >= 2 detectors; the array itself
 * is copied). Scores are reproducible (PyOD redraws randomness per call). */
typedef struct {
    size_t local_region_size, n_bins;
    double local_max_features, contamination;
    uint64_t random_state;
} pyod_lscp_params;
pyod_lscp_params pyod_lscp_defaults(void);
pyod_detector *pyod_lscp_create(pyod_detector **detectors, size_t n, const pyod_lscp_params *p);

/* SUOD members: fit on a JL random projection and/or replace predictions by
 * a random-forest regressor trained on the member's training scores. */
typedef struct {
    pyod_detector *detector; /* ownership is taken */
    int project, approximate;
} pyod_suod_member;
typedef struct {
    double target_dim_frac;
    pyod_combination combination;
    double contamination;
    uint64_t random_state;
} pyod_suod_params;
pyod_suod_params pyod_suod_defaults(void);
/* members NULL / n 0 = PyOD's default bank (2 LOF, 2 HBOS, COPOD, 3 IForest). */
pyod_detector *pyod_suod_create(pyod_suod_member *members, size_t n, const pyod_suod_params *p);

/* Binary classifier interface (XGBOD's supervised model). */
typedef struct {
    void *self;
    pyod_status (*fit)(void *self, const pyod_matrix *x, const uint8_t *y);
    pyod_status (*predict_proba)(const void *self, const pyod_matrix *x, double *out);
    void (*destroy)(void *self);
} pyod_classifier;

/* XGBoost-style gradient-boosted trees (logistic loss, Newton leaves). */
typedef struct {
    size_t n_estimators, max_depth;
    double learning_rate, lambda, min_child_weight, base_score;
    uint64_t random_state;
} pyod_gbdt_params;
pyod_gbdt_params pyod_gbdt_defaults(void);
pyod_classifier pyod_gbdt_classifier(const pyod_gbdt_params *p);

/* XGBOD: unsupervised scores appended as features to a classifier. Supervised:
 * use pyod_fit_supervised. members NULL = PyOD's default bank; `standardize[i]`
 * says whether member i is fitted on standardised data. classifier NULL =
 * gradient boosting with XGBOD's defaults. Takes ownership of both. */
typedef struct { uint64_t random_state; } pyod_xgbod_params;
pyod_xgbod_params pyod_xgbod_defaults(void);
pyod_detector *pyod_xgbod_create(pyod_detector **members, const int *standardize, size_t n,
                                 const pyod_classifier *classifier, const pyod_xgbod_params *p);

/* Embedding-based detection of unstructured items (text, images, audio):
 * an encoder turns `n` opaque items into an n x dim matrix (allocate it with
 * pyod_matrix_alloc), then an ordinary detector scores the embeddings. */
typedef pyod_status (*pyod_encoder)(const void *items, size_t n, pyod_matrix *out, void *ctx);
typedef struct pyod_embedding_od pyod_embedding_od;
typedef struct {
    int standardize;   /* standardise embeddings (default 1) */
    size_t reduce_dim; /* PCA to this many dimensions; 0 = off */
} pyod_embedding_params;
pyod_embedding_params pyod_embedding_defaults(void);
/* detector NULL = KNN. Takes ownership of the detector. */
pyod_embedding_od *pyod_embedding_create(pyod_encoder enc, void *enc_ctx, pyod_detector *detector,
                                         const pyod_embedding_params *p);
pyod_status pyod_embedding_fit(pyod_embedding_od *e, const void *items, size_t n);
pyod_status pyod_embedding_decision_function(const pyod_embedding_od *e, const void *items, size_t n, double *out);
pyod_status pyod_embedding_predict(const pyod_embedding_od *e, const void *items, size_t n, uint8_t *labels);
/* The inner detector (decision scores, labels, threshold). */
const pyod_detector *pyod_embedding_detector(const pyod_embedding_od *e);
void pyod_embedding_free(pyod_embedding_od *e);

/* MultiModalOD: several embedding detectors over views of the same items;
 * scores are standardised (optionally) and combined. Takes ownership. */
typedef struct pyod_multimodal_od pyod_multimodal_od;
pyod_multimodal_od *pyod_multimodal_create(pyod_embedding_od **modalities, size_t n, pyod_combination comb,
                                           int standardize_scores, double contamination);
pyod_status pyod_multimodal_fit(pyod_multimodal_od *m, const void *items, size_t n);
pyod_status pyod_multimodal_decision_function(const pyod_multimodal_od *m, const void *items, size_t n, double *out);
pyod_status pyod_multimodal_predict(const pyod_multimodal_od *m, const void *items, size_t n, uint8_t *labels);
const double *pyod_multimodal_scores(const pyod_multimodal_od *m, size_t *n);
void pyod_multimodal_free(pyod_multimodal_od *m);

/* ------------------------------------------------------------------ */
/* Time-series detectors: x has one row per timestamp, one column per  */
/* channel. Timestamps no window covers get the threshold as score.    */
/* ------------------------------------------------------------------ */

typedef enum { PYOD_AGG_MAX = 0, PYOD_AGG_MEAN = 1 } pyod_aggregation;

/* Sliding-window wrapper around any tabular detector (factory NULL = IForest). */
typedef struct {
    pyod_factory detector;
    void *factory_ctx;
    size_t window_size, step;
    pyod_aggregation score_aggregation;
    double contamination;
} pyod_ts_od_params;
pyod_ts_od_params pyod_ts_od_defaults(void);
pyod_detector *pyod_ts_od_create(const pyod_ts_od_params *p);

/* Spectral Residual saliency (Ren et al., KDD 2019). Scores the given series
 * on its own spectrum. */
typedef struct { size_t score_window; pyod_aggregation channel_aggregation; double contamination; } pyod_spectral_residual_params;
pyod_spectral_residual_params pyod_spectral_residual_defaults(void);
pyod_detector *pyod_spectral_residual_create(const pyod_spectral_residual_params *p);

/* Matrix Profile discords (STOMP). Transductive: decision_function returns
 * PYOD_ERR_UNSUPPORTED; use the fitted decision scores. */
typedef struct { size_t window_size; pyod_aggregation channel_aggregation; double contamination; } pyod_matrix_profile_params;
pyod_matrix_profile_params pyod_matrix_profile_defaults(void);
pyod_detector *pyod_matrix_profile_create(const pyod_matrix_profile_params *p);

/* k-Shape subsequence clustering; windows far (shape-based distance) from
 * every centroid are anomalous. */
typedef struct {
    size_t n_clusters, window_size, max_iter;
    pyod_aggregation channel_aggregation;
    double contamination;
    uint64_t random_state;
} pyod_kshape_params;
pyod_kshape_params pyod_kshape_defaults(void);
pyod_detector *pyod_kshape_create(const pyod_kshape_params *p);

/* SAND: streaming subsequence anomaly detection (k-Shape centroids refreshed
 * batch by batch with exponential smoothing). */
typedef struct {
    size_t window_size, n_clusters, batch_size, max_iter;
    double alpha;
    pyod_aggregation channel_aggregation;
    double contamination;
    uint64_t random_state;
} pyod_sand_params;
pyod_sand_params pyod_sand_defaults(void);
pyod_detector *pyod_sand_create(const pyod_sand_params *p);

/* ------------------------------------------------------------------ */
/* Graph detectors (transductive, like PyOD's pyg_* models): node      */
/* features x (one row per node) plus directed edges (source, target); */
/* list both directions for an undirected graph.                        */
/* ------------------------------------------------------------------ */

typedef struct pyod_graph_detector pyod_graph_detector;

pyod_status pyod_graph_fit(pyod_graph_detector *g, const pyod_matrix *x, const size_t (*edges)[2], size_t n_edges);
const double *pyod_graph_scores(const pyod_graph_detector *g, size_t *n);
const uint8_t *pyod_graph_labels(const pyod_graph_detector *g, size_t *n);
double pyod_graph_threshold(const pyod_graph_detector *g);
const char *pyod_graph_name(const pyod_graph_detector *g);
void pyod_graph_free(pyod_graph_detector *g);

/* SCAN structural clustering: hubs and outliers score high. */
typedef struct { double epsilon; size_t mu; double contamination; } pyod_scan_params;
pyod_scan_params pyod_scan_defaults(void);
pyod_graph_detector *pyod_scan_create(const pyod_scan_params *p);

/* ANOMALOUS: joint attribute selection and residual analysis. */
typedef struct { double alpha, gamma, lambda_r; size_t max_iter; double contamination; } pyod_anomalous_params;
pyod_anomalous_params pyod_anomalous_defaults(void);
pyod_graph_detector *pyod_anomalous_create(const pyod_anomalous_params *p);

/* Radar: residual analysis in attributed networks. */
typedef struct { double alpha, gamma; size_t max_iter; double contamination; } pyod_radar_params;
pyod_radar_params pyod_radar_defaults(void);
pyod_graph_detector *pyod_radar_create(const pyod_radar_params *p);

/* ------------------------------------------------------------------ */
/* Audio features (librosa defaults as used by PyOD's AudioAE)         */
/* ------------------------------------------------------------------ */

/* Log-mel spectrogram in dB (n_fft 2048, Hann, centred, Slaney mel,
 * power_to_db with top_db 80). Allocates out (n_mels x n_frames). */
pyod_status pyod_log_mel_spectrogram(const double *y, size_t n, double sr, size_t n_mels, size_t hop_length,
                                     pyod_matrix *out);

/* ------------------------------------------------------------------ */
/* Neural detectors (hand-written autodiff, CPU, double precision)     */
/* Layer-size arrays hold up to 8 entries and end at the first 0.      */
/* ------------------------------------------------------------------ */

typedef enum {
    PYOD_ACT_RELU = 0, PYOD_ACT_LEAKY_RELU, PYOD_ACT_TANH, PYOD_ACT_SIGMOID,
    PYOD_ACT_ELU, PYOD_ACT_SOFTPLUS, PYOD_ACT_IDENTITY
} pyod_activation;

/* Shared training settings. weight_decay is PyTorch-style L2 on gradients;
 * preprocessing standardises inputs with training mean / std. */
typedef struct {
    size_t epochs, batch_size;
    double lr, weight_decay;
    int preprocessing;
    uint64_t random_state;
} pyod_train_params;

/* AutoEncoder: reconstruction error. */
typedef struct {
    pyod_train_params train;
    size_t hidden_neurons[8];
    pyod_activation hidden_activation;
    int batch_norm;
    double dropout_rate, contamination;
} pyod_autoencoder_params;
pyod_autoencoder_params pyod_autoencoder_defaults(void);
pyod_detector *pyod_autoencoder_create(const pyod_autoencoder_params *p);

/* VAE: error of a sampled reconstruction. As in PyOD the KL term is clamped to
 * [0, capacity], so the default capacity 0 trains on reconstruction only. */
typedef struct {
    pyod_train_params train;
    size_t encoder_neurons[8], decoder_neurons[8], latent_dim;
    pyod_activation hidden_activation, output_activation;
    int batch_norm;
    double dropout_rate, beta, capacity, logvar_min, logvar_max, contamination;
} pyod_vae_params;
pyod_vae_params pyod_vae_defaults(void);
pyod_detector *pyod_vae_create(const pyod_vae_params *p);

/* Deep SVDD: squared distance to a hypersphere centre in a bias-free net.
 * center NULL = initialise from the data (1 value = broadcast). */
typedef struct {
    pyod_train_params train;
    int use_ae;
    size_t hidden_neurons[8];
    pyod_activation hidden_activation, output_activation;
    double dropout_rate, contamination;
    const double *center;
    size_t center_len;
} pyod_deep_svdd_params;
pyod_deep_svdd_params pyod_deep_svdd_defaults(void);
pyod_detector *pyod_deep_svdd_create(const pyod_deep_svdd_params *p);

/* SO-GAAL / MO-GAAL (generative adversarial active learning). Scores are
 * 1 - D(x); PyOD returns D(x) (see README). */
typedef struct {
    size_t stop_epochs, k; /* k: sub-generators (MO-GAAL only) */
    double lr_d, lr_g, momentum, contamination;
    uint64_t random_state;
} pyod_gaal_params;
pyod_gaal_params pyod_gaal_defaults(void);
pyod_detector *pyod_so_gaal_create(const pyod_gaal_params *p);
pyod_detector *pyod_mo_gaal_create(const pyod_gaal_params *p);

/* ALAD: adversarially learned anomaly detection (scored with dropout off). */
typedef struct {
    pyod_activation activation_hidden_gen, activation_hidden_disc;
    int has_output_activation;
    pyod_activation output_activation;
    double dropout_rate;
    size_t latent_dim, dec_layers[8], enc_layers[8], disc_xx_layers[8], disc_zz_layers[8], disc_xz_layers[8];
    double learning_rate_gen, learning_rate_disc, lambda_recon_loss, noise_std, contamination;
    int add_recon_loss, add_disc_zz_loss, preprocessing;
    size_t epochs, batch_size;
    uint64_t random_state;
} pyod_alad_params;
pyod_alad_params pyod_alad_defaults(void);
pyod_detector *pyod_alad_create(const pyod_alad_params *p);

/* AnoGAN: per-sample latent query; new data is queried the same way (see README). */
typedef struct {
    pyod_activation activation_hidden;
    int has_output_activation;
    pyod_activation output_activation;
    double dropout_rate;
    size_t latent_dim_g, g_layers[8], d_layers[8], epochs, batch_size, epochs_query;
    double learning_rate, learning_rate_query, contamination;
    int preprocessing;
    uint64_t random_state;
} pyod_anogan_params;
pyod_anogan_params pyod_anogan_defaults(void);
pyod_detector *pyod_anogan_create(const pyod_anogan_params *p);

/* AE-1SVM: autoencoder trained jointly with a one-class SVM on random Fourier
 * features; scored by reconstruction error. */
typedef struct {
    pyod_train_params train;
    size_t hidden_neurons[8], encoding_dim, kernel_approx_features;
    pyod_activation hidden_activation;
    int batch_norm;
    double dropout_rate, alpha, sigma, contamination;
} pyod_ae1svm_params;
pyod_ae1svm_params pyod_ae1svm_defaults(void);
pyod_detector *pyod_ae1svm_create(const pyod_ae1svm_params *p);

/* DevNet: deviation network, weakly supervised (use pyod_fit_supervised).
 * network_depth 1, 2 or 4. Uses the paper's deviation loss and its balanced
 * batches (nb_batch batches per epoch, half labelled outliers); see README. */
typedef struct {
    size_t network_depth, batch_size, epochs, nb_batch;
    double confidence_margin, lr, contamination;
    uint64_t random_state;
} pyod_devnet_params;
pyod_devnet_params pyod_devnet_defaults(void);
pyod_detector *pyod_devnet_create(const pyod_devnet_params *p);

/* LUNAR: learnable aggregation of k-NN distances, trained against generated
 * negatives with validation-AUC model selection. */
typedef enum { PYOD_LUNAR_WEIGHT = 0, PYOD_LUNAR_SCORE } pyod_lunar_model;
typedef enum { PYOD_NEG_MIXED = 0, PYOD_NEG_UNIFORM, PYOD_NEG_SUBSPACE } pyod_lunar_negatives;
typedef struct {
    pyod_lunar_model model_type;
    size_t n_neighbours, n_epochs;
    pyod_lunar_negatives negative_sampling;
    double val_size, epsilon, proportion, lr, wd, contamination;
    int scale;
    uint64_t random_state;
} pyod_lunar_params;
pyod_lunar_params pyod_lunar_defaults(void);
pyod_detector *pyod_lunar_create(const pyod_lunar_params *p);

/* LSTM-AD (time series): LSTM forecasting, Mahalanobis-scored errors. */
typedef struct {
    size_t window_size, hidden_size, n_layers, epochs, batch_size;
    double lr, contamination;
    uint64_t random_state;
} pyod_lstm_ad_params;
pyod_lstm_ad_params pyod_lstm_ad_defaults(void);
pyod_detector *pyod_lstm_ad_create(const pyod_lstm_ad_params *p);

/* Anomaly Transformer (time series). */
typedef struct {
    size_t window_size, d_model, n_heads, n_layers, epochs, batch_size, step;
    double lr, lambda, dropout, contamination;
    uint64_t random_state;
} pyod_anomaly_transformer_params;
pyod_anomaly_transformer_params pyod_anomaly_transformer_defaults(void);
pyod_detector *pyod_anomaly_transformer_create(const pyod_anomaly_transformer_params *p);

/* AudioAE: AutoEncoder on log-mel windows of mono clips (already at `sr`);
 * a clip's score is the mean window reconstruction error. */
typedef struct pyod_audio_ae pyod_audio_ae;
typedef struct {
    size_t n_mels, context, hop_length, epochs, batch_size, hidden_neurons[8];
    double sr, lr, contamination;
    uint64_t random_state;
} pyod_audio_ae_params;
pyod_audio_ae_params pyod_audio_ae_defaults(void);
pyod_audio_ae *pyod_audio_ae_create(const pyod_audio_ae_params *p);
pyod_status pyod_audio_ae_fit(pyod_audio_ae *a, const double *const *clips, const size_t *lens, size_t n);
pyod_status pyod_audio_ae_decision_function(const pyod_audio_ae *a, const double *const *clips, const size_t *lens,
                                            size_t n, double *out);
const double *pyod_audio_ae_scores(const pyod_audio_ae *a, size_t *n);
double pyod_audio_ae_threshold(const pyod_audio_ae *a);
void pyod_audio_ae_free(pyod_audio_ae *a);

/* Neural graph detectors (dense GCN / GAT, transductive). */
typedef struct {
    size_t hidden_dim, num_layers, epochs;
    double dropout, lr, alpha, contamination;
    double aug_ratio;  /* CONAD */
    size_t num_heads;  /* AnomalyDAE (embed_dim = hidden_dim) */
    uint64_t random_state;
} pyod_graph_nn_params;
pyod_graph_nn_params pyod_dominant_defaults(void);
pyod_graph_nn_params pyod_anomalydae_defaults(void);
pyod_graph_nn_params pyod_cola_defaults(void);
pyod_graph_nn_params pyod_conad_defaults(void);
pyod_graph_nn_params pyod_guide_defaults(void);
pyod_graph_detector *pyod_dominant_create(const pyod_graph_nn_params *p);
pyod_graph_detector *pyod_anomalydae_create(const pyod_graph_nn_params *p);
pyod_graph_detector *pyod_cola_create(const pyod_graph_nn_params *p);
pyod_graph_detector *pyod_conad_create(const pyod_graph_nn_params *p);
pyod_graph_detector *pyod_guide_create(const pyod_graph_nn_params *p);

#ifdef __cplusplus
}
#endif

#endif /* PYOD_H */
