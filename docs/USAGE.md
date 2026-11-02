# Using pyod-c

`pyod-c` provides a C11 detector API through `include/pyod.h` and a static library. The library requires the C standard library and `libm`; it does not require Python, NumPy, PyTorch or BLAS at runtime. The neural implementations use the port's own CPU autodiff engine and double precision.

## Build and run

With a C11 compiler and Make:

```bash
git clone https://github.com/kishordgupta/pyod-c.git
cd pyod-c
make
make examples
./build/examples/quickstart
make test
```

Compile your own application from the repository root:

```bash
cc -std=c11 -O2 -Iinclude examples/quickstart.c build/libpyod.a -lm -o build/my_detector
./build/my_detector
```

The complete [quickstart.c](../examples/quickstart.c) checks construction and every fallible detector call. It demonstrates the following pattern:

```c
#include "pyod.h"
/* x_train / x_test are pyod_matrix values created from your feature buffers. */
pyod_ecod_params params = pyod_ecod_defaults();
params.contamination = 0.1;
pyod_detector *det = pyod_ecod_create(&params);
if (!det) { /* handle invalid parameters or allocation failure */ }
/* Only continue with a valid detector, and check every returned status. */
pyod_status status = pyod_fit(det, &x_train);
if (status == PYOD_OK) status = pyod_decision_function(det, &x_test, scores);
if (status == PYOD_OK) status = pyod_predict(det, &x_test, labels);
pyod_free(det);
```

The fragment illustrates an interface; use the complete example for a standalone compilable program. Allocate `scores` as `double[x_test.rows]` and `labels` as `uint8_t[x_test.rows]` before calling these APIs. Never read output buffers after a failed call.

## Data and ownership

`pyod_matrix` contains `double *data`, `size_t rows` and `size_t cols`. Data is contiguous in row-major order; `data[i * cols + j]` is feature j of row i.

```c
double data[] = {0.0, 0.1, 0.2, 0.3, 4.0, 5.0};
pyod_matrix view = pyod_matrix_view(data, 3, 2);
```

A view borrows its backing buffer. Do not call `pyod_matrix_free` on stack arrays or other borrowed views; the free function calls `free` on its data pointer. Free only matrices allocated by `pyod_matrix_alloc`/`pyod_generate_data` or buffers whose ownership you intentionally transfer. Keep backing storage valid through each operation that uses the view.

Constructors return NULL on invalid parameters or allocation failure. Fallible detector functions return `pyod_status`; use `pyod_strerror` to report errors. A caller supplies output buffers of the required length.

| Method | Returns or writes |
|---|---|
| `pyod_fit(det, &train)` | Fits detector and stores training state |
| `pyod_decision_scores(det, &n)` / `pyod_labels(det, &n)` | Borrowed training arrays; valid until the next fit or free |
| `pyod_threshold(det)` | Scalar fitted threshold (NaN before fitting) |
| `pyod_decision_function(det, &test, scores)` | `test.rows` raw scores |
| `pyod_predict(det, &test, labels)` | `test.rows` binary labels |
| `pyod_predict_proba(det, &test, PYOD_PROBA_LINEAR or PYOD_PROBA_UNIFY, out)` | One mapped outlier score per row, unlike PyOD's two probability columns |
| `pyod_predict_confidence(det, &test, out)` | One confidence value per prediction |
| `pyod_fit_predict(det, &train, labels)` | Fits and copies training labels into caller storage |

Free a detector once with `pyod_free`. Ensemble and embedding constructors that take ownership of child detectors require freeing only the outer object. A single detector must not be used concurrently by multiple threads.

Set model parameters through the corresponding defaults struct before construction:

```c
pyod_knn_params knn = pyod_knn_defaults();
knn.n_neighbors = 10;
knn.method = PYOD_KNN_LARGEST;
pyod_detector *detector = pyod_knn_create(&knn);
/* Check detector, fit/score with status checks, then pyod_free(detector). */
```

Use `pyod_fit_supervised(det, &x, y)` for DevNet/XGBOD, with one 0/1 label per row. Graph models use `pyod_graph_fit`; embedding models use a `pyod_encoder` callback. Check the relevant declarations in [pyod.h](../include/pyod.h).

`pyod_apply_thresholder` writes a new threshold and training labels to caller outputs. It does not mutate the fitted detector. Compare new scores with the returned threshold explicitly; a later `pyod_predict` uses the original stored threshold.

## Data and preparation

Use a nonempty, rectangular matrix of finite numeric features. Rows are observations; columns must have the same meaning and order in training and scoring data. Encode categorical fields deliberately, handle missing values, and exclude identifiers that merely encode row order. Check feature units before using distance-based methods.

Fit imputers, encoders and external scalers on training data and reuse those fitted transformations on validation and test data. Do not standardize the test set independently. Some detectors, including PCA and neural models, already have their own normalization options; inspect the model settings before adding another scaler. In C, `pyod_standardize` is an in-place convenience function that does not expose fitted training statistics for reuse on a separate test matrix.

Keep labels out of an unsupervised feature matrix. Evaluate on held-out labels when available. Split time series chronologically and avoid leaking future timestamps into preprocessing.

## Reading the outputs

| Output | Meaning | Interpretation |
|---|---|---|
| Training decision scores | Scores computed during fitting | Read the stored training attribute rather than assuming a later scoring call is identical. |
| New decision scores | Raw anomaly scores; larger generally means more unusual under the detector | Score units and ranges are model-specific. Do not compare an ECOD score directly with an HBOS or PCA score. |
| Threshold | Usually the `(1 - contamination)` training-score quantile | A label is 1 when its score is strictly greater than the threshold; ties can reduce the flagged fraction. MAD uses a fixed threshold. |
| Labels | 0 = inlier, 1 = outlier | These are decisions under the chosen threshold, not confirmed faults. Test data need not have the training contamination fraction. |
| Linear probability mapping | Training min-max scaling, clipped to [0, 1] | A value of 1 can mean only that a score reaches or exceeds the training maximum. It is not a calibrated probability of a real anomaly. |
| Unify probability mapping | Gaussian/erf score transformation using training score statistics | Also a score transformation; verify calibration on labeled data before treating it as a probability. |
| Prediction confidence | Rank/binomial-based confidence associated with a threshold decision | This is separate from the probability mapping. Small samples, ties, drift and violated assumptions limit its usefulness. |

`contamination=0.1` supplies an expected training outlier fraction for thresholding. It does not teach the model the true prevalence or imply every future batch contains 10% anomalies. These ports generally require contamination in `(0, 0.5]`. With normal-only training, choose and validate a threshold using held-out normal data and an acceptable false-alarm rate; an arbitrary positive contamination can manufacture training alerts.

The quick-start uses only eight training rows to expose the API. Its confidence output is especially coarse: `floor(n * contamination)` is zero, so the confidence calculation can return zero for a flagged sample. Use a representative, substantially larger training set for evaluation.

### Batches can affect scores

ECOD and COPOD concatenate stored training data with the incoming scoring batch when estimating empirical distributions. The same row can therefore receive a different score when scored alone, in another batch, or with duplicate rows. This also affects thresholded labels, mapped probabilities and confidence. Keep batch construction consistent between validation and deployment.

COF, SOD, SOS, LOCI, LMDD and Spectral Residual score the incoming batch using its own structure. R-Graph combines incoming and training rows. Matrix Profile and graph models are transductive and should be used through their fitted-data interfaces. Do not assume every detector supports independent scoring of a single unseen record.

### Accuracy and compatibility are separate checks

For deterministic models, compare training and test scores within a stated numerical tolerance, then check thresholds and labels, including ties. For randomized or neural models, compare AUC, average precision and task-specific operating points over multiple data splits and model seeds. Repeating a fixed seed measures timing variability, not seed sensitivity. The same seed number across libraries does not guarantee the same random stream.

ROC AUC measures ranking across thresholds, while average precision summarizes the precision-recall curve and depends on class prevalence. Neither measures accuracy at the selected threshold. For an alerting system, also measure precision, recall, false positives per day and detection delay at the chosen threshold.

The ports document deliberate differences for IForest, GAAL, AnoGAN, DevNet, ALAD, LSCP, VAE and degenerate PCA cases. In particular, the IForest score convention has a constant offset relative to PyOD for an otherwise identical forest; independently trained forests also differ through randomness. Matching AUC alone does not establish identical scores, calibrated probabilities or deployment performance. See the README for the model-specific differences.

## Choosing a method and interface

Start with a simple baseline such as ECOD, COPOD or HBOS, then evaluate alternatives on representative data. Use KNN/LOF when neighborhood structure is appropriate, accounting for the brute-force neighbor search in the ports. Use IForest when its sampling and tree assumptions fit the problem. Choose neural models for demonstrated validation benefit and budget for their training and backend costs.

Time-series methods expect one row per timestamp and one column per channel. Graph methods use a separate graph interface and score the fitted graph. Embedding methods require a user-supplied encoder; pretrained text/image encoders and audio file decoding are not bundled. DevNet and XGBOD require labels through their supervised fitting methods.

Large neighbor workloads and dense graph adjacency can consume substantial time or memory. The reported benchmark uses 10 features and at most 20,000 training rows; it does not establish million-row or sparse-graph scalability.

## CSV utilities and performance tests

Examples accept numeric comma-separated features with no header or labels. For a selected model list:

```bash
make examples
./build/examples/score_csv train.csv test.csv ECOD,HBOS,KNN
./build/examples/compare
```

`score_csv` prints `model|training_scores|test_scores|training_labels|threshold`. These small example readers are intended for clean numeric fixtures, not arbitrary CSV ingestion. The `deep_score_csv`, `ts_score_csv` and `graph_score_csv` programs have distinct interfaces documented in their source.

The `compare` example is a native demonstration on synthetic data, not the full three-library README benchmark. Its historical timer is CPU time. Follow [BENCHMARKING.md](BENCHMARKING.md) for a shared-input, monotonic wall-time comparison and the limits of speedup claims. The new `benchmark_csv` timer uses POSIX `clock_gettime` on macOS/Linux; this example requirement is separate from the portable C11 library.
