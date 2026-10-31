# pyod-c

A C11 port of [PyOD](https://github.com/yzhao062/pyod), covering every
detector in `pyod/models`, including the neural ones.

- **No dependencies** beyond the C standard library and `libm`.
- **Neural detectors** run on a small reverse-mode autodiff engine written for
  this port (`src/deep/ad.c`): CPU, double precision, with PyTorch's layer
  initialisation and Adam / SGD / RMSprop semantics.
- One header, `include/pyod.h`, and one static library, `build/libpyod.a`.

```bash
make            # builds build/libpyod.a
make test       # builds and runs the test suite
make examples   # builds build/examples/*
```

```c
#include "pyod.h"

pyod_ecod_params p = pyod_ecod_defaults();
p.contamination = 0.1;
pyod_detector *det = pyod_ecod_create(&p);

pyod_status st = pyod_fit(det, &x_train);   /* pyod_matrix: row-major, one row per sample */
if (st != PYOD_OK) fprintf(stderr, "%s\n", pyod_strerror(st));

size_t n;
const uint8_t *train_labels = pyod_labels(det, &n);       /* labels_ */
pyod_decision_function(det, &x_test, scores);             /* caller-allocated, x_test.rows entries */
pyod_predict(det, &x_test, labels);
pyod_predict_proba(det, &x_test, PYOD_PROBA_UNIFY, probs);
pyod_predict_confidence(det, &x_test, conf);
pyod_free(det);
```

Link with `-Iinclude build/libpyod.a -lm`. `examples/quickstart.c` is a complete
program; `examples/compare.c` benchmarks several detectors like PyOD's
`compare_all_models.py`.

## Interfaces

| PyOD | C |
|---|---|
| `BaseDetector` | `pyod_detector`: `pyod_fit`, `pyod_decision_function`, `pyod_predict`, `pyod_fit_predict`, `pyod_predict_proba` (linear/unify), `pyod_predict_confidence`, and the fitted `pyod_decision_scores` / `pyod_threshold` / `pyod_labels` |
| model constructors and keyword arguments | `pyod_<model>_params` struct, `pyod_<model>_defaults()` with PyOD's defaults, `pyod_<model>_create(&params)` (`NULL` = defaults) |
| supervised `fit(X, y)` (XGBOD, DevNet) | `pyod_fit_supervised(det, x, y)`; plain `pyod_fit` returns `PYOD_ERR_UNSUPPORTED` |
| time-series detectors | a `pyod_detector` on a matrix with one row per timestamp and one column per channel |
| `pyg_*` graph detectors | `pyod_graph_detector`: `pyod_graph_fit(g, node_features, edges, n_edges)`; transductive |
| `EmbeddingOD`, `MultiModalOD` | `pyod_embedding_od` / `pyod_multimodal_od` over opaque items, with a `pyod_encoder` callback you write |
| ensembles' base estimators | `pyod_factory` callbacks (`seed, ctx -> pyod_detector *`), or ready detectors whose ownership the ensemble takes |
| XGBOD's classifier | `pyod_classifier` (fit / predict_proba / destroy); `pyod_gbdt_classifier` is the built-in gradient-boosted trees |
| PyThresh `contamination` objects | `pyod_thresholder` callback + `pyod_apply_thresholder` |
| `combination` | `pyod_comb_average`, `_maximization`, `_median`, `_aom`, `_moa`, `_majority_vote` |
| `utils.data` / `utility` | `pyod_generate_data`, `pyod_roc_auc`, `pyod_precision_n`, `pyod_standardize` |

Conventions:
- Every fallible function returns a `pyod_status`; `pyod_strerror` describes it.
- Output buffers are allocated by the caller unless a comment in `pyod.h` says otherwise.
- Pointers returned by `pyod_decision_scores` and `pyod_labels` stay valid until the next fit or `pyod_free`.
- Constructors that take other detectors (LSCP, SUOD, XGBOD, the embedding models) take ownership of them, so free only the outer object.

## Models and how they compare with PyOD 3.6.7

Each model was run through PyOD and the port on the same data.
- **Deterministic models** are compared score by score.
- **Models that use randomness** (sampling, random init, SGD) can't match PyOD
  number for number, so they are compared by ROC AUC, and where the gap
  looked large, by AUC spread across several seeds on both sides.

### Same scores as PyOD (relative error ≤ 1e-12 unless noted)

ABOD, CD, COF, COPOD, ECOD, GMM (1 component), HBOS, HDBSCAN, KNN, LOCI,
LOF, MAD, PCA, QMCD, ROD, SOD, SOS, KDE (3e-7 on new data), KPCA (test
scores, 3e-6; see below), OCSVM (agrees to the SMO solver tolerance),
R-Graph (2e-3, lasso solver tolerance), Spectral Residual, Matrix Profile,
TimeSeriesOD (+KNN), SCAN, ANOMALOUS (1e-8), Radar (1e-7). The log-mel
features behind AudioAE match librosa 0.11 to 2.3e-7 dB.

### Randomised: same algorithm, comparable AUC

| Group | Models |
|---|---|
| Classical | IForest, INNE, LODA, LMDD, CBLOF, Sampling, MCD, DIF |
| Ensembles | FeatureBagging, LSCP, SUOD, XGBOD |
| Time series | k-Shape, SAND, LSTM-AD, Anomaly Transformer |
| Neural | AutoEncoder, VAE, DeepSVDD, AE1SVM, LUNAR, ALAD, SO-GAAL, MO-GAAL, AnoGAN, DevNet, AudioAE |
| Graph neural | DOMINANT, AnomalyDAE, CoLA, CONAD, GUIDE |

Examples from the cross-checks (port vs PyOD, test ROC AUC):
- **Neural:** AutoEncoder 1.00 vs 1.00, VAE 1.00 vs 1.00, DeepSVDD .983 vs .981, AnoGAN .998, AE1SVM 1.00, LUNAR 1.00.
- **GAAL:** across seeds, SO-GAAL .62–1.00 vs .75–1.00 and MO-GAAL .63–1.00 vs .70–1.00 (PyOD's score inverted; see below).
- **Neural time series:** LSTM-AD .86 vs .89. Anomaly Transformer .63–.70 across seeds vs .66–.74, slightly lower on average.
- **Graph neural models** fall inside PyOD's own seed-to-seed range.

MCD, CBLOF, INNE, LODA, LMDD, Sampling and FeatureBagging draw from the same
generator as [pyod-rs](https://github.com/kishordgupta/pyod-rs), so the two ports give
identical scores for the same seed.

### Where the port deliberately differs

- **IForest** returns the paper's score `2^(-E[h]/c(ψ))`. PyOD's differs from it by a constant offset.
- **SO-/MO-GAAL** score `1 − D(x)`; PyOD returns `D(x)`. GAAL is sensitive to data scale, and on some datasets the ranking flips either way.
- **AnoGAN** runs the per-sample latent query on new data too. PyOD reuses the query state left by the last training sample, which makes its test scores distances to that one sample.
- **DevNet** uses the deviation loss from the paper; PyOD calls it with its arguments swapped. When labelled outliers are present, each training batch alternates inliers and labelled outliers, as in the paper (`nb_batch` batches per epoch). Plain shuffled batches collapsed on some seeds.
- **ALAD** scores with dropout off, so scores are deterministic. PyOD leaves its networks in training mode while scoring.
- **LSCP and VAE** give the same scores when called twice on the same data. PyOD draws new randomness on each call.
- **PCA** floors zero explained-variance weights at machine epsilon, where PyOD would divide by zero.

### Kept as in PyOD, but worth knowing

- **Transductive methods** score a `pyod_decision_function` batch against itself, not the training data: COF, SOD, SOS, LOCI, LMDD and Spectral Residual. R-Graph scores it together with the training data. Matrix Profile and the graph models score only the fitted data.
- **KPCA** keeps every component by default, so training scores are rounding noise in both libraries. Set `n_components`.
- **The VAE's default `capacity = 0`** clamps the KL term to zero.
- **GUIDE** rejects triangle-free graphs.

## Not ported

- `so_gaal_new.py`: an unexported, unfinished refactor of SO-GAAL.
- PyThresh's ~35 threshold methods (`thresholds.py` wraps an external library). Plug your own in through `pyod_thresholder`.
- File decoding and resampling for AudioAE: pass mono waveforms already at `sr`.
- Ready-made text and image encoders for the embedding models (sentence-transformers, OpenAI, DINOv2): write a `pyod_encoder`.

## Practical notes

- Everything runs in double precision on one CPU core.
- Neighbour searches compare every pair of points, and the graph models use dense
  adjacency, so these models are best for up to tens of thousands of samples or a few
  thousand nodes.
- Each detector draws from its own generator seeded by `random_state`, so results
  are reproducible. The autodiff tape is thread-local, so different detectors can be
  fitted on different threads at once; a single detector must not be used from two
  threads at the same time.

## Layout

| Path | Contents |
|---|---|
| `include/pyod.h` | the public API |
| `src/base.c`, `util.c`, `linalg.c`, `trees.c` | detector plumbing, statistics, linear algebra, trees |
| `src/models/` | classical detectors and ensembles |
| `src/ts/`, `src/graph/` | time-series and graph detectors |
| `src/deep/` | autodiff (`ad.c`), layers (`nn.c`) and the neural detectors |
| `tests/` | unit tests, finite-difference gradient checks, per-detector AUC checks |
| `examples/` | quick start, benchmark, and the CSV scorers the PyOD cross-checks used |

`examples/score_csv.c`, `deep_score_csv.c`, `ts_score_csv.c` and
`graph_score_csv.c` print scores for CSV input.

The test suite also passes under AddressSanitizer and UndefinedBehaviorSanitizer:

```bash
make clean && make test CFLAGS="-O1 -g -fsanitize=address,undefined" LDFLAGS="-fsanitize=address,undefined"
```
