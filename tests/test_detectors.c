/* Every tabular detector trains, separates the synthetic outliers, and keeps
 * the BaseDetector contract (labels, threshold, predict, proba, confidence). */
#include "check.h"
#include "pyod.h"
#include <math.h>
#include <string.h>

typedef struct {
    const char *name;
    pyod_detector *d;
    double floor;
    int skip_flag_count; /* many tied scores (as in PyOD) */
} entry;

static pyod_detector *lscp_members(void) {
    pyod_detector *dets[3] = {pyod_knn_create(NULL), NULL, pyod_hbos_create(NULL)};
    pyod_lof_params l = pyod_lof_defaults();
    l.n_neighbors = 10;
    dets[1] = pyod_lof_create(&l);
    return pyod_lscp_create(dets, 3, NULL);
}

int main(void) {
    pyod_matrix xtr, xte;
    uint8_t ytr[300], yte[100];
    CHECK_OK(pyod_generate_data(300, 2, 0.1, 1, &xtr, ytr));
    CHECK_OK(pyod_generate_data(100, 2, 0.1, 2, &xte, yte));
    pyod_knn_params kmean = pyod_knn_defaults();
    kmean.method = PYOD_KNN_MEAN;
    pyod_gmm_params g = pyod_gmm_defaults();
    g.n_components = 2;
    pyod_lmdd_params lm = pyod_lmdd_defaults();
    lm.n_iter = 10;
    pyod_dif_params dif = pyod_dif_defaults();
    dif.n_ensemble = 10;
    pyod_abod_params ab = pyod_abod_defaults();
    ab.n_neighbors = 10;
    entry e[] = {
        {"ABOD", pyod_abod_create(&ab), 0.8, 0}, {"COPOD", pyod_copod_create(NULL), 0.8, 0},
        {"ECOD", pyod_ecod_create(NULL), 0.8, 0}, {"HBOS", pyod_hbos_create(NULL), 0.8, 0},
        {"IForest", pyod_iforest_create(NULL), 0.8, 0}, {"KNN", pyod_knn_create(NULL), 0.8, 0},
        {"KNN-mean", pyod_knn_create(&kmean), 0.8, 0}, {"LOF", pyod_lof_create(NULL), 0.8, 0},
        {"PCA", pyod_pca_create(NULL), 0.4, 0}, {"KDE", pyod_kde_create(NULL), 0.8, 0},
        {"GMM", pyod_gmm_create(&g), 0.8, 0}, {"MCD", pyod_mcd_create(NULL), 0.4, 0},
        {"OCSVM", pyod_ocsvm_create(NULL), 0.8, 0}, {"CBLOF", pyod_cblof_create(NULL), 0.8, 0},
        {"COF", pyod_cof_create(NULL), 0.8, 0}, {"INNE", pyod_inne_create(NULL), 0.8, 0},
        {"LODA", pyod_loda_create(NULL), 0.6, 1}, {"SOD", pyod_sod_create(NULL), 0.5, 0},
        {"SOS", pyod_sos_create(NULL), 0.5, 0}, {"LMDD", pyod_lmdd_create(&lm), 0.4, 0},
        {"LOCI", pyod_loci_create(NULL), 0.5, 0}, {"QMCD", pyod_qmcd_create(NULL), 0.8, 0},
        {"ROD", pyod_rod_create(NULL), 0.8, 0}, {"KPCA", pyod_kpca_create(NULL), 0.8, 1},
        {"CD", pyod_cd_create(NULL), 0.4, 0}, {"Sampling", pyod_sampling_create(NULL), 0.8, 0},
        {"HDBSCAN", pyod_hdbscan_create(NULL), 0.5, 0}, {"RGraph", pyod_rgraph_create(NULL), 0.2, 1},
        {"DIF", pyod_dif_create(&dif), 0.8, 0}, {"FeatureBagging", pyod_feature_bagging_create(NULL), 0.8, 0},
        {"SUOD", pyod_suod_create(NULL, 0, NULL), 0.8, 0}, {"LSCP", lscp_members(), 0.8, 0},
    };
    double s[100], p[100], c[100];
    uint8_t pred[100];
    for (size_t i = 0; i < sizeof e / sizeof e[0]; i++) {
        pyod_detector *d = e[i].d;
        CHECK_TRUE(d != NULL, "%s create", e[i].name);
        if (!d) continue;
        pyod_status st = pyod_fit(d, &xtr);
        CHECK_TRUE(st == PYOD_OK, "%s fit: %s", e[i].name, pyod_strerror(st));
        if (st != PYOD_OK) continue;
        size_t n;
        const double *train = pyod_decision_scores(d, &n);
        CHECK_TRUE(n == 300, "%s score count", e[i].name);
        CHECK_OK(pyod_decision_function(d, &xte, s));
        double auc = pyod_roc_auc(yte, s, 100);
        printf("%-15s train AUC %.3f  test AUC %.3f\n", e[i].name, pyod_roc_auc(ytr, train, 300), auc);
        CHECK_TRUE(auc > e[i].floor, "%s test AUC %.3f", e[i].name, auc);
        size_t flagged = 0;
        const uint8_t *lab = pyod_labels(d, NULL);
        for (size_t k = 0; k < n; k++) flagged += lab[k];
        CHECK_TRUE(e[i].skip_flag_count || (flagged >= 15 && flagged <= 45), "%s flagged %zu", e[i].name, flagged);
        CHECK_OK(pyod_predict(d, &xte, pred));
        for (size_t k = 0; k < 100; k++) CHECK_TRUE(pred[k] == (s[k] > pyod_threshold(d)), "%s predict %zu", e[i].name, k);
        CHECK_OK(pyod_predict_proba(d, &xte, PYOD_PROBA_LINEAR, p));
        CHECK_OK(pyod_predict_proba(d, &xte, PYOD_PROBA_UNIFY, c));
        for (size_t k = 0; k < 100; k++) CHECK_TRUE(p[k] >= 0 && p[k] <= 1 && c[k] >= 0 && c[k] <= 1, "%s proba", e[i].name);
        CHECK_OK(pyod_predict_confidence(d, &xte, c));
        for (size_t k = 0; k < 100; k++) CHECK_TRUE(c[k] >= 0 && c[k] <= 1, "%s confidence", e[i].name);
        pyod_free(d);
    }

    /* Error handling. */
    pyod_detector *d = pyod_ecod_create(NULL);
    CHECK_TRUE(pyod_predict(d, &xte, pred) == PYOD_ERR_NOT_FITTED, "not fitted");
    pyod_matrix empty = {NULL, 0, 0};
    CHECK_TRUE(pyod_fit(d, &empty) == PYOD_ERR_EMPTY_INPUT, "empty input");
    CHECK_OK(pyod_fit(d, &xtr));
    pyod_matrix wrong = pyod_matrix_view(xte.data, 50, 4);
    CHECK_TRUE(pyod_decision_function(d, &wrong, s) == PYOD_ERR_DIMENSION, "dimension mismatch");
    pyod_free(d);
    pyod_ecod_params bad = {0.9};
    d = pyod_ecod_create(&bad);
    CHECK_TRUE(pyod_fit(d, &xtr) == PYOD_ERR_PARAM, "bad contamination");
    pyod_free(d);

    /* MAD: fixed threshold on one feature. */
    double mv[21];
    for (int i = 0; i < 20; i++) mv[i] = 10.0 + (i % 5) * 0.1;
    mv[20] = 100.0;
    pyod_matrix mx = pyod_matrix_view(mv, 21, 1);
    d = pyod_mad_create(NULL);
    CHECK_OK(pyod_fit(d, &mx));
    CHECK_TRUE(pyod_threshold(d) == 3.5 && pyod_labels(d, NULL)[20] == 1, "MAD threshold");
    CHECK_TRUE(pyod_fit(d, &xtr) == PYOD_ERR_DIMENSION, "MAD needs one feature");
    pyod_free(d);

    /* ECOD reference values from PyOD 3.6.7. */
    double ev[10] = {1, 2, 2, 1, 3, 4, 4, 3, 10, -5};
    pyod_matrix em = pyod_matrix_view(ev, 5, 2);
    pyod_ecod_params ep = {0.2};
    d = pyod_ecod_create(&ep);
    CHECK_OK(pyod_fit(d, &em));
    const double ref[5] = {2.12026354, 1.83258146, 2.12026354, 1.83258146, 3.21887582};
    for (int i = 0; i < 5; i++) CHECK_TRUE(fabs(pyod_decision_scores(d, NULL)[i] - ref[i]) < 1e-8, "ECOD ref %d", i);
    pyod_free(d);

    /* Supervised XGBOD. */
    pyod_matrix sx, st_;
    uint8_t sy[200], sty[100];
    CHECK_OK(pyod_generate_data(200, 3, 0.1, 31, &sx, sy));
    CHECK_OK(pyod_generate_data(100, 3, 0.1, 32, &st_, sty));
    d = pyod_xgbod_create(NULL, NULL, 0, NULL, NULL);
    CHECK_TRUE(pyod_fit(d, &sx) == PYOD_ERR_UNSUPPORTED, "XGBOD needs labels");
    CHECK_OK(pyod_fit_supervised(d, &sx, sy));
    CHECK_OK(pyod_decision_function(d, &st_, s));
    CHECK_TRUE(pyod_roc_auc(sty, s, 100) > 0.9, "XGBOD AUC %.3f", pyod_roc_auc(sty, s, 100));
    pyod_free(d);

    /* Combination functions. */
    double cv[8] = {1, 3, 2, 0, 0, 0, 4, 2}, co[2];
    pyod_matrix cm = pyod_matrix_view(cv, 2, 4);
    pyod_comb_average(&cm, NULL, co);
    CHECK_TRUE(co[0] == 1.5 && co[1] == 1.5, "average");
    pyod_comb_maximization(&cm, co);
    CHECK_TRUE(co[0] == 3 && co[1] == 4, "max");
    CHECK_OK(pyod_comb_aom(&cm, 2, co));
    CHECK_TRUE(co[0] == 2.5 && co[1] == 2.0, "aom");
    CHECK_OK(pyod_comb_moa(&cm, 2, co));
    CHECK_TRUE(co[0] == 2.0 && co[1] == 3.0, "moa");

    pyod_matrix_free(&xtr), pyod_matrix_free(&xte), pyod_matrix_free(&sx), pyod_matrix_free(&st_);
    CHECK_REPORT("detectors");
}
