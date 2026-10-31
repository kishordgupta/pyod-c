/* Benchmark detectors on synthetic data, like PyOD's compare_all_models.py.
 *   make examples && ./build/examples/compare */
#include <stdio.h>
#include <time.h>
#include "pyod.h"

int main(void) {
    pyod_matrix xtr, xte;
    static uint8_t ytr[2000], yte[1000];
    pyod_generate_data(2000, 5, 0.1, 42, &xtr, ytr);
    pyod_generate_data(1000, 5, 0.1, 43, &xte, yte);
    struct { const char *name; pyod_detector *d; } dets[] = {
        {"ABOD", pyod_abod_create(NULL)},   {"COPOD", pyod_copod_create(NULL)},
        {"ECOD", pyod_ecod_create(NULL)},   {"HBOS", pyod_hbos_create(NULL)},
        {"IForest", pyod_iforest_create(NULL)}, {"KNN", pyod_knn_create(NULL)},
        {"LOF", pyod_lof_create(NULL)},     {"PCA", pyod_pca_create(NULL)},
        {"AutoEncoder", pyod_autoencoder_create(NULL)},
    };
    static double s[1000];
    printf("%-12s %8s %8s %10s\n", "model", "ROC", "P@n", "time (ms)");
    for (size_t i = 0; i < sizeof dets / sizeof dets[0]; i++) {
        clock_t t0 = clock();
        pyod_status st = pyod_fit(dets[i].d, &xtr);
        if (st == PYOD_OK) st = pyod_decision_function(dets[i].d, &xte, s);
        double ms = 1000.0 * (double)(clock() - t0) / CLOCKS_PER_SEC;
        if (st != PYOD_OK) printf("%-12s %s\n", dets[i].name, pyod_strerror(st));
        else printf("%-12s %8.4f %8.4f %10.1f\n", dets[i].name, pyod_roc_auc(yte, s, 1000), pyod_precision_n(yte, s, 1000), ms);
        pyod_free(dets[i].d);
    }
    pyod_matrix_free(&xtr);
    pyod_matrix_free(&xte);
    return 0;
}
