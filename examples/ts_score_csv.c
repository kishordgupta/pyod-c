/* Print each time-series detector's training scores for a CSV series (rows =
 * timestamps): name|scores. Used to cross-check against PyOD. */
#include "csvio.h"

static pyod_detector *knn_factory(uint64_t seed, void *ctx) {
    (void)seed, (void)ctx;
    return pyod_knn_create(NULL);
}

int main(int argc, char **argv) {
    pyod_matrix x;
    if (argc < 2 || read_csv(argv[1], &x)) {
        fprintf(stderr, "usage: ts_score_csv series.csv\n");
        return 2;
    }
    pyod_matrix_profile_params mp = pyod_matrix_profile_defaults();
    pyod_ts_od_params tp = pyod_ts_od_defaults();
    pyod_kshape_params kp = pyod_kshape_defaults();
    pyod_sand_params sp = pyod_sand_defaults();
    mp.window_size = 30;
    tp.detector = knn_factory;
    tp.window_size = 20;
    kp.window_size = 30;
    sp.window_size = 30;
    struct { const char *name; pyod_detector *d; } dets[] = {
        {"SpectralResidual", pyod_spectral_residual_create(NULL)},
        {"MatrixProfile", pyod_matrix_profile_create(&mp)},
        {"TimeSeriesOD_KNN", pyod_ts_od_create(&tp)},
        {"KShape", pyod_kshape_create(&kp)},
        {"SAND", pyod_sand_create(&sp)},
    };
    for (size_t i = 0; i < sizeof dets / sizeof dets[0]; i++) {
        pyod_status st = pyod_fit(dets[i].d, &x);
        if (st != PYOD_OK) {
            fprintf(stderr, "%s: %s\n", dets[i].name, pyod_strerror(st));
        } else {
            size_t n;
            const double *s = pyod_decision_scores(dets[i].d, &n);
            printf("%s|", dets[i].name);
            for (size_t k = 0; k < n; k++) printf("%s%.12e", k ? "," : "", s[k]);
            printf("\n");
        }
        pyod_free(dets[i].d);
    }
    free(x.data);
    return 0;
}
