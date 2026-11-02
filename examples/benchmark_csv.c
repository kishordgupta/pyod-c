/* Matched-input wall-time benchmark helper; see docs/BENCHMARKING.md.
 * The timer uses CLOCK_MONOTONIC on POSIX (macOS/Linux). */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <time.h>
#include "csvio.h"

static double now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) { perror("clock_gettime"); exit(1); }
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static pyod_detector *create(const char *name) {
    if (!strcmp(name, "ABOD")) return pyod_abod_create(NULL);
    if (!strcmp(name, "COPOD")) return pyod_copod_create(NULL);
    if (!strcmp(name, "ECOD")) return pyod_ecod_create(NULL);
    if (!strcmp(name, "HBOS")) return pyod_hbos_create(NULL);
    if (!strcmp(name, "IForest")) return pyod_iforest_create(NULL);
    if (!strcmp(name, "KNN")) return pyod_knn_create(NULL);
    if (!strcmp(name, "LOF")) return pyod_lof_create(NULL);
    if (!strcmp(name, "PCA")) return pyod_pca_create(NULL);
    return NULL;
}

static int count_arg(const char *text, int allow_zero) {
    char *end;
    long value = strtol(text, &end, 10);
    return (*text && !*end && value >= (allow_zero ? 0 : 1) && value <= 10000) ? (int)value : -1;
}

int main(int argc, char **argv) {
    pyod_matrix train = {0}, test = {0};
    if (argc != 6) { fprintf(stderr, "usage: benchmark_csv MODEL train.csv test.csv repeats warmup\n"); return 2; }
    int repeats = count_arg(argv[4], 0), warmup = count_arg(argv[5], 1);
    if (repeats < 0 || warmup < 0 || read_csv(argv[2], &train) || read_csv(argv[3], &test)) {
        fprintf(stderr, "invalid counts or CSV input\n");
        free(train.data); free(test.data); return 2;
    }
    int result = 0;
    for (int iteration = 0; iteration < warmup + repeats; iteration++) {
        pyod_detector *det = create(argv[1]);
        if (!det) { fprintf(stderr, "unknown model or allocation failure: %s\n", argv[1]); result = 1; break; }
        double start = now();
        pyod_status st = pyod_fit(det, &train);
        double fit = now() - start;
        start = now();
        /* Include caller output-buffer allocation, as Rust/Python allocate their outputs. */
        double *scores = malloc(test.rows * sizeof(double));
        if (!scores) st = PYOD_ERR_NOMEM;
        if (st == PYOD_OK) st = pyod_decision_function(det, &test, scores);
        double score = now() - start;
        if (st == PYOD_OK)
            for (size_t i = 0; i < test.rows; i++) if (!isfinite(scores[i])) { st = PYOD_ERR_NUMERIC; break; }
        if (st != PYOD_OK) {
            fprintf(stderr, "%s\n", pyod_strerror(st)); result = 1;
        } else if (iteration >= warmup) {
            printf("{\"fit_seconds\":%.17g,\"score_seconds\":%.17g,\"scores\":[", fit, score);
            for (size_t i = 0; i < test.rows; i++) printf("%s%.17g", i ? "," : "", scores[i]);
            puts("]}");
        }
        free(scores); pyod_free(det);
        if (result) break;
    }
    free(train.data); free(test.data);
    return result;
}
