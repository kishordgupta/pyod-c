/* Print each neural detector's test scores for CSV train/test files:
 * name|scores. Used to compare against PyOD: deep_score_csv train test [names] */
#include "csvio.h"

typedef struct {
    const char *name;
    pyod_detector *(*make)(void);
} entry;

static pyod_detector *mk_ae(void) { return pyod_autoencoder_create(NULL); }
static pyod_detector *mk_vae(void) { return pyod_vae_create(NULL); }
static pyod_detector *mk_svdd(void) { return pyod_deep_svdd_create(NULL); }
static pyod_detector *mk_so(void) { return pyod_so_gaal_create(NULL); }
static pyod_detector *mk_mo(void) { return pyod_mo_gaal_create(NULL); }
static pyod_detector *mk_alad(void) { return pyod_alad_create(NULL); }
static pyod_detector *mk_anogan(void) { return pyod_anogan_create(NULL); }
static pyod_detector *mk_ae1svm(void) { return pyod_ae1svm_create(NULL); }
static pyod_detector *mk_lunar(void) { return pyod_lunar_create(NULL); }
static pyod_detector *mk_dif(void) { return pyod_dif_create(NULL); }

static const entry ENTRIES[] = {
    {"AutoEncoder", mk_ae}, {"VAE", mk_vae}, {"DeepSVDD", mk_svdd},
    {"SO_GAAL", mk_so}, {"MO_GAAL", mk_mo}, {"ALAD", mk_alad}, {"AnoGAN", mk_anogan},
    {"AE1SVM", mk_ae1svm}, {"LUNAR", mk_lunar}, {"DIF", mk_dif},
};

static int wanted(const char *name, const char *list) {
    if (!list) return 1;
    size_t len = strlen(name);
    for (const char *p = list; (p = strstr(p, name)); p += len)
        if ((p == list || p[-1] == ',') && (p[len] == ',' || p[len] == '\0')) return 1;
    return 0;
}

int main(int argc, char **argv) {
    pyod_matrix train, test;
    if (argc < 3 || read_csv(argv[1], &train) || read_csv(argv[2], &test)) {
        fprintf(stderr, "usage: deep_score_csv train.csv test.csv [names]\n");
        return 2;
    }
    double *out = malloc(test.rows * sizeof(double));
    for (size_t e = 0; e < sizeof ENTRIES / sizeof ENTRIES[0]; e++) {
        if (!wanted(ENTRIES[e].name, argc > 3 ? argv[3] : NULL)) continue;
        pyod_detector *d = ENTRIES[e].make();
        pyod_status st = d ? pyod_fit(d, &train) : PYOD_ERR_NOMEM;
        if (st == PYOD_OK) st = pyod_decision_function(d, &test, out);
        if (st != PYOD_OK) {
            fprintf(stderr, "%s: %s\n", ENTRIES[e].name, pyod_strerror(st));
        } else {
            printf("%s|", ENTRIES[e].name);
            for (size_t i = 0; i < test.rows; i++) printf("%s%.9e", i ? "," : "", out[i]);
            printf("\n");
        }
        pyod_free(d);
    }
    free(out), free(train.data), free(test.data);
    return 0;
}
