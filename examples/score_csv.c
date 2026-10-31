/* Print each detector's train scores, test scores, labels and threshold for
 * CSV train/test files, one line per model:
 *   name|train scores|test scores|labels|threshold
 * Used to cross-check against PyOD: score_csv train.csv test.csv [names] */
#include "csvio.h"

typedef struct {
    const char *name;
    pyod_detector *(*make)(void);
    int univariate; /* fit on the first column only */
} entry;

static pyod_detector *mk_abod(void) { return pyod_abod_create(NULL); }
static pyod_detector *mk_copod(void) { return pyod_copod_create(NULL); }
static pyod_detector *mk_ecod(void) { return pyod_ecod_create(NULL); }
static pyod_detector *mk_hbos(void) { return pyod_hbos_create(NULL); }
static pyod_detector *mk_knn(void) { return pyod_knn_create(NULL); }
static pyod_detector *mk_knn_mean(void) {
    pyod_knn_params p = pyod_knn_defaults();
    p.method = PYOD_KNN_MEAN;
    return pyod_knn_create(&p);
}
static pyod_detector *mk_lof(void) { return pyod_lof_create(NULL); }
static pyod_detector *mk_pca(void) { return pyod_pca_create(NULL); }
static pyod_detector *mk_mad(void) { return pyod_mad_create(NULL); }
static pyod_detector *mk_kde(void) { return pyod_kde_create(NULL); }
static pyod_detector *mk_gmm(void) { return pyod_gmm_create(NULL); }
static pyod_detector *mk_mcd(void) { return pyod_mcd_create(NULL); }
static pyod_detector *mk_ocsvm(void) { return pyod_ocsvm_create(NULL); }
static pyod_detector *mk_cblof(void) { return pyod_cblof_create(NULL); }
static pyod_detector *mk_cof(void) { return pyod_cof_create(NULL); }
static pyod_detector *mk_inne(void) { return pyod_inne_create(NULL); }
static pyod_detector *mk_loda(void) { return pyod_loda_create(NULL); }
static pyod_detector *mk_sod(void) { return pyod_sod_create(NULL); }
static pyod_detector *mk_sos(void) { return pyod_sos_create(NULL); }
static pyod_detector *mk_lmdd(void) { return pyod_lmdd_create(NULL); }
static pyod_detector *mk_loci(void) { return pyod_loci_create(NULL); }
static pyod_detector *mk_qmcd(void) { return pyod_qmcd_create(NULL); }
static pyod_detector *mk_rod(void) { return pyod_rod_create(NULL); }
static pyod_detector *mk_kpca(void) { return pyod_kpca_create(NULL); }
static pyod_detector *mk_cd(void) { return pyod_cd_create(NULL); }
static pyod_detector *mk_sampling(void) { return pyod_sampling_create(NULL); }
static pyod_detector *mk_hdbscan(void) { return pyod_hdbscan_create(NULL); }
static pyod_detector *mk_rgraph(void) { return pyod_rgraph_create(NULL); }
static pyod_detector *mk_dif(void) { return pyod_dif_create(NULL); }
static pyod_detector *mk_fb(void) { return pyod_feature_bagging_create(NULL); }
static pyod_detector *mk_lscp(void) {
    pyod_knn_params k = pyod_knn_defaults();
    pyod_lof_params l = pyod_lof_defaults();
    pyod_detector *dets[4];
    k.n_neighbors = 5;
    dets[0] = pyod_knn_create(&k);
    k.n_neighbors = 10;
    dets[1] = pyod_knn_create(&k);
    l.n_neighbors = 10;
    dets[2] = pyod_lof_create(&l);
    l.n_neighbors = 20;
    dets[3] = pyod_lof_create(&l);
    return pyod_lscp_create(dets, 4, NULL);
}
static pyod_detector *mk_suod(void) { return pyod_suod_create(NULL, 0, NULL); }

static const entry ENTRIES[] = {
    {"ABOD", mk_abod, 0}, {"COPOD", mk_copod, 0}, {"ECOD", mk_ecod, 0}, {"HBOS", mk_hbos, 0},
    {"KNN", mk_knn, 0},   {"KNN_mean", mk_knn_mean, 0}, {"LOF", mk_lof, 0}, {"PCA", mk_pca, 0},
    {"MAD", mk_mad, 1},
    {"KDE", mk_kde, 0}, {"GMM", mk_gmm, 0}, {"MCD", mk_mcd, 0}, {"OCSVM", mk_ocsvm, 0},
    {"CBLOF", mk_cblof, 0}, {"COF", mk_cof, 0}, {"INNE", mk_inne, 0}, {"LODA", mk_loda, 0},
    {"SOD", mk_sod, 0}, {"SOS", mk_sos, 0},
    {"LMDD", mk_lmdd, 0}, {"LOCI", mk_loci, 0}, {"QMCD", mk_qmcd, 0}, {"ROD", mk_rod, 0},
    {"KPCA", mk_kpca, 0}, {"CD", mk_cd, 0}, {"Sampling", mk_sampling, 0}, {"HDBSCAN", mk_hdbscan, 0},
    {"RGraph", mk_rgraph, 0}, {"DIF", mk_dif, 0},
    {"FeatureBagging", mk_fb, 0}, {"LSCP", mk_lscp, 0}, {"SUOD", mk_suod, 0},
};

static void print_vec(const double *v, size_t n) {
    for (size_t i = 0; i < n; i++) printf("%s%.12e", i ? "," : "", v[i]);
}

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
        fprintf(stderr, "usage: score_csv train.csv test.csv [names]\n");
        return 2;
    }
    for (size_t e = 0; e < sizeof ENTRIES / sizeof ENTRIES[0]; e++) {
        if (!wanted(ENTRIES[e].name, argc > 3 ? argv[3] : NULL)) continue;
        pyod_matrix tr = train, te = test, tr1 = {0}, te1 = {0};
        if (ENTRIES[e].univariate) {
            pyod_matrix_alloc(&tr1, train.rows, 1);
            pyod_matrix_alloc(&te1, test.rows, 1);
            for (size_t i = 0; i < train.rows; i++) tr1.data[i] = PYOD_AT(&train, i, 0);
            for (size_t i = 0; i < test.rows; i++) te1.data[i] = PYOD_AT(&test, i, 0);
            tr = tr1, te = te1;
        }
        pyod_detector *d = ENTRIES[e].make();
        pyod_status st = d ? pyod_fit(d, &tr) : PYOD_ERR_NOMEM;
        double *out = malloc(te.rows * sizeof(double));
        if (st == PYOD_OK) st = pyod_decision_function(d, &te, out);
        if (st != PYOD_OK) {
            fprintf(stderr, "%s: %s\n", ENTRIES[e].name, pyod_strerror(st));
        } else {
            size_t n;
            const double *s = pyod_decision_scores(d, &n);
            const uint8_t *lab = pyod_labels(d, NULL);
            printf("%s|", ENTRIES[e].name);
            print_vec(s, n);
            printf("|");
            print_vec(out, te.rows);
            printf("|");
            for (size_t i = 0; i < n; i++) printf("%s%d", i ? "," : "", lab[i]);
            printf("|%.12e\n", pyod_threshold(d));
        }
        free(out);
        pyod_free(d);
        pyod_matrix_free(&tr1);
        pyod_matrix_free(&te1);
    }
    free(train.data);
    free(test.data);
    return 0;
}
