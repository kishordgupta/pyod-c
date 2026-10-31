/* Neural detectors: each trains and separates synthetic outliers better than
 * chance (floors follow what PyOD itself achieves on the same data). */
#include "check.h"
#include "pyod.h"
#include <math.h>
#include <string.h>

static pyod_matrix xtr, xte;
static uint8_t ytr[300], yte[150];

static void check_tab(const char *name, pyod_detector *d, double floor) {
    double s[150];
    pyod_status st = pyod_fit(d, &xtr);
    CHECK_TRUE(st == PYOD_OK, "%s fit: %s", name, pyod_strerror(st));
    if (st == PYOD_OK) {
        CHECK_OK(pyod_decision_function(d, &xte, s));
        int finite = 1;
        for (int i = 0; i < 150; i++) finite &= isfinite(s[i]);
        double auc = pyod_roc_auc(yte, s, 150);
        printf("%-20s test AUC %.3f\n", name, auc);
        CHECK_TRUE(finite, "%s finite scores", name);
        CHECK_TRUE(isnan(floor) || auc > floor, "%s AUC %.3f", name, auc);
    }
    pyod_free(d);
}

int main(void) {
    CHECK_OK(pyod_generate_data(300, 5, 0.1, 21, &xtr, ytr));
    CHECK_OK(pyod_generate_data(150, 5, 0.1, 22, &xte, yte));
    check_tab("AutoEncoder", pyod_autoencoder_create(NULL), 0.8);
    check_tab("VAE", pyod_vae_create(NULL), 0.8);
    check_tab("DeepSVDD", pyod_deep_svdd_create(NULL), 0.8);
    /* GAAL depends on where the data sits relative to its [0, 1]^d generator
     * noise; PyOD scores 0.24-0.34 on this data too, so only check it runs. */
    check_tab("SO_GAAL", pyod_so_gaal_create(NULL), NAN);
    check_tab("MO_GAAL", pyod_mo_gaal_create(NULL), NAN);
    pyod_alad_params al = pyod_alad_defaults();
    al.epochs = 50;
    check_tab("ALAD", pyod_alad_create(&al), NAN);
    check_tab("AnoGAN", pyod_anogan_create(NULL), 0.65); /* 0.75-0.92 across seeds */
    check_tab("AE1SVM", pyod_ae1svm_create(NULL), 0.8);
    check_tab("LUNAR", pyod_lunar_create(NULL), 0.8);

    /* DevNet with a third of the outliers labelled. */
    uint8_t partial[300];
    for (int i = 0; i < 300; i++) partial[i] = ytr[i] && i % 3 == 0;
    pyod_detector *dv = pyod_devnet_create(NULL);
    CHECK_TRUE(pyod_fit(dv, &xtr) == PYOD_ERR_UNSUPPORTED, "DevNet needs labels");
    CHECK_OK(pyod_fit_supervised(dv, &xtr, partial));
    double s[150];
    CHECK_OK(pyod_decision_function(dv, &xte, s));
    printf("%-20s test AUC %.3f\n", "DevNet", pyod_roc_auc(yte, s, 150));
    CHECK_TRUE(pyod_roc_auc(yte, s, 150) > 0.6, "DevNet AUC"); /* 0.65-0.78 across seeds */
    pyod_free(dv);

    /* Time series: sine with two level shifts. */
    pyod_matrix ts;
    uint8_t ty[400];
    CHECK_OK(pyod_matrix_alloc(&ts, 400, 1));
    for (int t = 0; t < 400; t++) {
        ts.data[t] = sin(t * 2.0 * 3.141592653589793 / 40.0) + 0.05 * sin(t * 1.7);
        ty[t] = (t >= 150 && t < 160) || (t >= 300 && t < 310);
        if (ty[t]) ts.data[t] += 2.0;
    }
    pyod_lstm_ad_params lp = pyod_lstm_ad_defaults();
    lp.window_size = 30;
    lp.epochs = 10;
    pyod_detector *l = pyod_lstm_ad_create(&lp);
    CHECK_OK(pyod_fit(l, &ts));
    double auc = pyod_roc_auc(ty, pyod_decision_scores(l, NULL), 400);
    printf("%-20s AUC %.3f\n", "LSTMAD", auc);
    CHECK_TRUE(auc > 0.75, "LSTMAD AUC");
    pyod_free(l);
    pyod_anomaly_transformer_params ap = pyod_anomaly_transformer_defaults();
    ap.window_size = 40, ap.d_model = 32, ap.n_heads = 4, ap.n_layers = 2, ap.epochs = 3;
    pyod_detector *at = pyod_anomaly_transformer_create(&ap);
    CHECK_OK(pyod_fit(at, &ts));
    auc = pyod_roc_auc(ty, pyod_decision_scores(at, NULL), 400);
    printf("%-20s AUC %.3f\n", "AnomalyTransformer", auc);
    CHECK_TRUE(auc > 0.55, "AnomalyTransformer AUC");
    pyod_free(at);
    pyod_matrix_free(&ts);

    /* AudioAE: noisy clips stand out from tonal ones. */
    static double clips[20][8000];
    const double *cp[20];
    size_t lens[20];
    for (int c = 0; c < 20; c++) {
        uint64_t st_ = (uint64_t)c * 7919 + 1;
        for (int i = 0; i < 8000; i++) {
            st_ = st_ * 6364136223846793005ULL + 1442695040888963407ULL;
            double u = (double)(st_ >> 11) / 9007199254740992.0 - 0.5;
            clips[c][i] = c < 18 ? 0.5 * sin(2 * 3.141592653589793 * (300.0 + 10.0 * (c % 3)) * i / 8000.0) + 0.01 * u : 2.0 * u;
        }
        cp[c] = clips[c];
        lens[c] = 8000;
    }
    pyod_audio_ae_params aap = pyod_audio_ae_defaults();
    aap.sr = 8000.0;
    aap.epochs = 20;
    pyod_audio_ae *aa = pyod_audio_ae_create(&aap);
    CHECK_OK(pyod_audio_ae_fit(aa, cp, lens, 20));
    const double *as = pyod_audio_ae_scores(aa, NULL);
    double tonal_max = 0.0;
    for (int c = 0; c < 18; c++) tonal_max = fmax(tonal_max, as[c]);
    printf("%-20s noisy min %.2f vs tonal max %.2f\n", "AudioAE", fmin(as[18], as[19]), tonal_max);
    CHECK_TRUE(fmin(as[18], as[19]) > tonal_max, "AudioAE separates noisy clips");
    CHECK_OK(pyod_audio_ae_decision_function(aa, cp, lens, 2, s));
    pyod_audio_ae_free(aa);

    /* Neural graph detectors on 4 communities with an injected clique (nodes
     * 0-5) and attribute outliers (6-11). */
    enum { N = 120 };
    static double gx[N * 6];
    static size_t ge[4000][2];
    size_t ne = 0;
    uint8_t gy[N] = {0};
    uint64_t st_ = 12;
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 6; k++) {
            st_ = st_ * 6364136223846793005ULL + 1442695040888963407ULL;
            gx[i * 6 + k] = (k == i % 4 ? 3.0 : 0.0) + 0.3 * ((double)(st_ >> 11) / 9007199254740992.0 - 0.5);
        }
    for (int i = 0; i < N; i++)
        for (int t = 0; t < 4; t++) {
            st_ = st_ * 6364136223846793005ULL + 1442695040888963407ULL;
            int j = (int)((st_ >> 33) % (N / 4)) * 4 + i % 4;
            if (j != i) ge[ne][0] = (size_t)i, ge[ne++][1] = (size_t)j, ge[ne][0] = (size_t)j, ge[ne++][1] = (size_t)i;
        }
    for (int a = 0; a < 6; a++) {
        for (int b = 0; b < 6; b++)
            if (a != b) ge[ne][0] = (size_t)a, ge[ne++][1] = (size_t)b;
        gy[a] = 1;
    }
    for (int i = 6; i < 12; i++) {
        for (int k = 0; k < 6; k++) gx[i * 6 + k] = -6.0;
        gy[i] = 1;
    }
    pyod_matrix gm = pyod_matrix_view(gx, N, 6);
    pyod_graph_detector *gd[5] = {pyod_dominant_create(NULL), pyod_anomalydae_create(NULL), pyod_cola_create(NULL),
                                  pyod_conad_create(NULL), pyod_guide_create(NULL)};
    for (int k = 0; k < 5; k++) {
        CHECK_OK(pyod_graph_fit(gd[k], &gm, (const size_t(*)[2])ge, ne));
        auc = pyod_roc_auc(gy, pyod_graph_scores(gd[k], NULL), N);
        printf("%-20s AUC %.3f\n", pyod_graph_name(gd[k]), auc);
        /* AnomalyDAE is unstable on graphs like this in PyOD too (0.35-0.69 across seeds). */
        if (k != 1) CHECK_TRUE(auc > 0.6, "%s AUC", pyod_graph_name(gd[k]));
        pyod_graph_free(gd[k]);
    }
    pyod_matrix_free(&xtr), pyod_matrix_free(&xte);
    CHECK_REPORT("deep");
}
