/* Time-series detectors, graph detectors, embedding detectors and audio features. */
#include "check.h"
#include "pyod.h"
#include <math.h>
#include <string.h>

static pyod_detector *knn_factory(uint64_t seed, void *ctx) {
    (void)seed, (void)ctx;
    return pyod_knn_create(NULL);
}

/* Sine series with three injected ramps on channel 0. */
static void sine_series(size_t n, size_t ch, pyod_matrix *x, uint8_t *y) {
    pyod_matrix_alloc(x, n, ch);
    uint64_t s = 5;
    for (size_t t = 0; t < n; t++) {
        y[t] = 0;
        for (size_t c = 0; c < ch; c++) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            double noise = ((double)(s >> 11) / 9007199254740992.0 - 0.5) * 0.3;
            PYOD_AT(x, t, c) = sin((double)t * 2.0 * 3.141592653589793 / 40.0 + (double)c) + noise;
        }
    }
    size_t starts[3] = {150, 330, 480};
    for (int k = 0; k < 3; k++)
        for (size_t j = 0; j < 12; j++) PYOD_AT(x, starts[k] + j, 0) += 2.5 * (double)j / 11.0, y[starts[k] + j] = 1;
}

/* Encoder for the embedding test: (length, digit count, space count) of a string. */
static pyod_status count_encoder(const void *items, size_t n, pyod_matrix *out, void *ctx) {
    (void)ctx;
    const char *const *s = items;
    CHECK_OK(pyod_matrix_alloc(out, n, 3));
    for (size_t i = 0; i < n; i++) {
        double digits = 0, spaces = 0;
        for (const char *c = s[i]; *c; c++) digits += *c >= '0' && *c <= '9', spaces += *c == ' ';
        PYOD_AT(out, i, 0) = (double)strlen(s[i]), PYOD_AT(out, i, 1) = digits, PYOD_AT(out, i, 2) = spaces;
    }
    return PYOD_OK;
}

int main(void) {
    for (size_t ch = 1; ch <= 2; ch++) {
        pyod_matrix x;
        uint8_t y[600];
        sine_series(600, ch, &x, y);
        pyod_matrix_profile_params mp = pyod_matrix_profile_defaults();
        pyod_ts_od_params tp = pyod_ts_od_defaults();
        pyod_kshape_params kp = pyod_kshape_defaults();
        pyod_sand_params sp = pyod_sand_defaults();
        mp.window_size = kp.window_size = sp.window_size = 30;
        tp.detector = knn_factory;
        tp.window_size = 20;
        struct { const char *name; pyod_detector *d; double floor; } e[] = {
            {"SpectralResidual", pyod_spectral_residual_create(NULL), 0.6},
            {"MatrixProfile", pyod_matrix_profile_create(&mp), 0.7},
            {"TimeSeriesOD", pyod_ts_od_create(&tp), 0.8},
            {"KShape", pyod_kshape_create(&kp), 0.5},
            {"SAND", pyod_sand_create(&sp), 0.4},
        };
        double out[600];
        for (size_t i = 0; i < 5; i++) {
            CHECK_OK(pyod_fit(e[i].d, &x));
            size_t n;
            double auc = pyod_roc_auc(y, pyod_decision_scores(e[i].d, &n), 600);
            printf("%-16s channels=%zu AUC %.3f\n", e[i].name, ch, auc);
            CHECK_TRUE(n == 600 && auc > e[i].floor, "%s AUC %.3f", e[i].name, auc);
            pyod_matrix part = pyod_matrix_view(x.data, 200, ch);
            pyod_status st = pyod_decision_function(e[i].d, &part, out);
            CHECK_TRUE(i == 1 ? st == PYOD_ERR_UNSUPPORTED : st == PYOD_OK, "%s decision_function %s", e[i].name, pyod_strerror(st));
            pyod_free(e[i].d);
        }
        pyod_matrix_free(&x);
    }

    /* Graph: two dense communities, a bridging node and a pendant node. */
    size_t edges[200][2], ne = 0;
    for (size_t c = 0; c < 2; c++)
        for (size_t i = 0; i < 10; i++)
            for (size_t j = 0; j < 10; j++)
                if (i != j) edges[ne][0] = c * 10 + i, edges[ne++][1] = c * 10 + j;
    size_t extra[6][2] = {{20, 0}, {0, 20}, {20, 10}, {10, 20}, {21, 5}, {5, 21}};
    for (int k = 0; k < 6; k++) edges[ne][0] = extra[k][0], edges[ne++][1] = extra[k][1];
    double gx[22 * 2];
    for (int i = 0; i < 22; i++) gx[2 * i] = i / 10, gx[2 * i + 1] = 1.0;
    pyod_matrix gm = pyod_matrix_view(gx, 22, 2);
    pyod_graph_detector *scan = pyod_scan_create(NULL);
    CHECK_OK(pyod_graph_fit(scan, &gm, (const size_t(*)[2])edges, ne));
    const double *ss = pyod_graph_scores(scan, NULL);
    CHECK_TRUE(ss[20] > ss[3] && ss[21] > ss[3], "SCAN ranks bridge/pendant above core");
    size_t bad_edge[1][2] = {{0, 99}};
    CHECK_TRUE(pyod_graph_fit(scan, &gm, (const size_t(*)[2])bad_edge, 1) == PYOD_ERR_PARAM, "edge out of range");
    pyod_graph_free(scan);
    pyod_graph_detector *gd[2] = {pyod_anomalous_create(NULL), pyod_radar_create(NULL)};
    for (int k = 0; k < 2; k++) {
        CHECK_OK(pyod_graph_fit(gd[k], &gm, (const size_t(*)[2])edges, ne));
        size_t n;
        CHECK_TRUE(pyod_graph_scores(gd[k], &n) && n == 22, "%s scores", pyod_graph_name(gd[k]));
        pyod_graph_free(gd[k]);
    }

    /* Embedding detector with a custom encoder. */
    const char *docs[61];
    static char buf[60][64];
    for (int i = 0; i < 60; i++) {
        snprintf(buf[i], sizeof buf[i], "plain words sentence number %d", i % 7);
        docs[i] = buf[i];
    }
    docs[60] = "4815162342 0000000000000000000000000000";
    pyod_embedding_od *emb = pyod_embedding_create(count_encoder, NULL, NULL, NULL);
    CHECK_OK(pyod_embedding_fit(emb, docs, 61));
    CHECK_TRUE(pyod_labels(pyod_embedding_detector(emb), NULL)[60] == 1, "embedding flags the odd document");
    pyod_embedding_od *mods[1] = {pyod_embedding_create(count_encoder, NULL, NULL, NULL)};
    pyod_multimodal_od *mm = pyod_multimodal_create(mods, 1, PYOD_COMBINE_AVERAGE, 1, 0.05);
    CHECK_OK(pyod_multimodal_fit(mm, docs, 61));
    uint8_t lab[2];
    CHECK_OK(pyod_multimodal_predict(mm, docs + 59, 2, lab));
    CHECK_TRUE(lab[0] == 0 && lab[1] == 1, "multimodal predict");
    pyod_embedding_free(emb);
    pyod_multimodal_free(mm);

    /* Log-mel: a 1 kHz tone (mel 15 on the Slaney scale) peaks in band ~19 of
     * 64 bands spanning 0-49.9 mel. */
    double tone[22050];
    for (int i = 0; i < 22050; i++) tone[i] = sin(2.0 * 3.141592653589793 * 1000.0 * i / 22050.0);
    pyod_matrix spec;
    CHECK_OK(pyod_log_mel_spectrogram(tone, 22050, 22050.0, 64, 512, &spec));
    size_t mid = spec.cols / 2, peak = 0;
    for (size_t m = 1; m < 64; m++)
        if (PYOD_AT(&spec, m, mid) > PYOD_AT(&spec, peak, mid)) peak = m;
    CHECK_TRUE(peak >= 18 && peak <= 20, "1 kHz peak in mel band %zu", peak);
    pyod_matrix_free(&spec);
    CHECK_REPORT("ts_graph");
}
