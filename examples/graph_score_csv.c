/* Print each graph detector's node scores for CSV node features and an edge
 * list (source,target per line): name|scores. Used to cross-check against PyOD. */
#include "csvio.h"

int main(int argc, char **argv) {
    pyod_matrix x, e;
    if (argc < 3 || read_csv(argv[1], &x) || read_csv(argv[2], &e)) {
        fprintf(stderr, "usage: graph_score_csv nodes.csv edges.csv\n");
        return 2;
    }
    size_t (*edges)[2] = malloc(e.rows * sizeof *edges);
    for (size_t i = 0; i < e.rows; i++) edges[i][0] = (size_t)PYOD_AT(&e, i, 0), edges[i][1] = (size_t)PYOD_AT(&e, i, 1);
    pyod_graph_detector *dets[16];
    size_t nd = 0;
    dets[nd++] = pyod_scan_create(NULL);
    dets[nd++] = pyod_anomalous_create(NULL);
    dets[nd++] = pyod_radar_create(NULL);
    dets[nd++] = pyod_dominant_create(NULL);
    dets[nd++] = pyod_anomalydae_create(NULL);
    dets[nd++] = pyod_cola_create(NULL);
    dets[nd++] = pyod_conad_create(NULL);
    dets[nd++] = pyod_guide_create(NULL);
    for (size_t i = 0; i < nd; i++) {
        pyod_status st = pyod_graph_fit(dets[i], &x, (const size_t(*)[2])edges, e.rows);
        if (st != PYOD_OK) {
            fprintf(stderr, "%s: %s\n", pyod_graph_name(dets[i]), pyod_strerror(st));
        } else {
            size_t n;
            const double *s = pyod_graph_scores(dets[i], &n);
            printf("%s|", pyod_graph_name(dets[i]));
            for (size_t k = 0; k < n; k++) printf("%s%.12e", k ? "," : "", s[k]);
            printf("\n");
        }
        pyod_graph_free(dets[i]);
    }
    free(edges), free(x.data), free(e.data);
    return 0;
}
