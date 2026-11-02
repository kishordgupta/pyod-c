/* Complete ECOD example. Build and run: make examples && build/examples/quickstart */
#include <stdio.h>
#include "pyod.h"

int main(void) {
    double train_data[] = {0,0, .1,.2, -.2,.1, .2,-.1, -.1,-.2, .05,.08, -.05,.03, 3,3};
    double test_data[] = {.02,.04, 7,-7, .1,-.15};
    /* Views borrow these arrays. Do not free views with pyod_matrix_free. */
    pyod_matrix train = pyod_matrix_view(train_data, 8, 2);
    pyod_matrix test = pyod_matrix_view(test_data, 3, 2);
    double scores[3], scaled[3], confidence[3];
    uint8_t labels[3];
    pyod_ecod_params params = pyod_ecod_defaults();
    params.contamination = 0.1;
    pyod_detector *det = pyod_ecod_create(&params);
    if (!det) { fprintf(stderr, "could not create ECOD\n"); return 1; }
    pyod_status st = pyod_fit(det, &train);
    if (st == PYOD_OK) st = pyod_decision_function(det, &test, scores);
    if (st == PYOD_OK) st = pyod_predict(det, &test, labels);
    if (st == PYOD_OK) st = pyod_predict_proba(det, &test, PYOD_PROBA_LINEAR, scaled);
    if (st == PYOD_OK) st = pyod_predict_confidence(det, &test, confidence);
    if (st != PYOD_OK) {
        fprintf(stderr, "%s\n", pyod_strerror(st));
        pyod_free(det);
        return 1;
    }
    printf("threshold: %.6f\n", pyod_threshold(det));
    for (size_t i = 0; i < test.rows; i++)
        printf("row %zu: score=%.6f label=%u scaled_score=%.6f confidence=%.6f\n",
               i, scores[i], (unsigned)labels[i], scaled[i], confidence[i]);
    /* Scaled scores and confidence are not calibrated real-world probabilities. */
    pyod_free(det);
    return 0;
}
