/* Minimal usage: fit ECOD, score new data, get labels and probabilities. */
#include <stdio.h>
#include "pyod.h"

int main(void) {
    pyod_matrix train, test;
    uint8_t ytrain[200], ytest[50], labels[50];
    pyod_generate_data(200, 2, 0.1, 1, &train, ytrain);
    pyod_generate_data(50, 2, 0.1, 2, &test, ytest);

    pyod_ecod_params p = pyod_ecod_defaults();
    p.contamination = 0.1;
    pyod_detector *det = pyod_ecod_create(&p);
    pyod_status st = pyod_fit(det, &train);
    if (st != PYOD_OK) {
        fprintf(stderr, "fit failed: %s\n", pyod_strerror(st));
        return 1;
    }
    double scores[50], probs[50];
    pyod_decision_function(det, &test, scores);
    pyod_predict(det, &test, labels);
    pyod_predict_proba(det, &test, PYOD_PROBA_UNIFY, probs);
    printf("threshold %.3f, test ROC AUC %.3f\n", pyod_threshold(det), pyod_roc_auc(ytest, scores, 50));
    for (int i = 45; i < 50; i++) printf("sample %d: score %.2f  label %d  P(outlier) %.2f\n", i, scores[i], labels[i], probs[i]);
    pyod_free(det);
    pyod_matrix_free(&train);
    pyod_matrix_free(&test);
    return 0;
}
