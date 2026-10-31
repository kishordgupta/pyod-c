/* Helpers shared by the neural detectors. */
#ifndef PYOD_DEEP_COMMON_H
#define PYOD_DEEP_COMMON_H

#include "deep/nn.h"

/* Number of entries before the first 0 in a layer-size array (max 8). */
static inline size_t layer_count(const size_t *a) {
    size_t n = 0;
    while (n < 8 && a[n]) n++;
    return n;
}

/* Squared Frobenius-style helpers are built from autodiff ops in each model. */

#endif
