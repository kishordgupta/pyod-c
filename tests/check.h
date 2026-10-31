/* Minimal test helpers. */
#ifndef PYOD_CHECK_H
#define PYOD_CHECK_H
#include <stdio.h>
#include <stdlib.h>

static int check_failures = 0;

#define CHECK_TRUE(cond, ...)                                                      \
    do {                                                                           \
        if (!(cond)) {                                                             \
            fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond);     \
            fprintf(stderr, __VA_ARGS__);                                          \
            fprintf(stderr, "\n");                                                 \
            check_failures++;                                                      \
        }                                                                          \
    } while (0)

#define CHECK_OK(expr) CHECK_TRUE((expr) == PYOD_OK, "%s", pyod_strerror(expr))

#define CHECK_REPORT(name)                                                         \
    do {                                                                           \
        if (check_failures) {                                                      \
            fprintf(stderr, "%s: %d failure(s)\n", name, check_failures);         \
            return 1;                                                              \
        }                                                                          \
        printf("%s: ok\n", name);                                                  \
        return 0;                                                                  \
    } while (0)
#endif
