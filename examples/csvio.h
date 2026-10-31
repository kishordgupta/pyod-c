/* Tiny CSV reader for the examples (numbers only, comma-separated). */
#ifndef PYOD_CSVIO_H
#define PYOD_CSVIO_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pyod.h"

static int read_csv(const char *path, pyod_matrix *m) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    size_t cap = 1024, n = 0, rows = 0, cols = 0;
    double *data = malloc(cap * sizeof(double));
    char line[1 << 16];
    while (data && fgets(line, sizeof line, f)) {
        size_t c = 0;
        for (char *tok = strtok(line, ",\n\r"); tok; tok = strtok(NULL, ",\n\r")) {
            if (n == cap) {
                double *grown = realloc(data, (cap *= 2) * sizeof(double));
                if (!grown) { free(data); data = NULL; break; }
                data = grown;
            }
            data[n++] = strtod(tok, NULL);
            c++;
        }
        if (c == 0) continue;
        if (cols == 0) cols = c;
        rows++;
    }
    fclose(f);
    if (!data || rows == 0) { free(data); return -1; }
    *m = pyod_matrix_view(data, rows, cols);
    return 0;
}
#endif
