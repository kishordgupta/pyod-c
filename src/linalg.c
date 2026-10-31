/* Dense linear algebra standing in for NumPy / SciPy / sklearn helpers. */
#include "internal.h"

void mat_mul(const double *a, const double *b, double *c, size_t n, size_t k, size_t m) {
    memset(c, 0, n * m * sizeof(double));
    for (size_t i = 0; i < n; i++)
        for (size_t p = 0; p < k; p++) {
            double aip = a[i * k + p];
            if (aip == 0.0) continue;
            const double *bp = b + p * m;
            double *ci = c + i * m;
            for (size_t j = 0; j < m; j++) ci[j] += aip * bp[j];
        }
}

void mat_transpose(const double *a, double *t, size_t n, size_t m) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++) t[j * n + i] = a[i * m + j];
}

void col_mean(const pyod_matrix *x, double *mean) {
    memset(mean, 0, x->cols * sizeof(double));
    for (size_t i = 0; i < x->rows; i++)
        for (size_t j = 0; j < x->cols; j++) mean[j] += PYOD_AT(x, i, j);
    for (size_t j = 0; j < x->cols; j++) mean[j] /= (double)x->rows;
}

void covariance(const pyod_matrix *x, const double *center, size_t ddof, double *out) {
    size_t d = x->cols;
    memset(out, 0, d * d * sizeof(double));
    for (size_t r = 0; r < x->rows; r++) {
        const double *row = &PYOD_AT(x, r, 0);
        for (size_t i = 0; i < d; i++) {
            double di = row[i] - center[i];
            for (size_t j = i; j < d; j++) out[i * d + j] += di * (row[j] - center[j]);
        }
    }
    double denom = x->rows > ddof ? (double)(x->rows - ddof) : 1.0;
    for (size_t i = 0; i < d; i++)
        for (size_t j = i; j < d; j++) {
            out[i * d + j] /= denom;
            out[j * d + i] = out[i * d + j];
        }
}

/* Householder tridiagonalisation then implicit QL (EISPACK tred2 / tql2, as in JAMA). */
void sym_eigen(const double *a, size_t n, double *vals, double *vecs) {
    if (n == 0) return;
    double *V = NEW(double, n * n), *d = NEW(double, n), *e = NEW(double, n);
    if (!V || !d || !e) {
        free(V), free(d), free(e);
        return;
    }
#define v(i, j) V[(i) * n + (j)]
    memcpy(V, a, n * n * sizeof(double));
    for (size_t j = 0; j < n; j++) d[j] = v(n - 1, j);
    for (size_t i = n - 1; i > 0; i--) {
        double scale = 0.0, h = 0.0;
        for (size_t k = 0; k < i; k++) scale += fabs(d[k]);
        if (scale == 0.0) {
            e[i] = d[i - 1];
            for (size_t j = 0; j < i; j++) {
                d[j] = v(i - 1, j);
                v(i, j) = 0.0;
                v(j, i) = 0.0;
            }
        } else {
            for (size_t k = 0; k < i; k++) {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            double f = d[i - 1], g = sqrt(h);
            if (f > 0) g = -g;
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for (size_t j = 0; j < i; j++) e[j] = 0.0;
            for (size_t j = 0; j < i; j++) {
                f = d[j];
                v(j, i) = f;
                g = e[j] + v(j, j) * f;
                for (size_t k = j + 1; k < i; k++) {
                    g += v(k, j) * d[k];
                    e[k] += v(k, j) * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for (size_t j = 0; j < i; j++) {
                e[j] /= h;
                f += e[j] * d[j];
            }
            double hh = f / (h + h);
            for (size_t j = 0; j < i; j++) e[j] -= hh * d[j];
            for (size_t j = 0; j < i; j++) {
                f = d[j];
                g = e[j];
                for (size_t k = j; k < i; k++) v(k, j) -= f * e[k] + g * d[k];
                d[j] = v(i - 1, j);
                v(i, j) = 0.0;
            }
        }
        d[i] = h;
    }
    for (size_t i = 0; i + 1 < n; i++) {
        v(n - 1, i) = v(i, i);
        v(i, i) = 1.0;
        double h = d[i + 1];
        if (h != 0.0) {
            for (size_t k = 0; k <= i; k++) d[k] = v(k, i + 1) / h;
            for (size_t j = 0; j <= i; j++) {
                double g = 0.0;
                for (size_t k = 0; k <= i; k++) g += v(k, i + 1) * v(k, j);
                for (size_t k = 0; k <= i; k++) v(k, j) -= g * d[k];
            }
        }
        for (size_t k = 0; k <= i; k++) v(k, i + 1) = 0.0;
    }
    for (size_t j = 0; j < n; j++) {
        d[j] = v(n - 1, j);
        v(n - 1, j) = 0.0;
    }
    v(n - 1, n - 1) = 1.0;
    e[0] = 0.0;
    /* tql2 */
    for (size_t i = 1; i < n; i++) e[i - 1] = e[i];
    e[n - 1] = 0.0;
    double f = 0.0, tst1 = 0.0;
    for (size_t l = 0; l < n; l++) {
        tst1 = fmax(tst1, fabs(d[l]) + fabs(e[l]));
        size_t m = l;
        while (m < n) {
            if (fabs(e[m]) <= PYOD_DBL_EPS * tst1) break;
            m++;
        }
        if (m >= n) m = n - 1;
        if (m > l) {
            for (int iter = 0; iter < 200; iter++) {
                double g = d[l], p = (d[l + 1] - g) / (2.0 * e[l]);
                double r = hypot(p, 1.0);
                if (p < 0) r = -r;
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                double dl1 = d[l + 1], h = g - d[l];
                for (size_t i = l + 2; i < n; i++) d[i] -= h;
                f += h;
                p = d[m];
                double c = 1.0, c2 = c, c3 = c, el1 = e[l + 1], s = 0.0, s2 = 0.0;
                for (size_t ii = m; ii-- > l;) {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[ii];
                    h = c * p;
                    r = hypot(p, e[ii]);
                    e[ii + 1] = s * r;
                    s = e[ii] / r;
                    c = p / r;
                    p = c * d[ii] - s * g;
                    d[ii + 1] = h + s * (c * g + s * d[ii]);
                    for (size_t k = 0; k < n; k++) {
                        h = v(k, ii + 1);
                        v(k, ii + 1) = s * v(k, ii) + c * h;
                        v(k, ii) = c * v(k, ii) - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if (fabs(e[l]) <= PYOD_DBL_EPS * tst1) break;
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
    size_t *order = NEW(size_t, n);
    if (order) {
        double *neg = NEW(double, n);
        if (neg) {
            for (size_t i = 0; i < n; i++) neg[i] = -d[i];
            argsort(neg, n, order);
            for (size_t i = 0; i < n; i++) {
                vals[i] = d[order[i]];
                for (size_t k = 0; k < n; k++) vecs[i * n + k] = v(k, order[i]);
            }
            free(neg);
        }
        free(order);
    }
#undef v
    free(V), free(d), free(e);
}

int cholesky(const double *a, size_t n, double *l) {
    memset(l, 0, n * n * sizeof(double));
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j <= i; j++) {
            double s = 0.0;
            for (size_t k = 0; k < j; k++) s += l[i * n + k] * l[j * n + k];
            if (i == j) {
                double v = a[i * n + i] - s;
                if (v <= 0.0 || !isfinite(v)) return 0;
                l[i * n + j] = sqrt(v);
            } else {
                l[i * n + j] = (a[i * n + j] - s) / l[j * n + j];
            }
        }
    return 1;
}

void forward_solve(const double *l, const double *b, double *y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double s = 0.0;
        for (size_t k = 0; k < i; k++) s += l[i * n + k] * y[k];
        y[i] = (b[i] - s) / l[i * n + i];
    }
}

void pinvh(const double *a, size_t n, double *out) {
    double *vals = NEW(double, n), *vecs = NEW(double, n * n);
    memset(out, 0, n * n * sizeof(double));
    if (!vals || !vecs) {
        free(vals), free(vecs);
        return;
    }
    sym_eigen(a, n, vals, vecs);
    double mx = 0.0;
    for (size_t i = 0; i < n; i++) mx = fmax(mx, fabs(vals[i]));
    double cutoff = mx * (double)n * PYOD_DBL_EPS;
    for (size_t t = 0; t < n; t++) {
        if (fabs(vals[t]) <= cutoff) continue;
        const double *v = vecs + t * n;
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++) out[i * n + j] += v[i] * v[j] / vals[t];
    }
    free(vals), free(vecs);
}

double fast_logdet(const double *a, size_t n) {
    double *l = NEW(double, n * n);
    if (!l) return NAN;
    double r = -INFINITY;
    if (cholesky(a, n, l)) {
        r = 0.0;
        for (size_t i = 0; i < n; i++) r += 2.0 * log(l[i * n + i]);
    }
    free(l);
    return r;
}

int lin_solve(const double *a_in, const double *b, double *x, size_t n, size_t m) {
    double *a = NEW(double, n * n);
    if (!a) return 0;
    memcpy(a, a_in, n * n * sizeof(double));
    memcpy(x, b, n * m * sizeof(double));
    for (size_t col = 0; col < n; col++) {
        size_t piv = col;
        for (size_t i = col + 1; i < n; i++)
            if (fabs(a[i * n + col]) > fabs(a[piv * n + col])) piv = i;
        if (fabs(a[piv * n + col]) < 1e-300) {
            free(a);
            return 0;
        }
        if (piv != col) {
            for (size_t k = 0; k < n; k++) {
                double t = a[col * n + k];
                a[col * n + k] = a[piv * n + k];
                a[piv * n + k] = t;
            }
            for (size_t k = 0; k < m; k++) {
                double t = x[col * m + k];
                x[col * m + k] = x[piv * m + k];
                x[piv * m + k] = t;
            }
        }
        for (size_t r = col + 1; r < n; r++) {
            double f = a[r * n + col] / a[col * n + col];
            if (f == 0.0) continue;
            for (size_t k = col; k < n; k++) a[r * n + k] -= f * a[col * n + k];
            for (size_t k = 0; k < m; k++) x[r * m + k] -= f * x[col * m + k];
        }
    }
    for (size_t col = n; col-- > 0;)
        for (size_t k = 0; k < m; k++) {
            double s = 0.0;
            for (size_t j = col + 1; j < n; j++) s += a[col * n + j] * x[j * m + k];
            x[col * m + k] = (x[col * m + k] - s) / a[col * n + col];
        }
    free(a);
    return 1;
}

void mahalanobis_sq(const pyod_matrix *x, const double *loc, const double *prec, double *out) {
    size_t d = x->cols;
    double *c = NEW(double, d);
    if (!c) return;
    for (size_t r = 0; r < x->rows; r++) {
        for (size_t j = 0; j < d; j++) c[j] = PYOD_AT(x, r, j) - loc[j];
        double s = 0.0;
        for (size_t i = 0; i < d; i++) s += c[i] * dot(prec + i * d, c, d);
        out[r] = s;
    }
    free(c);
}

size_t nearest_center(const double *centers, size_t k, size_t d, const double *row, double *dist2) {
    size_t best = 0;
    double bd = INFINITY;
    for (size_t c = 0; c < k; c++) {
        double v = sq_euclidean(centers + c * d, row, d);
        if (v < bd) {
            bd = v;
            best = c;
        }
    }
    if (dist2) *dist2 = bd;
    return best;
}

double kmeans(const pyod_matrix *x, size_t k, size_t n_init, size_t max_iter, pyod_rng *r, double *centers,
              size_t *labels) {
    size_t n = x->rows, d = x->cols;
    double *cen = NEW(double, k * d), *d2 = NEW(double, n), *sums = NEW(double, k * d);
    size_t *lab = NEW(size_t, n), *counts = NEW(size_t, k);
    double best = INFINITY;
    if (!cen || !d2 || !sums || !lab || !counts) goto done;
    for (size_t init = 0; init < (n_init ? n_init : 1); init++) {
        memcpy(cen, &PYOD_AT(x, rng_below(r, n), 0), d * sizeof(double));
        for (size_t i = 0; i < n; i++) d2[i] = sq_euclidean(&PYOD_AT(x, i, 0), cen, d);
        for (size_t c = 1; c < k; c++) {
            double total = 0.0;
            for (size_t i = 0; i < n; i++) total += d2[i];
            size_t next = n - 1;
            if (total <= 0.0) {
                next = rng_below(r, n);
            } else {
                double t = rng_f64(r) * total;
                for (size_t i = 0; i < n; i++) {
                    t -= d2[i];
                    if (t <= 0.0) {
                        next = i;
                        break;
                    }
                }
            }
            memcpy(cen + c * d, &PYOD_AT(x, next, 0), d * sizeof(double));
            for (size_t i = 0; i < n; i++) d2[i] = fmin(d2[i], sq_euclidean(&PYOD_AT(x, i, 0), cen + c * d, d));
        }
        for (size_t i = 0; i < n; i++) lab[i] = SIZE_MAX;
        for (size_t it = 0; it < max_iter; it++) {
            int changed = 0;
            for (size_t i = 0; i < n; i++) {
                size_t l = nearest_center(cen, k, d, &PYOD_AT(x, i, 0), NULL);
                changed |= l != lab[i];
                lab[i] = l;
            }
            memset(sums, 0, k * d * sizeof(double));
            memset(counts, 0, k * sizeof(size_t));
            for (size_t i = 0; i < n; i++) {
                counts[lab[i]]++;
                for (size_t j = 0; j < d; j++) sums[lab[i] * d + j] += PYOD_AT(x, i, j);
            }
            for (size_t c = 0; c < k; c++)
                if (counts[c])
                    for (size_t j = 0; j < d; j++) cen[c * d + j] = sums[c * d + j] / (double)counts[c];
            if (!changed) break;
        }
        double inertia = 0.0;
        for (size_t i = 0; i < n; i++) inertia += sq_euclidean(&PYOD_AT(x, i, 0), cen + lab[i] * d, d);
        if (inertia < best) {
            best = inertia;
            memcpy(centers, cen, k * d * sizeof(double));
            memcpy(labels, lab, n * sizeof(size_t));
        }
    }
done:
    free(cen), free(d2), free(sums), free(lab), free(counts);
    return best;
}
