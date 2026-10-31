/* Random numbers, statistics, neighbours and special functions standing in for
 * the NumPy / SciPy / sklearn calls PyOD relies on. */
#include "internal.h"

/* ---------------- RNG: SplitMix64 ---------------- */

void rng_seed(pyod_rng *r, uint64_t seed) { r->s = seed; }

uint64_t rng_u64(pyod_rng *r) {
    r->s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = r->s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double rng_f64(pyod_rng *r) { return (double)(rng_u64(r) >> 11) / 9007199254740992.0; }

size_t rng_below(pyod_rng *r, size_t n) { return n ? (size_t)(rng_u64(r) % (uint64_t)n) : 0; }

double rng_normal(pyod_rng *r) {
    double u1 = fmax(rng_f64(r), 2.2250738585072014e-308);
    double u2 = rng_f64(r);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PYOD_PI * u2);
}

void rng_sample(pyod_rng *r, size_t n, size_t k, size_t *out) {
    size_t *idx = NEW(size_t, n);
    if (!idx) { /* fall back to the identity prefix */
        for (size_t i = 0; i < k; i++) out[i] = i;
        return;
    }
    for (size_t i = 0; i < n; i++) idx[i] = i;
    if (k > n) k = n;
    for (size_t i = 0; i < k; i++) {
        size_t j = i + rng_below(r, n - i);
        size_t t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
    }
    memcpy(out, idx, k * sizeof(size_t));
    free(idx);
}

/* ---------------- statistics ---------------- */

double vec_mean(const double *v, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += v[i];
    return n ? s / (double)n : NAN;
}

double vec_std(const double *v, size_t n) {
    double m = vec_mean(v, n), s = 0.0;
    for (size_t i = 0; i < n; i++) s += (v[i] - m) * (v[i] - m);
    return n ? sqrt(s / (double)n) : NAN;
}

int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

double vec_percentile(const double *v, size_t n, double q) {
    if (!n) return NAN;
    double *s = NEW(double, n);
    if (!s) return NAN;
    memcpy(s, v, n * sizeof(double));
    qsort(s, n, sizeof(double), cmp_double);
    q = fmin(100.0, fmax(0.0, q));
    double pos = q / 100.0 * (double)(n - 1);
    size_t lo = (size_t)floor(pos), hi = (size_t)ceil(pos);
    double r = s[lo] + (s[hi] - s[lo]) * (pos - (double)lo);
    free(s);
    return r;
}

double vec_median(const double *v, size_t n) { return vec_percentile(v, n, 50.0); }

double vec_skew(const double *v, size_t n) {
    double m = vec_mean(v, n), m2 = 0.0, m3 = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = v[i] - m;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= (double)n;
    m3 /= (double)n;
    return m2 == 0.0 ? 0.0 : m3 / pow(m2, 1.5);
}

double vec_kurtosis(const double *v, size_t n) {
    double m = vec_mean(v, n), m2 = 0.0, m4 = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (v[i] - m) * (v[i] - m);
        m2 += d;
        m4 += d * d;
    }
    m2 /= (double)n;
    m4 /= (double)n;
    return m2 == 0.0 ? -3.0 : m4 / (m2 * m2) - 3.0;
}

double vec_min(const double *v, size_t n) {
    double m = INFINITY;
    for (size_t i = 0; i < n; i++) m = fmin(m, v[i]);
    return m;
}

double vec_max(const double *v, size_t n) {
    double m = -INFINITY;
    for (size_t i = 0; i < n; i++) m = fmax(m, v[i]);
    return m;
}

double sign(double x) { return x > 0.0 ? 1.0 : x < 0.0 ? -1.0 : 0.0; }

/* Stable merge-sort argsort. */
static void merge_argsort(const double *v, size_t *idx, size_t *tmp, size_t n) {
    if (n < 2) return;
    size_t h = n / 2;
    merge_argsort(v, idx, tmp, h);
    merge_argsort(v, idx + h, tmp, n - h);
    size_t i = 0, j = h, k = 0;
    while (i < h && j < n) tmp[k++] = v[idx[j]] < v[idx[i]] ? idx[j++] : idx[i++];
    while (i < h) tmp[k++] = idx[i++];
    while (j < n) tmp[k++] = idx[j++];
    memcpy(idx, tmp, n * sizeof(size_t));
}

void argsort(const double *v, size_t n, size_t *idx) {
    for (size_t i = 0; i < n; i++) idx[i] = i;
    size_t *tmp = NEW(size_t, n);
    if (!tmp) return;
    merge_argsort(v, idx, tmp, n);
    free(tmp);
}

/* numba's quicksort: median-of-three partitioning, insertion sort below 15.
 * Signed indices, as in numba, so an empty range (high < low) is harmless. */
void numba_argsort_u32(const uint32_t *a, size_t n, size_t *r) {
    for (size_t i = 0; i < n; i++) r[i] = i;
    if (n < 2) return;
    long stack_lo[128], stack_hi[128];
    int sp = 0;
    stack_lo[sp] = 0;
    stack_hi[sp++] = (long)n - 1;
#define SWP(x, y) do { size_t t_ = r[x]; r[x] = r[y]; r[y] = t_; } while (0)
    while (sp > 0) {
        sp--;
        long low = stack_lo[sp], high = stack_hi[sp];
        while (high - low >= 15) {
            long mid = (low + high) >> 1;
            if (a[r[mid]] < a[r[low]]) SWP(low, mid);
            if (a[r[high]] < a[r[mid]]) SWP(high, mid);
            if (a[r[mid]] < a[r[low]]) SWP(low, mid);
            uint32_t pivot = a[r[mid]];
            SWP(high, mid);
            long i = low, j = high - 1;
            for (;;) {
                while (i < high && a[r[i]] < pivot) i++;
                while (j >= low && pivot < a[r[j]]) j--;
                if (i >= j) break;
                SWP(i, j);
                i++;
                j--;
            }
            SWP(i, high);
            if (high - i > i - low) {
                if (high > i) {
                    stack_lo[sp] = i + 1;
                    stack_hi[sp++] = high;
                }
                high = i - 1;
            } else {
                if (i > low) {
                    stack_lo[sp] = low;
                    stack_hi[sp++] = i - 1;
                }
                low = i + 1;
            }
        }
        for (long i = low + 1; i <= high; i++) {
            size_t k = r[i];
            uint32_t v = a[k];
            long j = i;
            while (j > low && v < a[r[j - 1]]) {
                r[j] = r[j - 1];
                j--;
            }
            r[j] = k;
        }
    }
#undef SWP
}

double sq_euclidean(const double *a, const double *b, size_t d) {
    double s = 0.0;
    for (size_t i = 0; i < d; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
    return s;
}

double euclidean(const double *a, const double *b, size_t d) { return sqrt(sq_euclidean(a, b, d)); }

double dot(const double *a, const double *b, size_t d) {
    double s = 0.0;
    for (size_t i = 0; i < d; i++) s += a[i] * b[i];
    return s;
}

void column_ecdf(const pyod_matrix *x, double *out) {
    size_t n = x->rows, d = x->cols;
    double *col = NEW(double, n);
    if (!col) return;
    for (size_t j = 0; j < d; j++) {
        for (size_t i = 0; i < n; i++) col[i] = PYOD_AT(x, i, j);
        qsort(col, n, sizeof(double), cmp_double);
        for (size_t i = 0; i < n; i++) {
            double v = PYOD_AT(x, i, j);
            size_t lo = 0, hi = n;
            while (lo < hi) {
                size_t mid = (lo + hi) / 2;
                if (col[mid] <= v) lo = mid + 1;
                else hi = mid;
            }
            out[i * d + j] = (double)lo / (double)n;
        }
    }
    free(col);
}

typedef struct {
    double d;
    size_t i;
} dist_idx;

static int cmp_dist_idx(const void *a, const void *b) {
    const dist_idx *x = a, *y = b;
    if (x->d != y->d) return (x->d > y->d) - (x->d < y->d);
    return (x->i > y->i) - (x->i < y->i);
}

size_t k_nearest(const pyod_matrix *train, const double *q, size_t k, size_t skip, size_t *idx, double *dist) {
    size_t n = train->rows, m = 0;
    dist_idx *all = NEW(dist_idx, n);
    if (!all) return 0;
    for (size_t i = 0; i < n; i++) {
        if (i == skip) continue;
        all[m].d = euclidean(&PYOD_AT(train, i, 0), q, train->cols);
        all[m++].i = i;
    }
    qsort(all, m, sizeof(dist_idx), cmp_dist_idx);
    if (k > m) k = m;
    for (size_t i = 0; i < k; i++) {
        if (idx) idx[i] = all[i].i;
        if (dist) dist[i] = all[i].d;
    }
    free(all);
    return k;
}

pyod_status scaler_fit(std_scaler *s, const pyod_matrix *x) {
    s->d = x->cols;
    s->mean = NEW(double, x->cols);
    s->scale = NEW(double, x->cols);
    if (!s->mean || !s->scale) {
        scaler_free(s);
        return PYOD_ERR_NOMEM;
    }
    for (size_t j = 0; j < x->cols; j++) {
        double m = 0.0, v = 0.0;
        for (size_t i = 0; i < x->rows; i++) m += PYOD_AT(x, i, j);
        m /= (double)x->rows;
        for (size_t i = 0; i < x->rows; i++) v += (PYOD_AT(x, i, j) - m) * (PYOD_AT(x, i, j) - m);
        v = sqrt(v / (double)x->rows);
        s->mean[j] = m;
        s->scale[j] = v == 0.0 ? 1.0 : v;
    }
    return PYOD_OK;
}

void scaler_transform(const std_scaler *s, const pyod_matrix *x, double *out) {
    for (size_t i = 0; i < x->rows; i++)
        for (size_t j = 0; j < x->cols; j++) out[i * x->cols + j] = (PYOD_AT(x, i, j) - s->mean[j]) / s->scale[j];
}

void scaler_free(std_scaler *s) {
    free(s->mean);
    free(s->scale);
    s->mean = s->scale = NULL;
}

/* ---------------- special functions ---------------- */

double lgamma_(double x) {
    static const double g[9] = {0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                                771.32342877765313, -176.61502916214059, 12.507343278686905,
                                -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};
    if (x < 0.5) return log(fabs(PYOD_PI / sin(PYOD_PI * x))) - lgamma_(1.0 - x);
    x -= 1.0;
    double t = x + 7.5, s = g[0];
    for (int i = 1; i < 9; i++) s += g[i] / (x + (double)i);
    return 0.5 * log(2.0 * PYOD_PI) + (x + 0.5) * log(t) - t + log(s);
}

static void gamma_pq(double a, double x, double *p, double *q) {
    if (x <= 0.0) {
        *p = 0.0;
        *q = 1.0;
        return;
    }
    double gln = lgamma_(a);
    if (x < a + 1.0) {
        double sum = 1.0 / a, del = sum, ap = a;
        for (int i = 0; i < 1000; i++) {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if (fabs(del) < fabs(sum) * 1e-16) break;
        }
        *p = exp(log(sum) - x + a * log(x) - gln);
        *q = 1.0 - *p;
    } else {
        double tiny = 1e-300, b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
        for (int i = 1; i < 1000; i++) {
            double an = -(double)i * ((double)i - a);
            b += 2.0;
            d = an * d + b;
            if (fabs(d) < tiny) d = tiny;
            c = b + an / c;
            if (fabs(c) < tiny) c = tiny;
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (fabs(del - 1.0) < 1e-16) break;
        }
        *q = exp(-x + a * log(x) - gln) * h;
        *p = 1.0 - *q;
    }
}

double gamma_p(double a, double x) {
    double p, q;
    gamma_pq(a, x, &p, &q);
    return p;
}

double gamma_q(double a, double x) {
    double p, q;
    gamma_pq(a, x, &p, &q);
    return q;
}

double erf_(double x) { return sign(x) * gamma_p(0.5, x * x); }

double erfc_(double x) { return x >= 0.0 ? gamma_q(0.5, x * x) : 1.0 + gamma_p(0.5, x * x); }

double chi2_ppf(double q, double df) {
    double lo = 0.0, hi = fmax(df, 1.0);
    while (gamma_p(df / 2.0, hi / 2.0) < q) hi *= 2.0;
    for (int i = 0; i < 200; i++) {
        double mid = 0.5 * (lo + hi);
        if (gamma_p(df / 2.0, mid / 2.0) < q) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

double norm_ppf(double p) {
    static const double a[6] = {-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
                                1.383577518672690e2, -3.066479806614716e1, 2.506628277459239};
    static const double b[5] = {-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
                                6.680131188771972e1, -1.328068155288572e1};
    static const double c[6] = {-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
                                -2.549732539343734, 4.374664141464968, 2.938163982698783};
    static const double d[4] = {7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416};
    double pl = 0.02425, x;
    if (p < pl) {
        double q = sqrt(-2.0 * log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= 1.0 - pl) {
        double q = p - 0.5, r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        double q = sqrt(-2.0 * log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    double e = 0.5 * erfc_(-x / sqrt(2.0)) - p;
    double u = e * sqrt(2.0 * PYOD_PI) * exp(x * x / 2.0);
    return x - u / (1.0 + x * u / 2.0);
}

double logsumexp(const double *v, size_t n) {
    double m = vec_max(v, n);
    if (m == -INFINITY) return m;
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += exp(v[i] - m);
    return m + log(s);
}

/* ---------------- isolation trees ---------------- */

double itree_c(size_t n) {
    if (n <= 1) return 0.0;
    if (n == 2) return 1.0;
    double m = (double)n;
    return 2.0 * (log(m - 1.0) + PYOD_EULER_GAMMA) - 2.0 * (m - 1.0) / m;
}

itree_node *itree_build(const pyod_matrix *x, size_t *idx, size_t n, size_t depth, size_t max_depth, pyod_rng *r) {
    itree_node *node = NEW(itree_node, 1);
    if (!node) return NULL;
    node->leaf = 1;
    node->size = n;
    if (depth >= max_depth || n <= 1) return node;
    size_t d = x->cols, nc = 0;
    size_t *cand = NEW(size_t, d);
    double *lo = NEW(double, d), *hi = NEW(double, d);
    if (!cand || !lo || !hi) {
        free(cand), free(lo), free(hi);
        return node;
    }
    for (size_t f = 0; f < d; f++) {
        double a = INFINITY, b = -INFINITY;
        for (size_t i = 0; i < n; i++) {
            double v = PYOD_AT(x, idx[i], f);
            a = fmin(a, v);
            b = fmax(b, v);
        }
        if (b > a) {
            cand[nc] = f;
            lo[nc] = a;
            hi[nc++] = b;
        }
    }
    if (nc == 0) {
        free(cand), free(lo), free(hi);
        return node;
    }
    size_t pick = rng_below(r, nc);
    size_t f = cand[pick];
    double value = lo[pick] + rng_f64(r) * (hi[pick] - lo[pick]);
    free(cand), free(lo), free(hi);
    size_t mid = 0;
    for (size_t k = 0; k < n; k++) {
        if (PYOD_AT(x, idx[k], f) < value) {
            size_t t = idx[k];
            idx[k] = idx[mid];
            idx[mid++] = t;
        }
    }
    node->leaf = 0;
    node->feature = f;
    node->value = value;
    node->left = itree_build(x, idx, mid, depth + 1, max_depth, r);
    node->right = itree_build(x, idx + mid, n - mid, depth + 1, max_depth, r);
    return node;
}

void itree_free(itree_node *t) {
    if (!t) return;
    itree_free(t->left);
    itree_free(t->right);
    free(t);
}

double itree_path(const itree_node *t, const double *row, double depth) {
    while (t && !t->leaf) {
        t = row[t->feature] < t->value ? t->left : t->right;
        depth += 1.0;
    }
    return t ? depth + itree_c(t->size) : depth;
}
