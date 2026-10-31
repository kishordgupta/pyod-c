/* ROD: rotation-based outlier detection (Almardeny et al., 2020). Port of
 * pyod.models.rod. Data with more than 3 features is robust-scaled and split
 * into every 3-D subspace; each is scored by the volume of rotating its points
 * about the geometric median, and sigmoid-squashed subspace scores are averaged. */
#include "internal.h"

typedef struct {
    double data_min, data_max, lo, hi;
} angle_scaler;

typedef struct {
    double gm[3], median;
    angle_scaler s1, s2;
} subspace_model;

typedef struct {
    pyod_rod_params p;
    size_t d;
    double *center, *scale; /* RobustScaler (d > 3) */
    subspace_model *subs;
    size_t n_subs;
} rod_impl;

static angle_scaler scaler_fit_angles(const double *v, size_t n, double lo, double hi) {
    angle_scaler s = {fmin(lo, hi), fmax(lo, hi), lo, hi};
    for (size_t i = 0; i < n; i++) s.data_min = fmin(s.data_min, v[i]), s.data_max = fmax(s.data_max, v[i]);
    return s;
}

static double scale_angle(const angle_scaler *s, double v) {
    double range = s->data_max - s->data_min == 0.0 ? 1.0 : s->data_max - s->data_min;
    return (v - s->data_min) / range * (s->hi - s->lo) + s->lo;
}

static int cmp_point3(const void *a, const void *b) {
    const double *x = a, *y = b;
    for (int k = 0; k < 3; k++)
        if (x[k] != y[k]) return (x[k] > y[k]) - (x[k] < y[k]);
    return 0;
}

static void geometric_median(const double *x, size_t n, double *gm) {
    double *pts = NEW(double, 3 * n);
    if (!pts) return;
    memcpy(pts, x, 3 * n * sizeof(double));
    qsort(pts, n, 3 * sizeof(double), cmp_point3);
    size_t m = 0;
    for (size_t i = 0; i < n; i++)
        if (m == 0 || cmp_point3(pts + 3 * i, pts + 3 * (m - 1)) != 0) memcpy(pts + 3 * m++, pts + 3 * i, 3 * sizeof(double));
    for (int k = 0; k < 3; k++) {
        gm[k] = 0.0;
        for (size_t i = 0; i < m; i++) gm[k] += pts[3 * i + k] / (double)m;
    }
    for (int iter = 0; iter < 100000; iter++) {
        double dinvs = 0.0, t[3] = {0, 0, 0};
        size_t n_zero = 0;
        for (size_t i = 0; i < m; i++) {
            double dd = euclidean(pts + 3 * i, gm, 3);
            if (dd == 0.0) {
                n_zero++;
                continue;
            }
            dinvs += 1.0 / dd;
            for (int k = 0; k < 3; k++) t[k] += pts[3 * i + k] / dd;
        }
        if (n_zero == m) break;
        for (int k = 0; k < 3; k++) t[k] /= dinvs;
        double next[3];
        if (n_zero == 0) {
            memcpy(next, t, sizeof next);
        } else {
            double rr = 0.0;
            for (int k = 0; k < 3; k++) rr += ((t[k] - gm[k]) * dinvs) * ((t[k] - gm[k]) * dinvs);
            rr = sqrt(rr);
            double r_inv = rr == 0.0 ? 0.0 : (double)n_zero / rr;
            for (int k = 0; k < 3; k++) next[k] = fmax(0.0, 1.0 - r_inv) * t[k] + fmin(1.0, r_inv) * gm[k];
        }
        int done = euclidean(gm, next, 3) < 1e-5;
        memcpy(gm, next, sizeof next);
        if (done) break;
    }
    free(pts);
}

/* Port of rod_3D: MAD-scaled rotation costs of n 3-D points. Fits `model`
 * when `fit` is set, otherwise uses it. */
static void rod_3d(const double *x, size_t n, subspace_model *model, int fit, double *out) {
    if (fit) geometric_median(x, n, model->gm);
    const double *gm = model->gm;
    double norm = sqrt(dot(gm, gm, 3));
    double *ang = NEW(double, n), *vn = NEW(double, n), *first = NEW(double, n), *second = NEW(double, n), *cost = NEW(double, n);
    int *valid = NEW(int, n);
    if (!ang || !vn || !first || !second || !cost || !valid) goto done;
    size_t nf = 0, ns = 0;
    for (size_t i = 0; i < n; i++) {
        double c[3];
        for (int k = 0; k < 3; k++) c[k] = x[3 * i + k] - gm[k];
        vn[i] = sqrt(dot(c, c, 3));
        valid[i] = 1;
        if (norm == 0.0) ang[i] = PYOD_PI / 2.0;
        else if (vn[i] * norm > 0.0) ang[i] = acos(fmin(1.0, fmax(-1.0, dot(c, gm, 3) / (vn[i] * norm))));
        else valid[i] = 0;
        if (valid[i]) {
            if (ang[i] <= PYOD_PI / 2.0) first[nf++] = ang[i];
            else second[ns++] = ang[i];
        }
    }
    if (fit) {
        model->s1 = scaler_fit_angles(first, nf, 0.001, 0.955);
        model->s2 = scaler_fit_angles(second, ns, PYOD_PI / 2.0 + 0.001, 2.186);
    }
    for (size_t i = 0; i < n; i++) {
        if (!valid[i]) {
            cost[i] = 0.0;
            continue;
        }
        double g = ang[i] <= PYOD_PI / 2.0 ? scale_angle(&model->s1, ang[i]) : scale_angle(&model->s2, ang[i]);
        cost[i] = pow(vn[i], 3) * cos(g) * sin(g) * sin(g);
    }
    if (fit) model->median = vec_median(cost, n);
    for (size_t i = 0; i < n; i++) first[i] = fabs(cost[i] - model->median);
    double mad = vec_median(first, n);
    for (size_t i = 0; i < n; i++) out[i] = 0.6745 * first[i] / mad;
done:
    free(ang), free(vn), free(first), free(second), free(cost), free(valid);
}

static double sigmoid_nan_to_num(double v) {
    if (isnan(v)) v = 0.0;
    else if (isinf(v)) v = v > 0 ? 1.7976931348623157e308 : -1.7976931348623157e308;
    return 1.0 / (1.0 + exp(-v));
}

static pyod_status rod_run(rod_impl *m, const pyod_matrix *x, int fit, double *out) {
    size_t n = x->rows, d = x->cols < 3 ? 3 : x->cols;
    double *xs = NEW(double, n * d), *sub = NEW(double, 3 * n), *s = NEW(double, n);
    pyod_status st = PYOD_ERR_NOMEM;
    if (!xs || !sub || !s) goto done;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < x->cols; j++) xs[i * d + j] = PYOD_AT(x, i, j);
    if (d == 3) {
        if (fit) {
            free(m->subs);
            m->subs = NEW(subspace_model, 1);
            m->n_subs = 1;
            if (!m->subs) goto done;
        }
        rod_3d(xs, n, &m->subs[0], fit, out);
        st = PYOD_OK;
        goto done;
    }
    if (fit) {
        free(m->center), free(m->scale), free(m->subs);
        m->center = NEW(double, d);
        m->scale = NEW(double, d);
        m->n_subs = d * (d - 1) * (d - 2) / 6;
        m->subs = NEW(subspace_model, m->n_subs);
        double *col = NEW(double, n);
        if (!m->center || !m->scale || !m->subs || !col) {
            free(col);
            goto done;
        }
        for (size_t j = 0; j < d; j++) { /* sklearn RobustScaler: median and IQR */
            for (size_t i = 0; i < n; i++) col[i] = xs[i * d + j];
            m->center[j] = vec_median(col, n);
            double iqr = vec_percentile(col, n, 75.0) - vec_percentile(col, n, 25.0);
            m->scale[j] = iqr == 0.0 ? 1.0 : iqr;
        }
        free(col);
    }
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < d; j++) xs[i * d + j] = (xs[i * d + j] - m->center[j]) / m->scale[j];
    for (size_t i = 0; i < n; i++) out[i] = 0.0;
    size_t c = 0;
    for (size_t a = 0; a < d; a++)
        for (size_t b = a + 1; b < d; b++)
            for (size_t e = b + 1; e < d; e++, c++) {
                for (size_t i = 0; i < n; i++) {
                    sub[3 * i] = xs[i * d + a];
                    sub[3 * i + 1] = xs[i * d + b];
                    sub[3 * i + 2] = xs[i * d + e];
                }
                rod_3d(sub, n, &m->subs[c], fit, s);
                for (size_t i = 0; i < n; i++) out[i] += sigmoid_nan_to_num(s[i]);
            }
    for (size_t i = 0; i < n; i++) out[i] /= (double)c;
    st = PYOD_OK;
done:
    free(xs), free(sub), free(s);
    return st;
}

static pyod_status rod_fit(pyod_detector *dt, const pyod_matrix *x) {
    rod_impl *m = dt->impl;
    CHECK(check_contamination(m->p.contamination));
    m->d = x->cols;
    double *s = NEW(double, x->rows);
    if (!s) return PYOD_ERR_NOMEM;
    pyod_status st = rod_run(m, x, 1, s);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(dt, s, x->rows, m->p.contamination);
}

static pyod_status rod_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    return rod_run(d->impl, x, 0, out);
}

static void rod_destroy(void *p) {
    rod_impl *m = p;
    free(m->center), free(m->scale), free(m->subs), free(m);
}

static const pyod_vtable ROD_VT = {"ROD", rod_fit, NULL, rod_score, rod_destroy};

pyod_rod_params pyod_rod_defaults(void) { return (pyod_rod_params){0.1}; }

pyod_detector *pyod_rod_create(const pyod_rod_params *p) {
    rod_impl *m = NEW(rod_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_rod_defaults();
    return detector_new(&ROD_VT, m);
}
