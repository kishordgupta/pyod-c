/* k-Shape (Paparrizos & Gravano, SIGMOD 2015) and SAND (Boniol et al., VLDB
 * 2021) subsequence anomaly detection. Ports of pyod.models.ts_kshape and
 * pyod.models.ts_sand: windows far, in shape-based distance, from every
 * cluster centroid are anomalous. */
#include "internal.h"

/* z-normalised sliding windows of one channel: nw x w. */
static double *znorm_windows(const pyod_matrix *x, size_t ch, size_t w, size_t *nw) {
    size_t n = x->rows;
    *nw = n >= w ? n - w + 1 : 0;
    double *out = NEW(double, *nw * w), *buf = NEW(double, w);
    if (!out || !buf) {
        free(out), free(buf);
        *nw = 0;
        return NULL;
    }
    for (size_t i = 0; i < *nw; i++) {
        for (size_t k = 0; k < w; k++) buf[k] = PYOD_AT(x, i + k, ch);
        znormalize(buf, w, out + i * w);
    }
    free(buf);
    return out;
}

static double nearest_sbd(const double *win, const double *cent, size_t k, size_t w) {
    double best = INFINITY;
    for (size_t c = 0; c < k; c++) best = fmin(best, sbd(win, cent + c * w, w, NULL));
    return best;
}

/* Shape-extraction centroid of aligned members (m x w). */
static void centroid(const double *mem, size_t m, size_t w, double *out) {
    if (m == 1) {
        znormalize(mem, w, out);
        return;
    }
    double *mat = NEW(double, w * w), *vals = NEW(double, w), *vecs = NEW(double, w * w), *mean = NEW(double, w);
    if (!mat || !vals || !vecs || !mean) goto done;
    for (size_t i = 0; i < w; i++)
        for (size_t j = 0; j < w; j++) {
            double q = 0.0;
            for (size_t r = 0; r < m; r++) q += mem[r * w + i] * mem[r * w + j];
            mat[i * w + j] = (double)(i == j) - q / (double)m;
        }
    sym_eigen(mat, w, vals, vecs);
    double *c = vecs + (w - 1) * w; /* smallest eigenvalue */
    for (size_t r = 0; r < m; r++)
        for (size_t j = 0; j < w; j++) mean[j] += mem[r * w + j] / (double)m;
    if (dot(mean, c, w) < 0.0)
        for (size_t j = 0; j < w; j++) c[j] = -c[j];
    znormalize(c, w, out);
done:
    free(mat), free(vals), free(vecs), free(mean);
}

/* Port of _kshape; cent is k x w, dist (optional) gets each window's distance. */
static pyod_status kshape(const double *win, size_t n, size_t w, size_t k, size_t max_iter, pyod_rng *r, double *cent, double *dist) {
    size_t *init = NEW(size_t, k), *labels = NEW(size_t, n), *nl = NEW(size_t, n);
    double *aligned = NEW(double, n * w), *sh = NEW(double, w);
    if (!init || !labels || !nl || !aligned || !sh) {
        free(init), free(labels), free(nl), free(aligned), free(sh);
        return PYOD_ERR_NOMEM;
    }
    rng_sample(r, n, k, init);
    for (size_t c = 0; c < k; c++) znormalize(win + init[c] * w, w, cent + c * w);
    for (size_t i = 0; i < n; i++) labels[i] = SIZE_MAX;
    for (size_t it = 0; it < max_iter; it++) {
        int changed = 0;
        for (size_t i = 0; i < n; i++) {
            double best = INFINITY;
            nl[i] = 0;
            for (size_t c = 0; c < k; c++) {
                double dd = sbd(win + i * w, cent + c * w, w, NULL);
                if (dd < best) best = dd, nl[i] = c;
            }
            changed |= nl[i] != labels[i];
        }
        if (!changed) break;
        memcpy(labels, nl, n * sizeof(size_t));
        for (size_t c = 0; c < k; c++) {
            size_t m = 0;
            for (size_t i = 0; i < n; i++) {
                if (labels[i] != c) continue;
                long s;
                sbd(win + i * w, cent + c * w, w, &s);
                for (long j = 0; j < (long)w; j++) { /* shift by -s, zero-filled */
                    long src = j + s;
                    sh[j] = src >= 0 && src < (long)w ? win[i * w + src] : 0.0;
                }
                znormalize(sh, w, aligned + m++ * w);
            }
            if (m == 0) znormalize(win + rng_below(r, n) * w, w, cent + c * w);
            else centroid(aligned, m, w, cent + c * w);
        }
    }
    if (dist)
        for (size_t i = 0; i < n; i++) dist[i] = nearest_sbd(win + i * w, cent, k, w);
    free(init), free(labels), free(nl), free(aligned), free(sh);
    return PYOD_OK;
}

/* ---------------- k-Shape detector ---------------- */

typedef struct {
    pyod_kshape_params p;
    double *cent; /* channels x k x w */
    size_t channels;
} kshape_impl;

static pyod_status channel_scores(const pyod_matrix *x, size_t ch, const double *cent, size_t k, size_t w, double *out) {
    size_t nw;
    double *win = znorm_windows(x, ch, w, &nw), *d = NEW(double, nw ? nw : 1);
    if (!win || !d) {
        free(win), free(d);
        return PYOD_ERR_NOMEM;
    }
    for (size_t i = 0; i < nw; i++) d[i] = nearest_sbd(win + i * w, cent, k, w);
    ts_map_scores(d, nw, w, 1, x->rows, PYOD_AGG_MAX, out);
    free(win), free(d);
    return PYOD_OK;
}

static pyod_status multi_channel(const pyod_matrix *x, const double *cent, size_t channels, size_t k, size_t w,
                                 pyod_aggregation agg, double *out) {
    double **pc = NEW(double *, channels);
    pyod_status st = pc ? PYOD_OK : PYOD_ERR_NOMEM;
    for (size_t ch = 0; ch < channels && st == PYOD_OK; ch++) {
        pc[ch] = NEW(double, x->rows);
        st = pc[ch] ? channel_scores(x, ch, cent + ch * k * w, k, w, pc[ch]) : PYOD_ERR_NOMEM;
    }
    if (st == PYOD_OK) ts_combine_channels(pc, channels, x->rows, agg, out);
    for (size_t ch = 0; pc && ch < channels; ch++) free(pc[ch]);
    free(pc);
    return st;
}

static pyod_status kshape_fit(pyod_detector *d, const pyod_matrix *x) {
    kshape_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t w = m->p.window_size, k = m->p.n_clusters, c = x->cols;
    if (w < 2 || x->rows < w || x->rows - w + 1 < k || k == 0) return PYOD_ERR_PARAM;
    free(m->cent);
    m->channels = c;
    m->cent = NEW(double, c * k * w);
    double *s = NEW(double, x->rows);
    if (!m->cent || !s) {
        free(s);
        return PYOD_ERR_NOMEM;
    }
    for (size_t ch = 0; ch < c; ch++) {
        size_t nw;
        double *win = znorm_windows(x, ch, w, &nw);
        pyod_rng r;
        rng_seed(&r, m->p.random_state);
        pyod_status st = win ? kshape(win, nw, w, k, m->p.max_iter, &r, m->cent + ch * k * w, NULL) : PYOD_ERR_NOMEM;
        free(win);
        if (st != PYOD_OK) {
            free(s);
            return st;
        }
    }
    pyod_status st = multi_channel(x, m->cent, c, k, w, m->p.channel_aggregation, s);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    return ts_fit_state(d, s, x->rows, m->p.contamination);
}

static pyod_status kshape_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    const kshape_impl *m = d->impl;
    if (x->rows < m->p.window_size) return PYOD_ERR_PARAM;
    CHECK(multi_channel(x, m->cent, m->channels, m->p.n_clusters, m->p.window_size, m->p.channel_aggregation, out));
    ts_fill(out, x->rows, d->state.threshold);
    return PYOD_OK;
}

static void kshape_destroy(void *p) {
    kshape_impl *m = p;
    free(m->cent), free(m);
}

static const pyod_vtable KSHAPE_VT = {"KShape", kshape_fit, NULL, kshape_score, kshape_destroy};

pyod_kshape_params pyod_kshape_defaults(void) { return (pyod_kshape_params){3, 50, 100, PYOD_AGG_MAX, 0.1, 42}; }

pyod_detector *pyod_kshape_create(const pyod_kshape_params *p) {
    kshape_impl *m = NEW(kshape_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_kshape_defaults();
    return detector_new(&KSHAPE_VT, m);
}

/* ---------------- SAND ---------------- */

typedef struct {
    pyod_sand_params p;
    double *cent;
    size_t channels;
} sand_impl;

/* Port of _streaming_fit_channel. */
static pyod_status sand_stream(const sand_impl *m, const double *win, size_t n, double *cent, double *scores) {
    size_t w = m->p.window_size, k = m->p.n_clusters, bs = m->p.batch_size < n ? m->p.batch_size : n;
    double *bc = NEW(double, k * w), *upd = NEW(double, w);
    if (!bc || !upd) {
        free(bc), free(upd);
        return PYOD_ERR_NOMEM;
    }
    pyod_rng r;
    rng_seed(&r, m->p.random_state);
    pyod_status st = kshape(win, bs, w, k, m->p.max_iter, &r, cent, NULL);
    for (size_t pos = 0; st == PYOD_OK && pos < n;) {
        size_t end = pos + bs < n ? pos + bs : n;
        for (size_t i = pos; i < end; i++) scores[i] = nearest_sbd(win + i * w, cent, k, w);
        if (end < n && end - pos >= k) {
            rng_seed(&r, m->p.random_state);
            if ((st = kshape(win + pos * w, end - pos, w, k, m->p.max_iter, &r, bc, NULL)) != PYOD_OK) break;
            for (size_t c = 0; c < k; c++) {
                for (size_t j = 0; j < w; j++) upd[j] = m->p.alpha * bc[c * w + j] + (1.0 - m->p.alpha) * cent[c * w + j];
                znormalize(upd, w, cent + c * w);
            }
        }
        pos = end;
    }
    free(bc), free(upd);
    return st;
}

static pyod_status sand_fit(pyod_detector *d, const pyod_matrix *x) {
    sand_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t w = m->p.window_size, k = m->p.n_clusters, c = x->cols, n = x->rows;
    if (w < 2 || n < w + 1 || k == 0) return PYOD_ERR_PARAM;
    size_t first = n - w + 1 < m->p.batch_size ? n - w + 1 : m->p.batch_size;
    if (first < k) return PYOD_ERR_PARAM;
    free(m->cent);
    m->channels = c;
    m->cent = NEW(double, c * k * w);
    double **pc = NEW(double *, c), *s = NEW(double, n);
    pyod_status st = m->cent && pc && s ? PYOD_OK : PYOD_ERR_NOMEM;
    for (size_t ch = 0; ch < c && st == PYOD_OK; ch++) {
        size_t nw;
        double *win = znorm_windows(x, ch, w, &nw), *ws = NEW(double, nw ? nw : 1);
        pc[ch] = NEW(double, n);
        st = win && ws && pc[ch] ? sand_stream(m, win, nw, m->cent + ch * k * w, ws) : PYOD_ERR_NOMEM;
        if (st == PYOD_OK) ts_map_scores(ws, nw, w, 1, n, PYOD_AGG_MAX, pc[ch]);
        free(win), free(ws);
    }
    if (st == PYOD_OK) ts_combine_channels(pc, c, n, m->p.channel_aggregation, s);
    for (size_t ch = 0; pc && ch < c; ch++) free(pc[ch]);
    free(pc);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    return ts_fit_state(d, s, n, m->p.contamination);
}

static pyod_status sand_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    const sand_impl *m = d->impl;
    if (x->rows < m->p.window_size) return PYOD_ERR_PARAM;
    CHECK(multi_channel(x, m->cent, m->channels, m->p.n_clusters, m->p.window_size, m->p.channel_aggregation, out));
    ts_fill(out, x->rows, d->state.threshold);
    return PYOD_OK;
}

static void sand_destroy(void *p) {
    sand_impl *m = p;
    free(m->cent), free(m);
}

static const pyod_vtable SAND_VT = {"SAND", sand_fit, NULL, sand_score, sand_destroy};

pyod_sand_params pyod_sand_defaults(void) { return (pyod_sand_params){50, 5, 100, 50, 0.5, PYOD_AGG_MAX, 0.1, 42}; }

pyod_detector *pyod_sand_create(const pyod_sand_params *p) {
    sand_impl *m = NEW(sand_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_sand_defaults();
    return detector_new(&SAND_VT, m);
}
