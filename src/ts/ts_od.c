/* Sliding-window wrapper applying any tabular detector to a time series.
 * Port of pyod.models.ts_od.TimeSeriesOD. */
#include "internal.h"

typedef struct {
    pyod_ts_od_params p;
    pyod_detector *inner;
} tsod_impl;

static pyod_detector *default_iforest(uint64_t seed, void *ctx) {
    (void)ctx;
    pyod_iforest_params p = pyod_iforest_defaults();
    p.random_state = seed;
    return pyod_iforest_create(&p);
}

static pyod_status tsod_fit(pyod_detector *d, const pyod_matrix *x) {
    tsod_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    size_t w = m->p.window_size, n = x->rows;
    if (w == 0 || m->p.step == 0 || n < w) return PYOD_ERR_PARAM;
    pyod_free(m->inner);
    pyod_factory f = m->p.detector ? m->p.detector : default_iforest;
    m->inner = f(0, m->p.factory_ctx);
    if (!m->inner) return PYOD_ERR_NOMEM;
    pyod_matrix win;
    size_t nw = ts_windows(x, w, m->p.step, &win);
    if (!nw) return PYOD_ERR_NOMEM;
    pyod_status st = pyod_fit(m->inner, &win);
    pyod_matrix_free(&win);
    CHECK(st);
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    ts_map_scores(pyod_decision_scores(m->inner, NULL), nw, w, m->p.step, n, m->p.score_aggregation, s);
    return ts_fit_state(d, s, n, m->p.contamination);
}

static pyod_status tsod_score(const pyod_detector *d, const pyod_matrix *x, double *out) {
    const tsod_impl *m = d->impl;
    if (x->rows < m->p.window_size) return PYOD_ERR_PARAM;
    pyod_matrix win;
    size_t nw = ts_windows(x, m->p.window_size, m->p.step, &win);
    double *ws = NEW(double, nw);
    pyod_status st = nw && ws ? pyod_decision_function(m->inner, &win, ws) : PYOD_ERR_NOMEM;
    if (st == PYOD_OK) {
        ts_map_scores(ws, nw, m->p.window_size, m->p.step, x->rows, m->p.score_aggregation, out);
        ts_fill(out, x->rows, d->state.threshold);
    }
    free(ws);
    pyod_matrix_free(&win);
    return st;
}

static void tsod_destroy(void *p) {
    tsod_impl *m = p;
    pyod_free(m->inner);
    free(m);
}

static const pyod_vtable TSOD_VT = {"TimeSeriesOD", tsod_fit, NULL, tsod_score, tsod_destroy};

pyod_ts_od_params pyod_ts_od_defaults(void) { return (pyod_ts_od_params){NULL, NULL, 50, 1, PYOD_AGG_MAX, 0.1}; }

pyod_detector *pyod_ts_od_create(const pyod_ts_od_params *p) {
    tsod_impl *m = NEW(tsod_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_ts_od_defaults();
    return detector_new(&TSOD_VT, m);
}
