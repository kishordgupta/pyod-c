/* Spectral Residual saliency for time series (Ren et al., KDD 2019).
 * Port of pyod.models.ts_spectral_residual. */
#include "internal.h"

typedef struct {
    pyod_spectral_residual_params p;
} sr_impl;

static pyod_status saliency(const double *x, size_t n, size_t q, double *out) {
    double *in = NEW(double, 2 * n), *f = NEW(double, 2 * n), *amp = NEW(double, n), *ph = NEW(double, n), *back = NEW(double, 2 * n);
    if (!in || !f || !amp || !ph || !back) {
        free(in), free(f), free(amp), free(ph), free(back);
        return PYOD_ERR_NOMEM;
    }
    for (size_t i = 0; i < n; i++) in[2 * i] = x[i], in[2 * i + 1] = 0.0;
    dft(in, n, 0, f);
    for (size_t i = 0; i < n; i++) {
        amp[i] = log(hypot(f[2 * i], f[2 * i + 1]) + 1e-10);
        ph[i] = atan2(f[2 * i + 1], f[2 * i]);
    }
    size_t off = (q - 1) / 2; /* np.convolve(amp, ones(q) / q, mode='same') */
    for (size_t i = 0; i < n; i++) {
        size_t k = i + off;
        double sm = 0.0;
        for (size_t j = 0; j < q; j++)
            if (k >= j && k - j < n) sm += amp[k - j];
        double mag = exp(amp[i] - sm / (double)q);
        in[2 * i] = mag * cos(ph[i]);
        in[2 * i + 1] = mag * sin(ph[i]);
    }
    dft(in, n, 1, back);
    for (size_t i = 0; i < n; i++) out[i] = hypot(back[2 * i], back[2 * i + 1]);
    free(in), free(f), free(amp), free(ph), free(back);
    return PYOD_OK;
}

static pyod_status sr_scores(const sr_impl *m, const pyod_matrix *x, double *out) {
    size_t n = x->rows, c = x->cols;
    double *col = NEW(double, n), **pc = NEW(double *, c);
    pyod_status st = col && pc ? PYOD_OK : PYOD_ERR_NOMEM;
    for (size_t ch = 0; ch < c && st == PYOD_OK; ch++) {
        pc[ch] = NEW(double, n);
        if (!pc[ch]) {
            st = PYOD_ERR_NOMEM;
            break;
        }
        for (size_t i = 0; i < n; i++) col[i] = PYOD_AT(x, i, ch);
        st = saliency(col, n, m->p.score_window, pc[ch]);
    }
    if (st == PYOD_OK) ts_combine_channels(pc, c, n, m->p.channel_aggregation, out);
    for (size_t ch = 0; pc && ch < c; ch++) free(pc[ch]);
    free(pc), free(col);
    return st;
}

static pyod_status sr_fit(pyod_detector *d, const pyod_matrix *x) {
    sr_impl *m = d->impl;
    CHECK(check_contamination(m->p.contamination));
    if (m->p.score_window == 0 || x->rows < (m->p.score_window > 2 ? m->p.score_window : 2)) return PYOD_ERR_PARAM;
    double *s = NEW(double, x->rows);
    if (!s) return PYOD_ERR_NOMEM;
    pyod_status st = sr_scores(m, x, s);
    if (st != PYOD_OK) {
        free(s);
        return st;
    }
    return fit_state_contamination(d, s, x->rows, m->p.contamination);
}

static pyod_status sr_score(const pyod_detector *d, const pyod_matrix *x, double *out) { return sr_scores(d->impl, x, out); }

static const pyod_vtable SR_VT = {"SpectralResidual", sr_fit, NULL, sr_score, free};

pyod_spectral_residual_params pyod_spectral_residual_defaults(void) {
    return (pyod_spectral_residual_params){3, PYOD_AGG_MAX, 0.1};
}

pyod_detector *pyod_spectral_residual_create(const pyod_spectral_residual_params *p) {
    sr_impl *m = NEW(sr_impl, 1);
    if (!m) return NULL;
    m->p = p ? *p : pyod_spectral_residual_defaults();
    return detector_new(&SR_VT, m);
}
