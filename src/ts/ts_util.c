/* Time-series helpers shared by the ts_* detectors. Port of
 * pyod.models._ts_utils plus the FFT and shape-based distance they rely on. */
#include "internal.h"

size_t ts_windows(const pyod_matrix *x, size_t window, size_t step, pyod_matrix *out) {
    size_t n = x->rows, c = x->cols;
    out->data = NULL;
    out->rows = out->cols = 0;
    if (n < window || window == 0 || step == 0) return 0;
    size_t nw = (n - window) / step + 1;
    if (pyod_matrix_alloc(out, nw, window * c) != PYOD_OK) return 0;
    for (size_t i = 0; i < nw; i++) memcpy(&PYOD_AT(out, i, 0), &PYOD_AT(x, i * step, 0), window * c * sizeof(double));
    return nw;
}

void ts_map_scores(const double *ws, size_t nw, size_t window, size_t step, size_t n, pyod_aggregation agg, double *out) {
    size_t *cnt = NEW(size_t, n);
    for (size_t i = 0; i < n; i++) out[i] = NAN;
    if (!cnt) return;
    for (size_t w = 0; w < nw; w++)
        for (size_t j = w * step; j < w * step + window && j < n; j++) {
            if (isnan(out[j])) out[j] = ws[w];
            else out[j] = agg == PYOD_AGG_MAX ? fmax(out[j], ws[w]) : out[j] + ws[w];
            cnt[j]++;
        }
    if (agg == PYOD_AGG_MEAN)
        for (size_t j = 0; j < n; j++)
            if (cnt[j]) out[j] /= (double)cnt[j];
    free(cnt);
}

void ts_combine_channels(double **pc, size_t c, size_t n, pyod_aggregation agg, double *out) {
    if (c == 1) {
        memcpy(out, pc[0], n * sizeof(double));
        return;
    }
    double *filled = NEW(double, n);
    if (!filled) return;
    for (size_t i = 0; i < n; i++) out[i] = agg == PYOD_AGG_MAX ? -INFINITY : 0.0;
    for (size_t ch = 0; ch < c; ch++) {
        for (size_t i = 0; i < n; i++) filled[i] = isnan(pc[ch][i]) ? 0.0 : pc[ch][i];
        double mu = vec_mean(filled, n), sd = vec_std(filled, n);
        for (size_t i = 0; i < n; i++) {
            double z = sd > 0.0 ? (filled[i] - mu) / sd : filled[i] - mu;
            out[i] = agg == PYOD_AGG_MAX ? fmax(out[i], z) : out[i] + z / (double)c;
        }
    }
    for (size_t i = 0; i < n; i++)
        for (size_t ch = 0; ch < c; ch++)
            if (isnan(pc[ch][i])) out[i] = NAN;
    free(filled);
}

void ts_fill(double *s, size_t n, double v) {
    for (size_t i = 0; i < n; i++)
        if (isnan(s[i])) s[i] = v;
}

pyod_status ts_fit_state(pyod_detector *d, double *s, size_t n, double contamination) {
    double *valid = NEW(double, n);
    if (!valid) {
        free(s);
        return PYOD_ERR_NOMEM;
    }
    size_t m = 0;
    for (size_t i = 0; i < n; i++)
        if (!isnan(s[i])) valid[m++] = s[i];
    double th = vec_percentile(valid, m, 100.0 * (1.0 - contamination));
    double mu = vec_mean(valid, m), sigma = vec_std(valid, m);
    free(valid);
    ts_fill(s, n, th);
    CHECK(fit_state_threshold(d, s, n, th));
    d->state.mu = mu;
    d->state.sigma = sigma;
    d->state.contamination = contamination;
    return PYOD_OK;
}

void znormalize(const double *x, size_t n, double *out) {
    double s = vec_std(x, n), m = vec_mean(x, n);
    for (size_t i = 0; i < n; i++) out[i] = s < 1e-10 ? 0.0 : (x[i] - m) / s;
}

double sbd(const double *x, const double *y, size_t m, long *shift) {
    double norm = sqrt(dot(x, x, m)) * sqrt(dot(y, y, m)), best = -INFINITY;
    long best_s = 0;
    if (norm < 1e-10) {
        if (shift) *shift = 0;
        return 2.0;
    }
    for (long s = -(long)m + 1; s < (long)m; s++) {
        double cc = 0.0;
        for (long i = 0; i < (long)m; i++)
            if (i + s >= 0 && i + s < (long)m) cc += x[i + s] * y[i];
        if (cc / norm > best) best = cc / norm, best_s = s;
    }
    if (shift) *shift = best_s;
    return fmin(2.0, fmax(0.0, 1.0 - best));
}

/* In-place iterative radix-2 FFT on interleaved complex data (unscaled). */
static void fft_pow2(double *a, size_t n, int inverse) {
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double t0 = a[2 * i], t1 = a[2 * i + 1];
            a[2 * i] = a[2 * j], a[2 * i + 1] = a[2 * j + 1];
            a[2 * j] = t0, a[2 * j + 1] = t1;
        }
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * PYOD_PI / (double)len * (inverse ? 1.0 : -1.0);
        for (size_t st = 0; st < n; st += len)
            for (size_t k = 0; k < len / 2; k++) {
                double wr = cos(ang * (double)k), wi = sin(ang * (double)k);
                double *u = a + 2 * (st + k), *v = a + 2 * (st + k + len / 2);
                double vr = v[0] * wr - v[1] * wi, vi = v[0] * wi + v[1] * wr;
                v[0] = u[0] - vr, v[1] = u[1] - vi;
                u[0] += vr, u[1] += vi;
            }
    }
}

void dft(const double *in, size_t n, int inverse, double *out) {
    if (n == 0) return;
    if ((n & (n - 1)) == 0) {
        memcpy(out, in, 2 * n * sizeof(double));
        fft_pow2(out, n, inverse);
    } else { /* Bluestein */
        size_t m = 1;
        while (m < 2 * n - 1) m <<= 1;
        double sgn = inverse ? 1.0 : -1.0;
        double *w = NEW(double, 2 * n), *a = NEW(double, 2 * m), *b = NEW(double, 2 * m);
        if (!w || !a || !b) {
            free(w), free(a), free(b);
            return;
        }
        for (size_t k = 0; k < n; k++) {
            double ang = sgn * PYOD_PI * (double)((k * k) % (2 * n)) / (double)n;
            w[2 * k] = cos(ang), w[2 * k + 1] = sin(ang);
            a[2 * k] = in[2 * k] * w[2 * k] - in[2 * k + 1] * w[2 * k + 1];
            a[2 * k + 1] = in[2 * k] * w[2 * k + 1] + in[2 * k + 1] * w[2 * k];
        }
        b[0] = w[0], b[1] = -w[1];
        for (size_t k = 1; k < n; k++) {
            b[2 * k] = b[2 * (m - k)] = w[2 * k];
            b[2 * k + 1] = b[2 * (m - k) + 1] = -w[2 * k + 1];
        }
        fft_pow2(a, m, 0);
        fft_pow2(b, m, 0);
        for (size_t k = 0; k < m; k++) {
            double re = a[2 * k] * b[2 * k] - a[2 * k + 1] * b[2 * k + 1];
            double im = a[2 * k] * b[2 * k + 1] + a[2 * k + 1] * b[2 * k];
            a[2 * k] = re, a[2 * k + 1] = im;
        }
        fft_pow2(a, m, 1);
        for (size_t k = 0; k < n; k++) {
            double cr = a[2 * k] / (double)m, ci = a[2 * k + 1] / (double)m;
            out[2 * k] = cr * w[2 * k] - ci * w[2 * k + 1];
            out[2 * k + 1] = cr * w[2 * k + 1] + ci * w[2 * k];
        }
        free(w), free(a), free(b);
    }
    if (inverse)
        for (size_t k = 0; k < 2 * n; k++) out[k] /= (double)n;
}

/* ---------------- log-mel spectrogram (librosa defaults) ---------------- */

static double hz_to_mel(double f) {
    double f_sp = 200.0 / 3.0, min_log_hz = 1000.0, logstep = log(6.4) / 27.0;
    return f >= min_log_hz ? min_log_hz / f_sp + log(f / min_log_hz) / logstep : f / f_sp;
}

static double mel_to_hz(double m) {
    double f_sp = 200.0 / 3.0, min_log_hz = 1000.0, logstep = log(6.4) / 27.0, min_log_mel = min_log_hz / f_sp;
    return m >= min_log_mel ? min_log_hz * exp(logstep * (m - min_log_mel)) : f_sp * m;
}

pyod_status pyod_log_mel_spectrogram(const double *y, size_t n, double sr, size_t n_mels, size_t hop, pyod_matrix *out) {
    const size_t n_fft = 2048, bins = n_fft / 2 + 1, pad = n_fft / 2;
    if (!y || n == 0 || n_mels == 0 || hop == 0) return PYOD_ERR_PARAM;
    size_t plen = n + 2 * pad, frames = 1 + (plen - n_fft) / hop;
    double *padded = NEW(double, plen), *win = NEW(double, n_fft), *filt = NEW(double, n_mels * bins);
    double *mel_f = NEW(double, n_mels + 2), *buf = NEW(double, 2 * n_fft), *spec = NEW(double, 2 * n_fft), *pw = NEW(double, bins);
    pyod_status st = PYOD_ERR_NOMEM;
    if (!padded || !win || !filt || !mel_f || !buf || !spec || !pw) goto done;
    if ((st = pyod_matrix_alloc(out, n_mels, frames)) != PYOD_OK) goto done;
    memcpy(padded + pad, y, n * sizeof(double));
    for (size_t k = 0; k < n_fft; k++) win[k] = 0.5 - 0.5 * cos(2.0 * PYOD_PI * (double)k / (double)n_fft);
    double lo = hz_to_mel(0.0), hi = hz_to_mel(sr / 2.0);
    for (size_t i = 0; i < n_mels + 2; i++) mel_f[i] = mel_to_hz(lo + (hi - lo) * (double)i / (double)(n_mels + 1));
    for (size_t i = 0; i < n_mels; i++) { /* Slaney-normalised triangles */
        double enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for (size_t b = 0; b < bins; b++) {
            double f = (double)b * (sr / 2.0) / (double)(bins - 1);
            double lower = (f - mel_f[i]) / (mel_f[i + 1] - mel_f[i]);
            double upper = (mel_f[i + 2] - f) / (mel_f[i + 2] - mel_f[i + 1]);
            filt[i * bins + b] = fmax(0.0, fmin(lower, upper)) * enorm;
        }
    }
    double max_db = -INFINITY;
    for (size_t t = 0; t < frames; t++) {
        for (size_t k = 0; k < n_fft; k++) buf[2 * k] = padded[t * hop + k] * win[k], buf[2 * k + 1] = 0.0;
        dft(buf, n_fft, 0, spec);
        for (size_t b = 0; b < bins; b++) pw[b] = spec[2 * b] * spec[2 * b] + spec[2 * b + 1] * spec[2 * b + 1];
        for (size_t i = 0; i < n_mels; i++) {
            double v = 10.0 * log10(fmax(dot(filt + i * bins, pw, bins), 1e-10));
            PYOD_AT(out, i, t) = v;
            max_db = fmax(max_db, v);
        }
    }
    for (size_t i = 0; i < n_mels * frames; i++) out->data[i] = fmax(out->data[i], max_db - 80.0);
    st = PYOD_OK;
done:
    free(padded), free(win), free(filt), free(mel_f), free(buf), free(spec), free(pw);
    return st;
}
