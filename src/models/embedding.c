/* Embedding-based outlier detection for unstructured data. Port of
 * pyod.models.embedding (EmbeddingOD and MultiModalOD): an encoder turns raw
 * items into vectors, which are optionally standardised and PCA-reduced before
 * a regular detector scores them. PyOD ships encoders backed by
 * sentence-transformers, HuggingFace and OpenAI; here you supply the encoder. */
#include "internal.h"

struct pyod_embedding_od {
    pyod_encoder enc;
    void *ctx;
    pyod_detector *det;
    pyod_embedding_params p;
    std_scaler scaler;
    int have_scaler;
    double *pca_mean, *pca_axes; /* d, k x d */
    size_t dim, k;
};

pyod_embedding_params pyod_embedding_defaults(void) { return (pyod_embedding_params){1, 0}; }

pyod_embedding_od *pyod_embedding_create(pyod_encoder enc, void *enc_ctx, pyod_detector *detector,
                                         const pyod_embedding_params *p) {
    if (!enc) return NULL;
    pyod_embedding_od *e = NEW(pyod_embedding_od, 1);
    if (!e) {
        pyod_free(detector);
        return NULL;
    }
    e->enc = enc;
    e->ctx = enc_ctx;
    e->det = detector ? detector : pyod_knn_create(NULL);
    e->p = p ? *p : pyod_embedding_defaults();
    if (!e->det) {
        free(e);
        return NULL;
    }
    return e;
}

/* Encode and preprocess into `out` (allocated). */
static pyod_status embed(const pyod_embedding_od *e, const void *items, size_t n, pyod_matrix *out) {
    pyod_matrix raw = {0};
    CHECK(e->enc(items, n, &raw, e->ctx));
    if (raw.rows != n || !raw.data) {
        pyod_matrix_free(&raw);
        return PYOD_ERR_DIMENSION;
    }
    if (e->have_scaler) {
        if (raw.cols != e->dim) {
            pyod_matrix_free(&raw);
            return PYOD_ERR_DIMENSION;
        }
        scaler_transform(&e->scaler, &raw, raw.data);
    }
    if (!e->pca_axes) {
        *out = raw;
        return PYOD_OK;
    }
    pyod_status st = pyod_matrix_alloc(out, n, e->k);
    if (st == PYOD_OK)
        for (size_t i = 0; i < n; i++)
            for (size_t c = 0; c < e->k; c++) {
                double s = 0.0;
                for (size_t j = 0; j < e->dim; j++) s += e->pca_axes[c * e->dim + j] * (PYOD_AT(&raw, i, j) - e->pca_mean[j]);
                PYOD_AT(out, i, c) = s;
            }
    pyod_matrix_free(&raw);
    return st;
}

pyod_status pyod_embedding_fit(pyod_embedding_od *e, const void *items, size_t n) {
    if (!e || n == 0) return PYOD_ERR_EMPTY_INPUT;
    if (e->have_scaler) scaler_free(&e->scaler);
    e->have_scaler = 0;
    free(e->pca_mean), free(e->pca_axes);
    e->pca_mean = e->pca_axes = NULL;
    pyod_matrix raw;
    CHECK(embed(e, items, n, &raw));
    e->dim = raw.cols;
    pyod_status st = PYOD_OK;
    if (e->p.standardize) {
        if ((st = scaler_fit(&e->scaler, &raw)) == PYOD_OK) {
            e->have_scaler = 1;
            scaler_transform(&e->scaler, &raw, raw.data);
        }
    }
    if (st == PYOD_OK && e->p.reduce_dim) {
        size_t d = raw.cols, k = e->p.reduce_dim;
        double *cov = NEW(double, d * d), *vals = NEW(double, d), *vecs = NEW(double, d * d);
        e->pca_mean = NEW(double, d);
        e->pca_axes = NEW(double, k * d);
        if (k > d) st = PYOD_ERR_PARAM;
        else if (!cov || !vals || !vecs || !e->pca_mean || !e->pca_axes) st = PYOD_ERR_NOMEM;
        else {
            col_mean(&raw, e->pca_mean);
            covariance(&raw, e->pca_mean, 1, cov);
            sym_eigen(cov, d, vals, vecs);
            memcpy(e->pca_axes, vecs, k * d * sizeof(double));
            e->k = k;
        }
        free(cov), free(vals), free(vecs);
        if (st == PYOD_OK) {
            pyod_matrix red;
            if ((st = pyod_matrix_alloc(&red, n, k)) == PYOD_OK) {
                for (size_t i = 0; i < n; i++)
                    for (size_t c = 0; c < k; c++) {
                        double s = 0.0;
                        for (size_t j = 0; j < d; j++) s += e->pca_axes[c * d + j] * (PYOD_AT(&raw, i, j) - e->pca_mean[j]);
                        PYOD_AT(&red, i, c) = s;
                    }
                pyod_matrix_free(&raw);
                raw = red;
            }
        } else {
            free(e->pca_mean), free(e->pca_axes);
            e->pca_mean = e->pca_axes = NULL;
        }
    }
    if (st == PYOD_OK) st = pyod_fit(e->det, &raw);
    pyod_matrix_free(&raw);
    return st;
}

pyod_status pyod_embedding_decision_function(const pyod_embedding_od *e, const void *items, size_t n, double *out) {
    if (!e) return PYOD_ERR_PARAM;
    if (!e->det->fitted) return PYOD_ERR_NOT_FITTED;
    pyod_matrix x;
    CHECK(embed(e, items, n, &x));
    pyod_status st = pyod_decision_function(e->det, &x, out);
    pyod_matrix_free(&x);
    return st;
}

pyod_status pyod_embedding_predict(const pyod_embedding_od *e, const void *items, size_t n, uint8_t *labels) {
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    pyod_status st = pyod_embedding_decision_function(e, items, n, s);
    if (st == PYOD_OK)
        for (size_t i = 0; i < n; i++) labels[i] = s[i] > e->det->state.threshold;
    free(s);
    return st;
}

const pyod_detector *pyod_embedding_detector(const pyod_embedding_od *e) { return e ? e->det : NULL; }

void pyod_embedding_free(pyod_embedding_od *e) {
    if (!e) return;
    pyod_free(e->det);
    if (e->have_scaler) scaler_free(&e->scaler);
    free(e->pca_mean), free(e->pca_axes), free(e);
}

/* ---------------- MultiModalOD ---------------- */

struct pyod_multimodal_od {
    pyod_embedding_od **mods;
    size_t n;
    pyod_combination comb;
    int standardize;
    double contamination;
    double *mu, *sd; /* per-modality score standardisation */
    pyod_detector state_holder; /* fit state only */
};

pyod_multimodal_od *pyod_multimodal_create(pyod_embedding_od **modalities, size_t n, pyod_combination comb,
                                           int standardize_scores, double contamination) {
    pyod_multimodal_od *m = NEW(pyod_multimodal_od, 1);
    if (!m) return NULL;
    m->mods = NEW(pyod_embedding_od *, n);
    m->mu = NEW(double, n);
    m->sd = NEW(double, n);
    if (!m->mods || !m->mu || !m->sd) {
        free(m->mods), free(m->mu), free(m->sd), free(m);
        return NULL;
    }
    memcpy(m->mods, modalities, n * sizeof(pyod_embedding_od *));
    m->n = n;
    m->comb = comb;
    m->standardize = standardize_scores;
    m->contamination = contamination;
    m->state_holder.state.threshold = NAN;
    return m;
}

static void mm_combine(const pyod_multimodal_od *m, pyod_matrix *cols, double *out) {
    if (m->standardize)
        for (size_t i = 0; i < cols->rows; i++)
            for (size_t e = 0; e < m->n; e++) PYOD_AT(cols, i, e) = (PYOD_AT(cols, i, e) - m->mu[e]) / m->sd[e];
    if (m->comb == PYOD_COMBINE_MAX) pyod_comb_maximization(cols, out);
    else pyod_comb_average(cols, NULL, out);
}

pyod_status pyod_multimodal_fit(pyod_multimodal_od *m, const void *items, size_t n) {
    if (!m || m->n == 0) return PYOD_ERR_PARAM;
    CHECK(check_contamination(m->contamination));
    pyod_matrix cols;
    CHECK(pyod_matrix_alloc(&cols, n, m->n));
    pyod_status st = PYOD_OK;
    for (size_t e = 0; e < m->n && st == PYOD_OK; e++) {
        if ((st = pyod_embedding_fit(m->mods[e], items, n)) != PYOD_OK) break;
        const double *s = pyod_decision_scores(m->mods[e]->det, NULL);
        for (size_t i = 0; i < n; i++) PYOD_AT(&cols, i, e) = s[i];
        double sd = vec_std(s, n);
        m->mu[e] = vec_mean(s, n);
        m->sd[e] = sd == 0.0 ? 1.0 : sd;
    }
    double *out = NEW(double, n);
    if (st == PYOD_OK && !out) st = PYOD_ERR_NOMEM;
    if (st == PYOD_OK) {
        mm_combine(m, &cols, out);
        st = fit_state_contamination(&m->state_holder, out, n, m->contamination);
        out = NULL;
    }
    free(out);
    pyod_matrix_free(&cols);
    return st;
}

pyod_status pyod_multimodal_decision_function(const pyod_multimodal_od *m, const void *items, size_t n, double *out) {
    if (!m) return PYOD_ERR_PARAM;
    if (!m->state_holder.fitted) return PYOD_ERR_NOT_FITTED;
    pyod_matrix cols;
    CHECK(pyod_matrix_alloc(&cols, n, m->n));
    double *buf = NEW(double, n);
    pyod_status st = buf ? PYOD_OK : PYOD_ERR_NOMEM;
    for (size_t e = 0; e < m->n && st == PYOD_OK; e++) {
        st = pyod_embedding_decision_function(m->mods[e], items, n, buf);
        for (size_t i = 0; st == PYOD_OK && i < n; i++) PYOD_AT(&cols, i, e) = buf[i];
    }
    if (st == PYOD_OK) mm_combine(m, &cols, out);
    free(buf);
    pyod_matrix_free(&cols);
    return st;
}

pyod_status pyod_multimodal_predict(const pyod_multimodal_od *m, const void *items, size_t n, uint8_t *labels) {
    double *s = NEW(double, n);
    if (!s) return PYOD_ERR_NOMEM;
    pyod_status st = pyod_multimodal_decision_function(m, items, n, s);
    if (st == PYOD_OK)
        for (size_t i = 0; i < n; i++) labels[i] = s[i] > m->state_holder.state.threshold;
    free(s);
    return st;
}

const double *pyod_multimodal_scores(const pyod_multimodal_od *m, size_t *n) {
    if (!m || !m->state_holder.fitted) return NULL;
    if (n) *n = m->state_holder.state.n;
    return m->state_holder.state.scores;
}

void pyod_multimodal_free(pyod_multimodal_od *m) {
    if (!m) return;
    for (size_t e = 0; e < m->n; e++) pyod_embedding_free(m->mods[e]);
    fit_state_clear(&m->state_holder.state);
    free(m->mods), free(m->mu), free(m->sd), free(m);
}
