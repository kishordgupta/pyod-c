/* Numerical core: special functions, statistics, linear algebra, RNG. */
#include "check.h"
#include "../src/internal.h"

static double rel(double a, double b) { return fabs(a - b) / fmax(1e-300, fabs(b)); }

int main(void) {
    /* np.percentile / scipy reference values */
    double v[4] = {1, 2, 3, 4};
    CHECK_TRUE(fabs(vec_percentile(v, 4, 50) - 2.5) < 1e-12, "median");
    CHECK_TRUE(fabs(vec_percentile(v, 4, 90) - 3.7) < 1e-12, "p90");
    CHECK_TRUE(fabs(erf_(1.0) - 0.8427007929497149) < 1e-14, "erf");
    CHECK_TRUE(fabs(erfc_(5.0) - 1.5374597944280348e-12) < 1e-24, "erfc tail");
    CHECK_TRUE(fabs(lgamma_(5.0) - log(24.0)) < 1e-12, "lgamma");
    CHECK_TRUE(fabs(chi2_ppf(0.5, 2.0) - 1.3862943611198906) < 1e-10, "chi2 median");
    CHECK_TRUE(fabs(chi2_ppf(0.975, 4.0) - 11.143286781877796) < 1e-9, "chi2 0.975");
    CHECK_TRUE(fabs(norm_ppf(0.975) - 1.959963984540054) < 1e-12, "norm_ppf");

    /* ECDF ties share the highest rank */
    double e_in[4] = {1, 0, 1, 2}, e_out[4];
    pyod_matrix em = pyod_matrix_view(e_in, 4, 1);
    column_ecdf(&em, e_out);
    CHECK_TRUE(e_out[0] == 0.75 && e_out[1] == 0.25 && e_out[2] == 0.75 && e_out[3] == 1.0, "ecdf");

    /* eigen-decomposition reconstructs A v = lambda v */
    pyod_rng r;
    rng_seed(&r, 3);
    size_t sizes[4] = {1, 2, 5, 30};
    for (int s = 0; s < 4; s++) {
        size_t n = sizes[s];
        double *a = NEW(double, n * n), *vals = NEW(double, n), *vecs = NEW(double, n * n);
        for (size_t i = 0; i < n; i++)
            for (size_t j = i; j < n; j++) a[i * n + j] = a[j * n + i] = rng_normal(&r);
        sym_eigen(a, n, vals, vecs);
        double worst = 0.0;
        for (size_t t = 0; t < n; t++) {
            if (t + 1 < n) CHECK_TRUE(vals[t] >= vals[t + 1], "eigenvalues descending");
            for (size_t i = 0; i < n; i++) worst = fmax(worst, fabs(dot(a + i * n, vecs + t * n, n) - vals[t] * vecs[t * n + i]));
        }
        CHECK_TRUE(worst < 1e-9, "eigen residual %g (n=%zu)", worst, n);
        free(a), free(vals), free(vecs);
    }

    /* linear solve and singular detection */
    double A[9] = {0, 2, 1, 1, 1, 0, 3, 0, 1}, B[3] = {5, 3, 6}, X[3];
    CHECK_TRUE(lin_solve(A, B, X, 3, 1) && rel(X[0], 1.4) < 1e-12 && rel(X[1], 1.6) < 1e-12 && rel(X[2], 1.8) < 1e-12, "solve");
    double S[4] = {1, 2, 2, 4};
    CHECK_TRUE(!lin_solve(S, B, X, 2, 1), "singular");

    /* SplitMix64 + Box-Muller identical to the Rust port */
    rng_seed(&r, 42);
    CHECK_TRUE(rng_u64(&r) == 13679457532755275413ULL, "splitmix64 first draw");

    /* numba argsort with many ties, including a pivot at index 0 */
    uint32_t keys[40];
    size_t idx[40];
    for (int i = 0; i < 40; i++) keys[i] = (uint32_t)(i % 3 == 0 ? 0 : 5 + i % 2);
    numba_argsort_u32(keys, 40, idx);
    for (int i = 1; i < 40; i++) CHECK_TRUE(keys[idx[i - 1]] <= keys[idx[i]], "argsort order at %d", i);

    CHECK_REPORT("core");
}
