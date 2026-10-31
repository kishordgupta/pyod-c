/* Finite-difference gradient checks for every autodiff op. */
#include "check.h"
#include "deep/ad.h"

typedef ad_tensor *(*fn)(ad_tensor **p);

static double eval(fn f, ad_tensor **p) {
    double v = ad_item(f(p));
    ad_tape_clear();
    return v;
}

/* Max relative error between analytic and central-difference gradients. */
static double gradcheck(fn f, ad_tensor **p, size_t np) {
    ad_zero_grad(p, np);
    ad_backward(f(p));
    ad_tape_clear();
    double worst = 0.0, h = 1e-6;
    for (size_t q = 0; q < np; q++)
        for (size_t i = 0; i < p[q]->size; i++) {
            double orig = p[q]->data[i];
            p[q]->data[i] = orig + h;
            double up = eval(f, p);
            p[q]->data[i] = orig - h;
            double dn = eval(f, p);
            p[q]->data[i] = orig;
            double num = (up - dn) / (2 * h), ana = p[q]->grad[i];
            worst = fmax(worst, fabs(num - ana) / fmax(1e-4, fabs(num) + fabs(ana)));
        }
    return worst;
}

/* Weighted sum so every output element gets a distinct gradient. */
static ad_tensor *wsum(ad_tensor *t) {
    ad_tensor *w = ad_const(t->ndim, t->shape, NULL);
    for (size_t i = 0; i < t->size; i++) w->data[i] = 0.3 + 0.17 * (double)(i % 7) - 0.05 * (double)(i % 3);
    return ad_sum(ad_mul(t, w));
}

static ad_tensor *f_broadcast(ad_tensor **p) { return wsum(ad_div(ad_add(ad_mul(p[0], p[1]), p[2]), ad_add_s(ad_mul(p[2], p[2]), 1.5))); }
static ad_tensor *f_sub(ad_tensor **p) { return wsum(ad_sub(p[0], p[2])); }
static ad_tensor *f_unary(ad_tensor **p) {
    ad_tensor *x = p[0];
    ad_tensor *t = ad_add(ad_tanh(x), ad_sigmoid(x));
    t = ad_add(t, ad_softplus(x));
    t = ad_add(t, ad_gelu(x));
    t = ad_add(t, ad_elu(x));
    t = ad_add(t, ad_cos(x));
    t = ad_add(t, ad_leaky_relu(x, 0.2));
    t = ad_add(t, ad_mul_s(ad_exp(ad_mul_s(x, 0.3)), 0.5));
    t = ad_add(t, ad_log(ad_add_s(ad_pow_s(x, 2.0), 1.0)));
    t = ad_add(t, ad_sqrt(ad_add_s(ad_abs(x), 0.5)));
    t = ad_add(t, ad_relu(x));
    t = ad_add(t, ad_clamp(x, -0.4, 0.6));
    return wsum(t);
}
static ad_tensor *f_matmul(ad_tensor **p) { return wsum(ad_matmul(p[0], ad_transpose(p[1]))); }
static ad_tensor *f_batched(ad_tensor **p) {
    ad_tensor *a = ad_reshape(p[3], 3, AD_SHAPE(2, 3, 4)), *b = ad_swap(a, 1, 2);
    return wsum(ad_softmax(ad_matmul(a, b)));
}
static ad_tensor *f_reduce(ad_tensor **p) {
    ad_tensor *s1 = ad_sum_dim(p[3], 1), *s2 = ad_mean_dim(p[3], 0);
    return ad_add(ad_add(wsum(s1), wsum(s2)), ad_mean(p[3]));
}
static ad_tensor *f_shape(ad_tensor **p) {
    ad_tensor *c = ad_concat(p[0], p[2], 0);
    ad_tensor *n = ad_narrow(c, 1, 1, 2);
    ad_tensor *x3 = ad_reshape(p[3], 3, AD_SHAPE(2, 4, 3));
    return ad_add(wsum(n), wsum(ad_unfold3(x3)));
}
static ad_tensor *f_bce(ad_tensor **p) {
    ad_tensor *t = ad_const(2, p[0]->shape, NULL);
    for (size_t i = 0; i < t->size; i++) t->data[i] = (double)(i % 2);
    return ad_bce_logits(p[0], t);
}

int main(void) {
    pyod_rng r;
    rng_seed(&r, 7);
    ad_tensor *p[4] = {AD_PARAM(3, 4), AD_PARAM(1, 4), AD_PARAM(3, 4), AD_PARAM(6, 4)};
    for (int q = 0; q < 4; q++)
        for (size_t i = 0; i < p[q]->size; i++) p[q]->data[i] = rng_normal(&r);
    struct { const char *name; fn f; } cases[] = {
        {"broadcast", f_broadcast}, {"sub", f_sub}, {"unary", f_unary}, {"matmul", f_matmul},
        {"batched+softmax", f_batched}, {"reduce", f_reduce}, {"shape", f_shape}, {"bce", f_bce},
    };
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        double err = gradcheck(cases[c].f, p, 4);
        CHECK_TRUE(err < 1e-5, "%s gradient error %g", cases[c].name, err);
    }
    CHECK_TRUE(ad_tape_size() == 0, "tape cleared");

    /* Adam minimises a quadratic. */
    ad_tensor *w = AD_PARAM(1, 2);
    ad_optimizer opt;
    CHECK_OK(ad_opt_init(&opt, AD_ADAM, &w, 1, 0.1, 0.0, 0.0));
    for (int it = 0; it < 300; it++) {
        ad_zero_grad(&w, 1);
        ad_tensor *target = AD_CONST(((double[]){3.0, -2.0}), 1, 2);
        ad_backward(ad_sum(ad_pow_s(ad_sub(w, target), 2.0)));
        ad_tape_clear();
        ad_opt_step(&opt);
    }
    CHECK_TRUE(fabs(w->data[0] - 3.0) < 1e-2 && fabs(w->data[1] + 2.0) < 1e-2, "adam converged to (%g, %g)", w->data[0], w->data[1]);
    ad_opt_free(&opt);
    ad_param_free(w);
    for (int q = 0; q < 4; q++) ad_param_free(p[q]);
    CHECK_REPORT("autodiff");
}
