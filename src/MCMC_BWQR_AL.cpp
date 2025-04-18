// [[Rcpp::depends(RcppArmadillo, GIGrvg)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// call_rgig: raw SEXP from GIGrvg::rgig
SEXP call_rgig(double lambda, double chi, double psi) {
  static Function rgig_fun("rgig", Environment::namespace_env("GIGrvg"));
  return rgig_fun(1, lambda, chi, psi);
}

// Full conditional for beta
arma::vec atualizarBETA(const vec& b,
                        const mat& B,
                        const mat& X,
                        const vec& w,
                        double sigma,
                        double delta2,
                        double theta,
                        const vec& v,
                        const vec& dados) {
  mat B_inv   = inv_sympd(B);
  vec wv      = w / v;
  mat XtWvX   = X.t() * (X.each_col() % wv);
  mat covar   = inv_sympd(B_inv + (1.0/(delta2*sigma)) * XtWvX);
  vec media   = covar * (B_inv * b +
                 (1.0/(delta2*sigma)) * X.t() * ((w % (dados - theta*v)) / v));
  vec z       = randn<vec>(b.n_elem);
  mat L       = chol(covar, "lower");
  return media + L * z;
}

// Full conditional for sigma
double atualizarSIGMA(double c0,
                      double C0,
                      const mat& X,
                      const vec& w,
                      const vec& beta,
                      double tau2,
                      double theta,
                      const vec& v,
                      const vec& dados) {
  int n = dados.n_elem;
  double alpha1 = c0 + 1.5 * n;
  vec resid     = dados - X * beta - theta * v;
  double sum_wv = sum(w % v);
  double quad   = dot((w % resid) / v, resid);
  double beta1  = C0 + sum_wv + quad / (2.0 * tau2);
  double g      = R::rgamma(alpha1, 1.0 / beta1);
  return 1.0 / g;
}

// Full conditional for latent v
arma::vec atualizarV(const vec& dados,
                     const mat& X,
                     const vec& w,
                     const vec& beta,
                     double delta2,
                     double theta,
                     double sigma) {
  int N    = dados.n_elem;
  vec v(N);
  double lambda = 0.5;
  for(int i = 0; i < N; ++i) {
    double resid = dados[i] - dot(X.row(i), beta);
    double chi   = w[i] * resid * resid / (delta2 * sigma);
    double psi   = w[i] * (2.0 / sigma + theta * theta / (delta2 * sigma));
    SEXP s_ans   = call_rgig(lambda, chi, psi);
    NumericVector out(s_ans);
    v[i]        = out[0];
  }
  return v;
}

// [[Rcpp::export]]
List bayesQR_weighted(const vec& y,
                      const mat& X,
                      const vec& w,
                      double tau,
                      int n_mcmc,
                      int burnin,
                      int thin) {
  int n = y.n_elem;
  int p = X.n_cols;
  mat beta_chain(n_mcmc, p);
  vec sigma_chain(n_mcmc);

  // Initial values
  beta_chain.row(0) = solve(X, y).t();
  sigma_chain[0]    = 1.0;
  vec v             = randg<vec>(n, distr_param(2.0, 1.0));

  double delta2 = 2.0 / (tau * (1.0 - tau));
  double theta  = (1.0 - 2.0 * tau) / (tau * (1.0 - tau));
  double c0     = 0.001;
  double C0     = 0.001;
  double tau2   = delta2;

  // MCMC loop
  for(int k = 1; k < n_mcmc; ++k) {
    beta_chain.row(k) = atualizarBETA(zeros<vec>(p),
                                      diagmat(vec(p).fill(1000.0)),
                                      X, w,
                                      sigma_chain[k-1],
                                      delta2, theta,
                                      v, y).t();
    v                = atualizarV(y, X, w,
                                 beta_chain.row(k).t(),
                                 delta2, theta,
                                 sigma_chain[k-1]);
    sigma_chain[k]   = atualizarSIGMA(c0, C0,
                                      X, w,
                                      beta_chain.row(k).t(),
                                      tau2, theta,
                                      v, y);
  }

  // Burnin and thinning
  uvec idx = regspace<uvec>(burnin, thin, n_mcmc - 1);
  int M    = idx.n_elem;

  // Prepare outputs without wrap<>
  NumericMatrix beta_out(M, p);
  for(int i = 0; i < M; ++i) {
    for(int j = 0; j < p; ++j) {
      beta_out(i, j) = beta_chain(idx[i], j);
    }
  }
  NumericVector sigma_out(M);
  for(int i = 0; i < M; ++i) {
    sigma_out[i] = sigma_chain(idx[i]);
  }

  return List::create(
    Named("beta")  = beta_out,
    Named("sigma") = sigma_out
  );
}
