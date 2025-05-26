#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

using arma::vec;
using arma::mat;

namespace Rwrap {
  static Rcpp::Function rmvnorm(Rcpp::Environment::namespace_env("mvtnorm")["rmvnorm"]);
  static Rcpp::Function rgig   (Rcpp::Environment::namespace_env("GIGrvg")["rgig"]);
}

// ───── Random helpers via R packages ──────────────────────────────────────────
inline vec rmvnorm_cpp(const vec& mean, const mat& Sigma) {
  Rcpp::NumericMatrix res = Rwrap::rmvnorm(1, mean, Sigma);
  return vec(res.begin(), Sigma.n_cols, false);
}

inline double rgig_cpp(double lambda, double chi, double psi) {
  Rcpp::NumericVector v = Rwrap::rgig(1, lambda, chi, psi);
  return v[0];
}

inline double rinv_gamma(double a, double b) {
  return 1.0 / R::rgamma(a, 1.0 / b); // R shape/scale
}

// ───── Full conditionals ─────────────────────────────────────────────────────
vec atualizarBETA(const vec& b, const mat& B,
                  const mat& X, const vec& w,
                  double sigma, double delta2, double theta,
                  const vec& v, const vec& y) {
  mat  B_inv = arma::inv_sympd(B);
  vec  w_over_v = w / v;
  mat  XtWX = X.t() * arma::diagmat(w_over_v) * X;
  mat  covar = arma::inv_sympd(B_inv + (1.0 / (delta2 * sigma)) * XtWX);
  vec  media = covar * (B_inv * b + (1.0 / (delta2 * sigma)) * X.t() * arma::diagmat(w_over_v) * (y - theta * v));
  return rmvnorm_cpp(media, covar);
}

double atualizarSIGMA(double c, double C,
                      const mat& X, const vec& w,
                      const vec& beta, double tau2,
                      double theta, const vec& v,
                      const vec& y, unsigned int n) {
  double alpha1 = c + 1.5 * static_cast<double>(n);
  vec resid = y - X * beta - theta * v;
  vec term  = w % arma::square(resid) / v;
  double beta1 = C + arma::dot(w, v) + arma::sum(term) / (2.0 * tau2);
  return rinv_gamma(alpha1, beta1);
}

vec atualizarV(const vec& y, const mat& X, const vec& w,
               const vec& beta, double delta2, double theta,
               double sigma) {
  unsigned int N = y.n_elem;
  vec v(N);
  vec resid = y - X * beta;
  vec p2 = w % arma::square(resid) / (delta2 * sigma);
  vec p3 = w * (2.0 / sigma + std::pow(theta, 2) / (delta2 * sigma));
  for (unsigned int i = 0; i < N; ++i) v[i] = rgig_cpp(0.5, p2[i], p3[i]);
  return v;
}

// ───── MCMC core ─────────────────────────────────────────────────────────────
// [[Rcpp::export(name = "bayesQR_weighted_cpp")]]
Rcpp::List bayesQR_weighted(const vec& y, const mat& X, const vec& w,
                           double tau, unsigned int n_mcmc,
                           unsigned int burnin, unsigned int thin) {
  unsigned int n = y.n_elem;
  unsigned int p = X.n_cols;
  unsigned int kept = (n_mcmc > burnin) ? (n_mcmc - burnin + thin - 1) / thin : 0;

  mat beta_store(kept, p);
  vec sigma_store(kept);

  vec beta_curr = arma::solve(X, y);      // OLS start
  double sigma_curr = 1.0;
  vec v = vec(n, arma::fill::ones);

  double delta2 = 2.0 / (tau * (1.0 - tau));
  double theta  = (1.0 - 2.0 * tau) / (tau * (1.0 - tau));

  unsigned int idx = 0;
  for (unsigned int k = 1; k < n_mcmc; ++k) {
    beta_curr  = atualizarBETA(vec(p, arma::fill::zeros), 1000.0 * arma::eye(p, p),
                               X, w, sigma_curr, delta2, theta, v, y);
    v          = atualizarV(y, X, w, beta_curr,
                            delta2, theta, /*R passes sigma = 1*/ 1.0);
    sigma_curr = atualizarSIGMA(0.001, 0.001, X, w, beta_curr, delta2, theta, v, y, n);

    if (k >= burnin && ((k - burnin) % thin == 0)) {
      beta_store.row(idx) = beta_curr.t();
      sigma_store(idx)    = sigma_curr;
      ++idx;
    }
  }

  return Rcpp::List::create(Rcpp::Named("beta")  = beta_store,
                            Rcpp::Named("sigma") = sigma_store);
}
