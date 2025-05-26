// [[Rcpp::depends(RcppArmadillo, GIGrvg)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
#include <Rmath.h>
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

  // Prepare outputs
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

// [[Rcpp::export]]
List bayesQR_EM(const vec& y,
                const mat& X,
                const vec& w,
                const NumericVector& tau_vec,
                const vec& mu0,
                const mat& sigma0,
                double a0_prior,
                double b0_prior) {
  int p = X.n_cols;
  int q = tau_vec.size();
  int n = y.n_elem;

  // Precompute auxiliary constants
  vec delta2(q), theta(q);
  for(int k = 0; k < q; ++k) {
    delta2[k] = 2.0 / (tau_vec[k] * (1.0 - tau_vec[k]));
    theta[k]  = (1.0 - 2.0 * tau_vec[k]) / (tau_vec[k] * (1.0 - tau_vec[k]));
  }

  // Initialize storage
  mat beta_hist(1, p * q);
  vec sigma_hist(1);

  // 1) Initial quantile regression via quantreg::rq.fit
  Environment quantreg = Environment::namespace_env("quantreg");
  Function rq_fit = quantreg["rq.fit"];
  NumericMatrix X_wrap = wrap(X);
  NumericVector y_wrap = wrap(y);
  List fit = rq_fit(_["x"] = X_wrap, _["y"] = y_wrap, _["tau"] = tau_vec);
  NumericMatrix coefs = fit["coefficients"]; // p x q
  for(int k = 0; k < q; ++k)
    for(int j = 0; j < p; ++j)
      beta_hist(0, k*p + j) = coefs(j, k);
  sigma_hist[0] = 1.0;

  // 2) Initial beta from bayesQR_weighted
  List init = bayesQR_weighted(y, X, w, tau_vec[0], 15000, 5000, 5);
  NumericMatrix beta_mcmc = init["beta"];
  NumericVector initial_beta = beta_mcmc(0, _);

  // EM iterations (E- and M-steps)
  double criteria = 1e-4;
  vec aux_crit(p*q + q, 1.0);
  int iter = 1;
  while(arma::sum(aux_crit > criteria) > 0) {
    vec sigma_aux(q);

    // Expand cov_inv and y_star across quantiles
    vec cov_inv_all;
    vec y_star_all;

    for(int k = 0; k < q; ++k) {
      // E-step: draw latent variables
      vec resid = y - X * beta_hist.row(iter-1).cols(k*p, k*p + p - 1).t();
      vec a_star = (w % (resid % resid)) / (delta2[k] * sigma_hist[iter-1]);
      vec b_star = w * (2.0 / sigma_hist[iter-1] + (theta[k]*theta[k]) / (delta2[k] * sigma_hist[iter-1]));
      a_star.elem(find(a_star == 0)).fill(1e-10);
      vec inv_nu = sqrt(b_star) / sqrt(a_star);
      vec aux = sqrt(a_star % b_star);
      vec e_nu(n);
      for(int i = 0; i < n; ++i)
        e_nu[i] = (1.0 / inv_nu[i]) * (R::bessel_k(aux[i], 1.5, 0) / R::bessel_k(aux[i], 0.5, 0));

      cov_inv_all.insert_rows(cov_inv_all.n_rows,
                              inv_nu * (1.0 / (delta2[k] * sigma_hist[iter-1])));
      y_star_all.insert_rows(y_star_all.n_rows,
                             y - theta[k] / inv_nu);

      // M-step: update sigma
      vec diff = y - theta[k]/inv_nu - X * beta_hist.row(iter-1).cols(k*p, k*p + p - 1).t();
      double num = arma::sum((w % inv_nu % (diff % diff)) / (2.0 * delta2[k]))
                   + arma::sum((theta[k]*theta[k]) * (e_nu - 1.0/inv_nu) / (2.0 * delta2[k]))
                   + arma::sum(e_nu) + b0_prior;
      double den = (3.0*n + a0_prior + 1.0)/2.0;
      sigma_aux[k] = num / den;
    }

    sigma_hist.insert_rows(iter, sigma_aux.t());

    // M-step: update beta (QP solve placeholder)
    beta_hist.insert_rows(iter, beta_hist.row(iter-1));

    // Check convergence
    aux_crit.subvec(0, p*q - 1) = square(beta_hist.row(iter) - beta_hist.row(iter-1));
    aux_crit.subvec(p*q, p*q + q - 1) = square(sigma_hist.row(iter) - sigma_hist.row(iter-1));
    ++iter;
  }

  // Return last estimates
  return List::create(
    Named("beta")  = wrap(beta_hist.row(iter-1)),
    Named("sigma") = wrap(sigma_hist.row(iter-1))
  );
}