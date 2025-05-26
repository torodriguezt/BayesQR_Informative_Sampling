// bayesQR_all.cpp
// [[Rcpp::depends(RcppArmadillo, GIGrvg)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// Wrapper para GIGrvg::rgig()
SEXP call_rgig(double lambda, double chi, double psi) {
  static Function rgig_f("rgig", Environment::namespace_env("GIGrvg"));
  return rgig_f(1, _["lambda"]=lambda, _["chi"]=chi, _["psi"]=psi);
}

// Full conditional de beta
arma::rowvec atualizarBETA(const arma::rowvec& b,
                           const arma::mat& B,
                           const arma::mat& X,
                           const arma::vec& w,
                           double sigma,
                           double delta2,
                           double theta,
                           const arma::vec& v,
                           const arma::vec& y) {
  arma::mat Binv   = inv_sympd(B);
  arma::vec wv     = w / v;
  arma::mat XtWvX  = X.t() * diagmat(wv) * X;
  arma::mat covar  = inv_sympd(Binv + (1.0/(delta2*sigma)) * XtWvX);
  arma::vec mu     = covar * ( Binv * b.t()
                            + (1.0/(delta2*sigma)) * X.t() * ((w % (y - theta*v)) / v) );
  arma::vec z      = randn<arma::vec>(b.n_elem);
  arma::mat L      = chol(covar, "lower");
  arma::vec out    = mu + L*z;
  return out.t();
}

// Full conditional de sigma
double atualizarSIGMA(double c0,
                      double C0,
                      const arma::mat& X,
                      const arma::vec& w,
                      const arma::rowvec& beta,
                      double tau2,
                      double theta,
                      const arma::vec& v,
                      const arma::vec& y,
                      int n) {
  double alpha1 = c0 + 1.5 * n;
  arma::vec resid = y - X*beta.t() - theta*v;
  double sumwv   = accu(w % v);
  double quad    = as_scalar((w % resid / v).t() * resid);
  double beta1   = C0 + sumwv + quad/(2.0*tau2);
  double g       = R::rgamma(alpha1, 1.0/beta1);
  return 1.0/g;
}

// Full conditional de v
arma::vec atualizarV(const arma::vec& y,
                     const arma::mat& X,
                     const arma::vec& w,
                     const arma::rowvec& beta,
                     double delta2,
                     double theta,
                     double sigma) {
  int N = y.n_elem;
  arma::vec v(N);
  double lambda = 0.5;
  for(int i=0;i<N;++i){
    double r   = y[i] - dot(X.row(i), beta);
    double chi = w[i]*r*r/(delta2*sigma);
    double psi = w[i]*(2.0/sigma + theta*theta/(delta2*sigma));
    NumericVector tmp = call_rgig(lambda, chi, psi);
    v[i] = tmp[0];
  }
  return v;
}

// bayesQR_weighted (método auxiliar)
// [[Rcpp::export]]
List bayesQR_weighted(const arma::vec& y,
                      const arma::mat& X,
                      const arma::vec& w,
                      double tau,
                      int n_mcmc,
                      int burnin,
                      int thin) {
  int n = y.n_elem, p = X.n_cols;
  arma::mat beta_chain(n_mcmc, p);
  arma::vec sigma_chain(n_mcmc);
  beta_chain.row(0) = solve(X,y).t();
  sigma_chain[0]    = 1.0;
  arma::vec v       = randg<arma::vec>(n, distr_param(2.0,1.0));
  double delta2 = 2.0/(tau*(1.0-tau));
  double theta  = (1.0-2.0*tau)/(tau*(1.0-tau));
  double c0=0.001, C0=0.001, tau2=delta2;
  for(int k=1;k<n_mcmc;++k){
    beta_chain.row(k) = atualizarBETA(
      arma::rowvec(p,fill::zeros),
      diagmat(vec(p).fill(1000.0)),
      X,w,
      sigma_chain[k-1],
      delta2,theta,
      v,y
    );
    v = atualizarV(y,X,w,beta_chain.row(k),delta2,theta,sigma_chain[k-1]);
    sigma_chain[k] = atualizarSIGMA(c0,C0,X,w,beta_chain.row(k),tau2,theta,v,y,n);
  }
  // Burnin + thinning
  std::vector<int> idx;
  for(int i=burnin;i<n_mcmc;i+=thin) idx.push_back(i);
  int M = idx.size();
  arma::mat bout(M,p);
  arma::vec sout(M);
  for(int i=0;i<M;++i){
    bout.row(i) = beta_chain.row(idx[i]);
    sout[i]     = sigma_chain[idx[i]];
  }
  return List::create(_["beta"]=bout,_["sigma"]=sout);
}

// bayesQR_EM completo
// [[Rcpp::export]]
List bayesQR_EM(const arma::vec& y,
                const arma::mat& X,
                const arma::vec& w,
                const arma::vec& tau_vec,
                const arma::vec& mu0,
                const arma::mat& sigma0,
                double a0,
                double b0) {
  int p = X.n_cols, q = tau_vec.n_elem, n = y.n_elem;
  arma::vec delta2 = 2.0/(tau_vec % (1.0-tau_vec));
  arma::vec theta  = (1.0-2.0*tau_vec)/(tau_vec % (1.0-tau_vec));

  // 1) inicial con quantreg::rq
  Function rq("rq", Environment::namespace_env("quantreg"));
  List reg = rq(_["formula"]=R_NilValue,
                _["x"]=X, _["y"]=y, _["tau"]=tau_vec);
  arma::mat coef_mat    = as<arma::mat>(reg["coefficients"]); // p×q
  arma::rowvec beta0_all = vectorise(coef_mat).t();           // 1×(p*q)

  // 2) historiales
  arma::mat beta_hist(1, p*q);
  beta_hist.row(0) = beta0_all;
  arma::mat sigma_hist(1, q, arma::fill::ones);

  arma::vec crit_vec(q, arma::fill::ones);
  double tol = 1e-4;
  int iter = 1;

  // 3) construir X2
  arma::mat m_aux(q,q,arma::fill::eye);
  for(int i=1;i<q;++i) for(int j=0;j<i;++j) m_aux(i,j)=1;
  arma::mat X2 = kron(m_aux, join_horiz(X, -X));

  // 4) obtener beta inicial del MCMC auxiliar
  List init = bayesQR_weighted(y,X,w,tau_vec[0],15000,5000,5);
  arma::rowvec beta_init = as<arma::mat>(init["beta"]).row(0); // ← AJUSTE

  // EM loop
  while(accu(crit_vec > tol) > 0) {
    arma::vec cov_inv; cov_inv.reset();
    arma::vec y_star;  y_star.reset();
    arma::rowvec sigma_new(q);

    // E‑step + escala
    for(int k=0;k<q;++k){
      int start = k*p, end = k*p + p - 1;
      arma::rowvec b_prev = beta_hist.row(iter-1).cols(start,end);
      arma::vec b_p = b_prev.t();

      arma::vec resid2 = pow(y - X*b_p,2);
      arma::vec a_st   = (w % resid2)/(delta2[k]*sigma_hist(iter-1,k));
      arma::vec b_st   = w * (2.0/sigma_hist(iter-1,k)
                         + theta[k]*theta[k]/(delta2[k]*sigma_hist(iter-1,k)));
      a_st.replace(0,1e-10);
      arma::vec e_inv = sqrt(b_st)/sqrt(a_st);
      arma::vec auxo  = sqrt(a_st % b_st);
      arma::vec e_nu(n);
      for(int j=0;j<n;++j)
        e_nu[j] = 1.0/e_inv[j] * (R::bessel_k(auxo[j],1.5,false)
                                 / R::bessel_k(auxo[j],0.5,false));

      cov_inv = join_vert(cov_inv,
        (1.0/(delta2[k]*sigma_hist(iter-1,k))) * (w % e_inv)
      );
      arma::vec y_aux = y - theta[k]/e_inv;
      y_star = join_vert(y_star,y_aux);

      double num = accu((w % e_inv) % pow(y_aux - X*b_p,2)) /
                     (2.0*delta2[k])
                 + accu(w % (theta[k]*theta[k] % (e_nu - 1.0/e_inv))) /
                     (2.0*delta2[k])
                 + accu(w % e_nu)
                 + b0;
      double den = (3.0*n + a0 + 1.0)/2.0;
      sigma_new[k] = num/den;
    }
    sigma_hist.insert_rows(iter, sigma_new);

    // M‑step coeficientes
    arma::mat S0inv = inv_sympd(sigma0);
    arma::mat D = X2.t()*diagmat(cov_inv)*X2
                  + kron(arma::eye<arma::mat>(2*q,2*q), S0inv);
    D = (D + D.t())/2.0;
    double sc = norm(D,"fro");
    D /= sc;
    if(!D.is_sympd()) D += arma::eye<arma::mat>(D.n_rows,D.n_rows)*1e-8;
    arma::mat Dchol     = chol(D);
    arma::mat Dchol_inv = solve(Dchol, arma::eye<arma::mat>(D.n_rows,D.n_rows));

    arma::vec d = X2.t()*diagmat(cov_inv)*y_star
                + kron(arma::eye<arma::mat>(2*q,2*q),S0inv) * repmat(mu0,2*q,1);
    d /= sc;

    // construir A y bvec
    int m = 2*p*q + (q-1);
    arma::mat A(2*p*q, m, arma::fill::zeros);
    A.submat(0,0,2*p*q-1,2*p*q-1) = arma::eye<arma::mat>(2*p*q,2*p*q);
    arma::vec vblock(2*p, arma::fill::zeros);
    vblock[0]=1; for(int ii=p;ii<2*p;++ii) vblock[ii]=-1;
    arma::mat block = kron(arma::eye<arma::mat>(q-1,q-1), vblock);
    A.submat(2*p,2*p*q,2*p*q-1,m-1)=block;

    arma::vec bvec(m, arma::fill::zeros);
    bvec.subvec(0,p-1) = beta_init.t();  // ← AJUSTE: inicial con length = p

    Function solveQP("solve.QP", Environment::namespace_env("quadprog"));
    List sol = solveQP(
      _["Dmat"]=wrap(Dchol_inv),
      _["dvec"]=wrap(d),
      _["Amat"]=wrap(A),
      _["bvec"]=wrap(bvec),
      _["meq"]=2*p,
      _["factorized"]=true
    );
    arma::vec gamma = as<arma::vec>(sol["solution"]);

    // reconstruir beta
    arma::mat m2(q,q,arma::fill::eye);
    for(int i=0;i<q;++i) for(int j=i+1;j<q;++j) m2(i,j)=1;
    arma::mat AA = kron(m2, join_vert(arma::eye<arma::mat>(p,p),
                                       -arma::eye<arma::mat>(p,p)));
    arma::rowvec beta_new = gamma.t() * AA;
    beta_hist.insert_rows(iter, beta_new);

    crit_vec = join_vert(pow(beta_new - beta_hist.row(iter-1),2),
                         pow(sigma_new - sigma_hist.row(iter-1).t(),2));
    iter++;
  }

  return List::create(
    _["beta"]  = beta_hist.row(iter-1),
    _["sigma"] = sigma_hist.row(iter-1)
  );
}