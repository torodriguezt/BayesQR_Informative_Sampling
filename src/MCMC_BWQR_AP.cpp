#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include <iomanip>

constexpr double PI = 3.14159265358979323846;

static std::mt19937& global_gen() {
    static std::mt19937 gen(42);
    return gen;
}

double log_det_ldlt(const Eigen::MatrixXd& M) {
    Eigen::LDLT<Eigen::MatrixXd> ldlt(M);
    return ldlt.vectorD().array().log().sum();
}

Eigen::VectorXd rmvnorm(const Eigen::VectorXd& mean, const Eigen::MatrixXd& sigma) {
    std::normal_distribution<double> dist(0.0, 1.0);
    int dim = mean.size();
    Eigen::VectorXd z(dim);
    for (int i = 0; i < dim; ++i) z(i) = dist(global_gen());
    Eigen::LLT<Eigen::MatrixXd> llt(sigma);
    Eigen::MatrixXd L = llt.matrixL();
    return mean + L * z;
}

struct MHResult {
    Eigen::VectorXd beta;
    bool accepted;
    double new_ct;
};

double condicionalBETA_MH_bio(
    const Eigen::VectorXd& beta,
    const Eigen::VectorXd& b,
    const Eigen::MatrixXd& B,
    const Eigen::VectorXd& dados,
    const Eigen::MatrixXd& x,
    const Eigen::VectorXd& w_vec,
    double w_scalar,
    double tau
) {
    int n = dados.size();
    Eigen::VectorXd diff = beta - b;
    double priori = -0.5 * diff.dot(B.ldlt().solve(diff));
    double w_un = w_vec.sum() / n * w_scalar;
    Eigen::VectorXd res = dados - x * beta;
    Eigen::ArrayXd tau_minus_one = Eigen::ArrayXd::Constant(n, tau - 1.0);
    Eigen::ArrayXd tau_arr = Eigen::ArrayXd::Constant(n, tau);
    Eigen::Array<bool, Eigen::Dynamic, 1> mask = (res.array() < 0);
    Eigen::VectorXd indicator = mask.select(tau_minus_one, tau_arr).matrix();
    Eigen::VectorXd s_tau = w_un * x.transpose() * indicator;
    int p = x.cols();
    Eigen::MatrixXd S(n, p);
    for (int i = 0; i < n; ++i) S.row(i) = x.row(i) * indicator(i);
    double factor = (1.0 - 1.0 / w_un) / std::pow(1.0 / w_un, 2);
    Eigen::MatrixXd w_cov_aux = factor * (S.transpose() * S);
    Eigen::MatrixXd w_cov = w_cov_aux.inverse();
    double quad = s_tau.dot(w_cov * s_tau);
    double logdet = p * std::log(2.0 * PI) + log_det_ldlt(w_cov_aux);
    double verossi = -0.5 * quad - 0.5 * logdet;
    return priori + verossi;
}

MHResult atualizarBETA_MH_bio(
    const Eigen::VectorXd& b,
    const Eigen::MatrixXd& B,
    const Eigen::VectorXd& dados,
    const Eigen::MatrixXd& x,
    const Eigen::VectorXd& beta,
    const Eigen::VectorXd& w_vec,
    double w_scalar,
    double tau,
    double ct,
    int k,
    double t_param,
    double c0,
    double c1
) {
    int n = dados.size();
    Eigen::MatrixXd W2X = x;
    for (int i = 0; i < n; ++i) W2X.row(i) *= std::pow(w_vec(i), 2);
    Eigen::MatrixXd cov_base = (1.0 / n) * (W2X.transpose() * x);
    Eigen::MatrixXd sigma_MH = (tau * (1.0 - tau)) * cov_base.inverse();
    Eigen::VectorXd proposed = rmvnorm(beta, ct * sigma_MH);
    double log_num = condicionalBETA_MH_bio(proposed, b, B, dados, x, w_vec, w_scalar, tau);
    double log_den = condicionalBETA_MH_bio(beta, b, B, dados, x, w_vec, w_scalar, tau);
    double alpha = std::exp(log_num - log_den);
    double u = std::uniform_real_distribution<double>(0.0, 1.0)(global_gen());
    MHResult res;
    res.accepted = (u < std::min(1.0, alpha));
    res.beta = res.accepted ? proposed : beta;
    double log_ct = std::log(ct) + c0 * (1.0 / std::pow(k, c1)) * (std::min(1.0, alpha) - 0.234);
    res.new_ct = std::exp(log_ct);
    return res;
}

Eigen::MatrixXd bayesQRSL_weighted_bio(
    const Eigen::VectorXd& y,
    const Eigen::MatrixXd& x,
    const Eigen::VectorXd& w_vec,
    double w_scalar,
    double tau,
    int n_mcmc,
    int burnin,
    int thin,
    double cte,
    double t_param,
    double c0,
    double c1,
    const Eigen::VectorXd& a1,
    const Eigen::MatrixXd& B1
) {
    int n = y.size(), p = x.cols();
    Eigen::MatrixXd beta_chain(n_mcmc, p);
    std::vector<double> ct_vals(n_mcmc);
    beta_chain.row(0) = (x.transpose() * x).inverse() * x.transpose() * y;
    ct_vals[0] = cte;
    for (int k = 1; k < n_mcmc; ++k) {
        MHResult mh = atualizarBETA_MH_bio(
            a1, B1, y, x, beta_chain.row(k - 1), w_vec, w_scalar,
            tau, ct_vals[k - 1], k, t_param, c0, c1
        );
        beta_chain.row(k) = mh.beta;
        ct_vals[k] = mh.new_ct;
    }
    int samples = (n_mcmc - burnin) / thin;
    Eigen::MatrixXd result(samples, p);
    int idx = 0;
    for (int i = burnin; i < n_mcmc; i += thin) result.row(idx++) = beta_chain.row(i);
    return result;
}

int main() {
    std::cout << "Bayesian Quantile Regression implementation in C++ (final)\n";
    std::cout << std::string(60, '-') << "\n";

    int n = 200, p = 3;
    Eigen::VectorXd true_coefs(p);
    true_coefs << 1.0, 2.5, -1.5;

    Eigen::VectorXd y(n);
    Eigen::MatrixXd X(n, p);
    auto& gen = global_gen();
    std::uniform_real_distribution<double> unif(-5, 5);
    std::normal_distribution<double> noise(0, 1);

    for (int i = 0; i < n; ++i) {
        X(i, 0) = 1;
        for (int j = 1; j < p; ++j) {
            X(i, j) = unif(gen);
        }
        y(i) = X.row(i).dot(true_coefs) + noise(gen);
    }

    Eigen::VectorXd w_vec = Eigen::VectorXd::Ones(n);
    double w_scalar = 2.0;
    int n_mcmc = 5000, burnin = 1000, thin = 10;
    double tau = 0.5, cte = 2.38, t_param = 1.0, c0 = 1.0, c1 = 0.8;

    Eigen::VectorXd a1 = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd B1 = 100.0 * Eigen::MatrixXd::Identity(p, p);

    auto samples = bayesQRSL_weighted_bio(
        y, X, w_vec, w_scalar, tau,
        n_mcmc, burnin, thin,
        cte, t_param, c0, c1, a1, B1
    );

    int S = samples.rows();
    Eigen::VectorXd means = samples.colwise().mean();
    Eigen::VectorXd lower(p), upper(p);

    for (int j = 0; j < p; ++j) {
        std::vector<double> vals(samples.col(j).data(), samples.col(j).data() + S);
        std::sort(vals.begin(), vals.end());
        lower(j) = vals[int(0.025 * S)];
        upper(j) = vals[int(0.975 * S)];
    }

    std::cout << std::setw(10) << "Param"
        << std::setw(15) << "True"
        << std::setw(15) << "Mean"
        << std::setw(15) << "2.5%"
        << std::setw(15) << "97.5%" << "\n";

    for (int j = 0; j < p; ++j) {
        std::cout << std::setw(10) << j
            << std::setw(15) << true_coefs(j)
            << std::setw(15) << means(j)
            << std::setw(15) << lower(j)
            << std::setw(15) << upper(j) << "\n";
    }

    return 0;
}
