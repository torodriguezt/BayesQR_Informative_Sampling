// =============================================
// bayesQRSL_weighted_corregido.cpp
// =============================================
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <tuple>
#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <cmath>

using namespace Eigen;
using std::vector;
using std::tuple;

// Funcion de log-posterior (igual que en R)
double condicionalBETA_MH(
    const VectorXd& beta,
    const VectorXd& b,
    const MatrixXd& B,
    const VectorXd& dados,
    const MatrixXd& x,
    const VectorXd& w,
    double tau
) {
    int n = dados.size();
    MatrixXd B_inv = B.llt().solve(MatrixXd::Identity(B.rows(), B.cols()));
    double prior = -0.5 * (beta - b).transpose() * B_inv * (beta - b);

    VectorXd resid = dados - x * beta;
    VectorXd I = (resid.array() < 0.0).cast<double>();
    VectorXd stau = x.transpose() * (w.array() * (tau - I.array())).matrix();

    VectorXd w2 = w.array().square();
    MatrixXd XtW2X = x.transpose() * w2.asDiagonal() * x;
    MatrixXd w_cov = (n / (tau * (1.0 - tau))) *
        XtW2X.llt().solve(MatrixXd::Identity(XtW2X.rows(), XtW2X.cols()));

    double verossi = -(1.0 / (2.0 * n)) * stau.transpose() * w_cov * stau;
    return prior + verossi;
}

// Un paso de Metropolis–Hastings
tuple<VectorXd, int, double> atualizarBETA_MH(
    const VectorXd& b,
    const MatrixXd& B,
    const VectorXd& dados,
    const MatrixXd& x,
    const VectorXd& beta,
    const VectorXd& w,
    double tau,
    double ct,
    int k,
    int t,
    double c0,
    double c1,
    std::mt19937& rng
) {
    int p = beta.size(), n = dados.size();
    VectorXd valor_atual = beta;

    VectorXd w2 = w.array().square();
    MatrixXd XtW2X = x.transpose() * w2.asDiagonal() * x;
    MatrixXd Sigma_MH = (tau * (1.0 - tau)) *
        ((1.0 / n) * XtW2X).llt().solve(MatrixXd::Identity(p, p));

    // Generar propuesta
    MatrixXd L(p, p);
    L = (ct * Sigma_MH).llt().matrixL();
    std::normal_distribution<double> N(0.0, 1.0);
    VectorXd z(p);
    for (int i = 0; i < p; ++i) z(i) = N(rng);
    VectorXd valor_proposto = valor_atual + L * z;

    // Ratio de aceptacion
    double log_num = condicionalBETA_MH(valor_proposto, b, B, dados, x, w, tau);
    double log_den = condicionalBETA_MH(valor_atual, b, B, dados, x, w, tau);
    double alpha = std::min(1.0, std::exp(log_num - log_den));

    std::uniform_real_distribution<double> U(0.0, 1.0);
    int aceita = (U(rng) < alpha) ? 1 : 0;
    VectorXd beta_new = aceita ? valor_proposto : valor_atual;

    // Actualizar ct
    double log_ct_new = std::log(ct) + c0 * (1.0 / std::pow(k, c1)) * (alpha - 0.234);
    double ct_new = std::exp(log_ct_new);

    return { beta_new, aceita, ct_new };
}

// Algoritmo principal (igual que en R, pero sin simular datos)
vector<MatrixXd> bayesQRSL_weighted(
    const VectorXd& y,
    const MatrixXd& x,
    const VectorXd& w,
    double tau,
    int n_mcmc,
    int burnin_mcmc,
    int thin_mcmc,
    double cte,
    int t,
    double c0,
    double c1,
    unsigned int seed
) {
    int n = y.size(), p = x.cols();
    MatrixXd beta_samples(n_mcmc, p);
    // Inicio en OLS (como lm())
    VectorXd beta0 = x.colPivHouseholderQr().solve(y);
    beta_samples.row(0) = beta0.transpose();

    vector<int> contador(n_mcmc, 0);
    vector<double> ct(n_mcmc, cte);
    std::mt19937 rng(seed);

    MatrixXd B = 1000.0 * MatrixXd::Identity(p, p);
    VectorXd b = VectorXd::Zero(p);

    for (int k = 1; k < n_mcmc; ++k) {
        auto [bk, ace, new_ct] = atualizarBETA_MH(
            b, B, y, x, beta_samples.row(k - 1).transpose(),
            w, tau, ct[k - 1], k, t, c0, c1, rng
        );
        beta_samples.row(k) = bk.transpose();
        contador[k] = ace;
        ct[k] = new_ct;
    }

    // Extraer posterior tras burn-in y thinning
    int out_iters = 0;
    for (int i = burnin_mcmc; i < n_mcmc; i += thin_mcmc) ++out_iters;
    MatrixXd post(out_iters, p);
    for (int i = burnin_mcmc, j = 0; i < n_mcmc; i += thin_mcmc, ++j) {
        post.row(j) = beta_samples.row(i);
    }

    // Informar resultado
    int accepts = std::accumulate(contador.begin(), contador.end(), 0);
    std::cout << "Tasa de aceptacion: " << (double)accepts / n_mcmc << "\n";
    VectorXd mean = post.colwise().mean();
    VectorXd var = (post.rowwise() - mean.transpose()).array().square().colwise().mean();
    VectorXd sd = var.array().sqrt();
    std::cout << "Beta 0: Media = " << mean(0) << " SD = " << sd(0) << "\n";
    std::cout << "Beta 1: Media = " << mean(1) << " SD = " << sd(1) << "\n";

    return { post };
}

int main() {
    // 1) Leer el CSV generado por R
    std::ifstream infile("datos_para_cpp.csv");
    if (!infile) {
        std::cerr << "Error al abrir datos_para_cpp.csv\n";
        return 1;
    }
    std::string header;
    std::getline(infile, header);

    vector<vector<double>> data;
    std::string line;
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        double yi, x0i, x1i, wi;
        char comma;
        ss >> yi >> comma >> x0i >> comma >> x1i >> comma >> wi;
        data.push_back({ yi, x0i, x1i, wi });
    }

    int n = data.size();
    MatrixXd x(n, 2);
    VectorXd y(n), w(n);
    for (int i = 0; i < n; ++i) {
        y(i) = data[i][0];
        x(i, 0) = data[i][1];
        x(i, 1) = data[i][2];
        w(i) = data[i][3];
    }

    // 2) Parametros
    double tau = 0.5;
    int n_mcmc = 10000, burnin = 1000, thin = 10;
    double cte = 1.0, c0 = 0.01, c1 = 0.5;
    int t = 50;
    unsigned int seed = 123;

    // 3) Ejecutar
    auto res = bayesQRSL_weighted(y, x, w, tau,
        n_mcmc, burnin, thin,
        cte, t, c0, c1, seed);

    // 4) Guardar los primeros 5 draws para comparar
    std::ofstream out5("primeros_5_cpp.csv");
    out5 << "Beta0,Beta1\n";
    for (int i = 0; i < 5; ++i) {
        out5 << res[0](i, 0) << "," << res[0](i, 1) << "\n";
    }

    return 0;
}
