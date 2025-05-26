#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>
#include <iomanip>

using namespace Eigen;

// em_bwqr_al_mo.cpp
// Estructura para devolver β y σ
struct BayesQRResult {
    VectorXd beta;
    double sigma;
};

// Lee un CSV sin encabezados en una matriz Eigen (filas × columnas)
void load_csv(const char* filename, MatrixXd& M) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Error al abrir '" << filename << "'\n";
        std::exit(1);
    }
    std::string line;
    int row = 0;
    while (std::getline(in, line) && row < M.rows()) {
        std::stringstream ss(line);
        for (int col = 0; col < M.cols(); ++col) {
            double v;
            ss >> v;
            if (col < M.cols() - 1) ss.ignore(1, ',');
            M(row, col) = v;
        }
        ++row;
    }
}

BayesQRResult bayesQR_weighted_EM(
    const MatrixXd& y,
    const MatrixXd& x,
    const VectorXd& w,
    const VectorXd& u,
    const VectorXd& gamma_u,
    double tau,
    const VectorXd& mu0,
    const MatrixXd& sigma0,
    double a0,
    double b0
) {
    int n = y.rows();
    int p = x.cols();
    int m = p + 1;

    // Construye x_star = [x | y*gamma_u]
    VectorXd y_u     = y * u;
    VectorXd y_gamma = y * gamma_u;
    MatrixXd x_star(n, m);
    x_star.leftCols(p) = x;
    x_star.col(p)      = y_gamma;

    double delta2 = 2.0 / (tau * (1.0 - tau));
    double theta  = (1.0 - 2.0 * tau) / (tau * (1.0 - tau));
    const double tol = 1e-5;

    // Pre-invierte sigma0
    MatrixXd sigma0_inv = sigma0.ldlt().solve(MatrixXd::Identity(m, m));

    // Inicialización OLS
    VectorXd beta_prev = (x_star.transpose() * x_star).ldlt()
                             .solve(x_star.transpose() * y_u);
    double sigma_prev = 1.0;

    // Buffers temporales
    VectorXd resid(n), a_star(n), b_star(n), e_nu_inv(n), aux_obj(n),
             e_nu(n), y_aux(n), wy(n);

    while (true) {
        double inv_ds = 1.0 / (delta2 * sigma_prev);

        // E-step
        for (int i = 0; i < n; ++i) {
            double r = y_u(i) - x_star.row(i).dot(beta_prev);
            resid(i) = r;
            double as = w(i) * r * r * inv_ds;
            a_star(i) = (as == 0.0 ? 1e-10 : as);
            double bs = w(i) * (2.0 / sigma_prev + theta * theta * inv_ds);
            b_star(i) = bs;
            double sa = std::sqrt(a_star(i));
            double sb = std::sqrt(bs);
            e_nu_inv(i) = sb / sa;
            aux_obj(i)   = sa * sb;
            // Ratio Bessel K_{3/2} / K_{1/2} = 1 + 1/z
            e_nu(i)      = (1.0 / e_nu_inv(i)) * (1.0 + 1.0 / aux_obj(i));
        }

        // y_aux = y_u - theta / e_nu_inv
        for (int i = 0; i < n; ++i)
            y_aux(i) = y_u(i) - theta / e_nu_inv(i);

        // M-step
        MatrixXd A = sigma0_inv;
        for (int i = 0; i < n; ++i) {
            double coef = w(i) * e_nu_inv(i) * inv_ds;
            for (int a = 0; a < m; ++a)
                for (int b = 0; b < m; ++b)
                    A(a, b) += coef * x_star(i, a) * x_star(i, b);
            wy(i) = w(i) * e_nu_inv(i) * y_aux(i);
        }

        VectorXd B = sigma0_inv * mu0;
        for (int a = 0; a < m; ++a) {
            double sum = 0.0;
            for (int i = 0; i < n; ++i)
                sum += x_star(i, a) * wy(i);
            B(a) += sum * inv_ds;
        }

        // Resolver β
        VectorXd beta_curr = A.ldlt().solve(B);

        // Actualizar σ
        double term1 = 0.0, term2 = 0.0, term3 = 0.0;
        for (int i = 0; i < n; ++i) {
            double r2 = y_aux(i) - x_star.row(i).dot(beta_curr);
            term1 += w(i) * e_nu_inv(i) * r2 * r2;
            term2 += w(i) * theta * theta * (e_nu(i) - 1.0 / e_nu_inv(i));
            term3 += w(i) * e_nu(i);
        }
        term1 /= (2.0 * delta2);
        term2 /= (2.0 * delta2);
        double sigma_curr = (term1 + term2 + term3 + b0) / ((3.0 * n + a0 + 1.0) / 2.0);

        // Convergencia
        double diff = (beta_curr - beta_prev).cwiseAbs().sum() + std::abs(sigma_curr - sigma_prev);
        if (diff < tol)
            return {beta_curr, sigma_curr};

        beta_prev  = beta_curr;
        sigma_prev = sigma_curr;
    }
}

int main() {
    std::cout << ">>> Cargando datos desde CSV...\n";
    const int n = 5000, p = 3, d = 2;  // tamaño pequeño para demo en R/C++

    MatrixXd y(n, d), x(n, p), u(d, 1), gamma_u(d, 1);
    VectorXd w = VectorXd::Ones(n);

    load_csv("y.csv",       y);
    load_csv("x.csv",       x);
    load_csv("u.csv",       u);
    load_csv("gamma_u.csv", gamma_u);

    VectorXd mu0    = VectorXd::Zero(p + 1);
    MatrixXd sigma0 = MatrixXd::Identity(p + 1, p + 1);
    double tau = 0.5, a0 = 2.0, b0 = 2.0;

    std::cout << ">>> Ejecutando EM...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    BayesQRResult res = bayesQR_weighted_EM(
        y, x, w, u, gamma_u,
        tau, mu0, sigma0, a0, b0
    );
    auto t1 = std::chrono::high_resolution_clock::now();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    // Mostrar resultados con precisión
    std::cout << "C++ β: ";
    for (int i = 0; i < res.beta.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << res.beta(i);
        if (i + 1 < res.beta.size()) std::cout << " ";
    }
    std::cout << "  σ = " << std::fixed << std::setprecision(6) << res.sigma;
    std::cout << "  (" << ms << " ms)\n";

    return 0;
}
