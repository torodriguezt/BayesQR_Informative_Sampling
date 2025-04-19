// em_bwqr_al_mo.cpp

#include <vector>
#include <stdexcept>
#include <Eigen/Dense>
#include <boost/math/special_functions/bessel.hpp>

std::vector<Eigen::VectorXd> EM_BWQR_AL_MO(
    const Eigen::MatrixXd& y,
    const Eigen::MatrixXd& x,
    const Eigen::VectorXd& w,
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& gamma_u,
    double tau,
    const Eigen::VectorXd& mu0,
    const Eigen::MatrixXd& sigma0,
    double a0,
    double b0) {

    // Dimensiones
    const Eigen::Index n = y.rows();
    const Eigen::Index m = y.cols();
    const Eigen::Index p = x.cols();

    // Validaciones de entrada
    if (m < 2)
        throw std::invalid_argument("y debe tener al menos 2 columnas");
    if (u.size() != m || gamma_u.size() != m)
        throw std::invalid_argument("u y gamma_u deben tener tamaño igual a columnas de y");
    if (w.size() != n)
        throw std::invalid_argument("w debe tener tamaño igual a filas de y");
    if (mu0.size() != p + 1 || sigma0.rows() != p + 1 || sigma0.cols() != p + 1)
        throw std::invalid_argument("mu0 y sigma0 deben tener dimensión p+1");

    // Paso direccional
    Eigen::VectorXd y_u     = y * u;
    Eigen::VectorXd y_gamma = y * gamma_u;

    // Matriz de diseño x_star = [x | y_gamma]
    Eigen::MatrixXd x_star(n, p + 1);
    if (p > 0) x_star.leftCols(p) = x;
    x_star.col(p) = y_gamma;
    Eigen::MatrixXd xT = x_star.transpose();

    // Inicialización via SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd0(x_star, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd beta_prev = svd0.solve(y_u);
    double sigma_prev = 1.0;

    std::vector<Eigen::VectorXd> beta_hist;
    std::vector<double>          sigma_hist;
    beta_hist.reserve(100);
    sigma_hist.reserve(100);
    beta_hist.push_back(beta_prev);
    sigma_hist.push_back(sigma_prev);

    // Constantes
    const double delta2 = 2.0 / (tau * (1.0 - tau));
    const double theta  = (1.0 - 2 * tau) / (tau * (1.0 - tau));
    const double tol    = 1e-6;
    const int    max_it = 100;

    for (int iter = 0; iter < max_it; ++iter) {
        // E-step
        Eigen::VectorXd resid   = y_u - x_star * beta_prev;
        Eigen::VectorXd a_star  = (w.array() * resid.array().square()) / (delta2 * sigma_prev);
        Eigen::VectorXd b_star  = w.array() * (2.0 / sigma_prev + theta * theta / (delta2 * sigma_prev));
        a_star = a_star.array().max(1e-10);

        Eigen::VectorXd inv_sqrt_a = a_star.array().rsqrt();
        Eigen::VectorXd sqrt_b     = b_star.array().sqrt();
        Eigen::VectorXd e_nu_inv   = sqrt_b.array() * inv_sqrt_a.array();

        Eigen::VectorXd aux = (a_star.array() * b_star.array()).sqrt();
        Eigen::VectorXd e_nu = Eigen::VectorXd::Ones(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            double k15 = boost::math::cyl_bessel_k(1.5, aux(i));
            double k05 = boost::math::cyl_bessel_k(0.5, aux(i));
            e_nu(i)    = (k15 / std::max(k05, 1e-10)) / e_nu_inv(i);
        }

        // M-step
        Eigen::MatrixXd Winv = Eigen::MatrixXd::Zero(n, n);
        for (Eigen::Index i = 0; i < n; ++i)
            Winv(i, i) = w(i) * e_nu_inv(i) / (delta2 * sigma_prev);

        Eigen::VectorXd y_tilde = y_u - theta * e_nu_inv.array().inverse().matrix();

        Eigen::MatrixXd sigma0_inv = sigma0.ldlt().solve(Eigen::MatrixXd::Identity(p + 1, p + 1));
        Eigen::MatrixXd A         = xT * Winv * x_star + sigma0_inv;
        Eigen::VectorXd b         = xT * Winv * y_tilde + sigma0_inv * mu0;
        Eigen::JacobiSVD<Eigen::MatrixXd> svdA(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd beta_next = svdA.solve(b);

        // Actualización de sigma
        double term1      = (w.array() * e_nu_inv.array() * (y_tilde - x_star * beta_next).array().square()).sum() / (2.0 * delta2);
        double term2      = (w.array() * theta * theta * (e_nu.array() - e_nu_inv.array().inverse())).sum() / (2.0 * delta2);
        double term3      = (w.array() * e_nu.array()).sum();
        double numerator  = term1 + term2 + term3 + b0;
        double denominator= (3.0 * n + a0 + 1.0) / 2.0;
        double sigma_next = numerator / denominator;
        sigma_next = std::max(sigma_next, 1e-10);

        // Convergencia
        if ((beta_next - beta_prev).norm() + std::abs(sigma_next - sigma_prev) < tol) {
            beta_hist.push_back(beta_next);
            sigma_hist.push_back(sigma_next);
            break;
        }
        beta_hist.push_back(beta_next);
        sigma_hist.push_back(sigma_next);
        beta_prev  = beta_next;
        sigma_prev = sigma_next;
    }

    // Resultado final
    return { beta_hist.back(), Eigen::VectorXd::Constant(1, sigma_hist.back()) };
}
