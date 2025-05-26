// em_bwqr_al_mo.h
#ifndef EM_BWQR_AL_MO_H
#define EM_BWQR_AL_MO_H

#include <vector>
#include <Eigen/Dense>

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
    double b0
);

#endif // EM_BWQR_AL_MO_H