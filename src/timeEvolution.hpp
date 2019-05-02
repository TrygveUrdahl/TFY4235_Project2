#pragma once


arma::cx_mat getSystemStateEvolution(const arma::mat &eigvec, const arma::vec &eigenEnergy,
                            const arma::vec &initialState, const arma::vec &alphas, double dx, double t,
                            int tSteps);

arma::cx_mat evolveSystemForwardEuler(const arma::mat &eigvec,
              const arma::vec &eigenEnergy, const arma::cx_vec &initialState,
              const arma::vec &xaxis, double (*potential)(double, double),
              double v0, double CFL, int tSteps);

arma::cx_mat evolveSystemCrankNicolson(const arma::cx_vec &initialState,
              const arma::vec &xaxis, double (*potential)(double, double),
              double v0, double t);
