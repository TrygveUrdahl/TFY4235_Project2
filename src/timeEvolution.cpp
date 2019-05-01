#include <iostream>
#include <armadillo>
#include <cmath>
#include <omp.h>

#include "utils.hpp"

arma::cx_mat getSystemStateEvolution(const arma::mat &eigvec, const arma::vec &eigenEnergy,
              const arma::vec &initialState, const arma::vec &alphas, double t, int tSteps) {
  arma::cx_mat states(eigvec.col(0).n_elem, tSteps);
  states.fill(0);
  const arma::cx_double im(0, 1);
  const double dx = 1.0/(eigvec.col(0).n_elem);
  double dt = t/static_cast<double>(tSteps);
  for (int i = 0; i < tSteps; i++) { // Time steps
    double time = i * dt;
    for (int n = 0; n < alphas.n_elem; n++) { // Energies
      states.col(i) += alphas(n) * std::exp(-im * eigenEnergy(n) * time) * eigvec(n);
    }
  }
  return states;
}


arma::cx_vec advanceSystemForwardEuler(const arma::cx_vec &initialState,
              const arma::vec &xaxis, double (*potential)(double, double),
              double v0, double dt) {

  const int N = initialState.n_elem;
  arma::cx_vec finalState(N);
  arma::cx_mat A(N, N);
  const arma::cx_double im(0, 1);
  const double dx = xaxis(1) - xaxis(0); // Uniform x-axis
  const double dx2 = dx * dx;
  // Fill A
  for (int i = 0; i < N; i++) {
    double v = potential(xaxis(i), v0);
    A(i, i) = 1.0 - (im * dt * v) - (2.0 * im * dt/dx2);

    if (i > 0) {
      A(i, i - 1) = im * dt / dx2;
    }
    if (i < N - 1) {
      A(i, i + 1) = im * dt / dx2;
    }
  }
  finalState = A * initialState;
  return finalState;
}


arma::cx_mat evolveSystemForwardEuler(const arma::mat &eigvec,
              const arma::vec &eigenEnergy, const arma::cx_vec &initialState,
              const arma::vec &xaxis, double (*potential)(double, double),
              double v0, double CFL, int tSteps) {

  arma::cx_mat states(initialState.n_elem, tSteps);
  states.col(0) = initialState;
  const double dx = xaxis(1) - xaxis(0);
  const double dx2 = dx * dx;
  double dt = dx2 * CFL;

  for (int t = 1; t < tSteps; t++) {
    states.col(t) = advanceSystemForwardEuler(states.col(t - 1), xaxis, potential, v0, dt);
  }

  return states;
}


arma::cx_vec advanceSystemCrankNicolson(const arma::cx_vec &initialState,
              const arma::vec &xaxis, double (*potential)(double, double),
              double v0, double dt) {

  const int N = initialState.n_elem;
  arma::cx_vec finalState(N);
  arma::cx_mat A(N, N);
  arma::cx_mat b(N, N);
  const arma::cx_double im(0, 1);
  const double dx = xaxis(1) - xaxis(0); // Uniform x-axis
  const double dx2 = dx * dx;
  // Fill A and b
  for (int i = 0; i < N; i++) {
    double v = potential(xaxis(i), v0);
    A(i, i) = 1.0 + ((im/2.0) * dt * (v + 2.0/dx2));
    b(i, i) = 1.0 - ((im/2.0) * dt * (v + 2.0/dx2));
    if( i > 0) {
      A(i, i - 1) = -(im / 2.0) * dt / dx2;
      b(i, i - 1) = (im / 2.0) * dt / dx2;
    }
    if (i < N - 1) {
      A(i, i + 1) = -(im / 2.0) * dt / dx2;
      b(i, i + 1) = (im / 2.0) * dt / dx2;
    }
  }
  arma::cx_vec rhs = b * initialState;
  return arma::solve(A, rhs);
}


arma::cx_mat evolveSystemCrankNicolson(const arma::mat &eigvec,
              const arma::vec &eigenEnergy, const arma::cx_vec &initialState,
              const arma::vec &xaxis, double (*potential)(double, double),
              double v0, double t, int tSteps) {

  arma::cx_mat states(initialState.n_elem, tSteps);
  states.col(0) = initialState;
  const double dt = t/tSteps;
  for (int t = 1; t < tSteps; t++) {
    states.col(t) = advanceSystemCrankNicolson(states.col(t - 1), xaxis, potential, v0, dt);
  }
  return states;
}
