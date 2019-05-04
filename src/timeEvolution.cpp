#include <iostream>
#include <armadillo>
#include <cmath>
#include <omp.h>

#include "utils.hpp"

// Evolve system by expansion in alphas
arma::cx_mat getSystemStateEvolution(const arma::mat &eigvec, const arma::vec &eigenEnergy,
              const arma::vec &initialState, const arma::vec &alphas, double dx, double t, int tSteps) {
  arma::cx_mat states(eigvec.col(0).n_elem, tSteps, arma::fill::zeros);
  const arma::cx_double im(0.0, 1.0);
  double dt = t/static_cast<double>(tSteps);
  #pragma omp parallel for schedule(static)
  for (int it = 0; it < tSteps; it++) { // Time steps
    double time = it * dt;
    for (int en = 0; en < alphas.n_elem; en++) { // Energies
      states.col(it) += alphas(en) * std::exp(im * eigenEnergy(en) * time) * eigvec.col(en);
    }
  }
  return states;
}

// One time step of Forward Euler
arma::cx_vec advanceSystemForwardEuler(const arma::cx_vec &initialState,
              const arma::vec &xaxis, double (*potential)(double, double),
              double v0, double dt) {

  const int N = initialState.n_elem;
  arma::cx_vec finalState(N);
  arma::cx_mat A(N, N, arma::fill::zeros);
  const arma::cx_double im(0, 1);
  const double dx = xaxis(1) - xaxis(0); // Uniform x-axis
  const double dx2 = dx * dx;
  // Fill A
  #pragma omp parallel for schedule(static)
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

// Evolve system by Forward Euler
arma::cx_mat evolveSystemForwardEuler(const arma::cx_vec &initialState,
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

// One time step of Crank-Nicolson
arma::cx_vec advanceSystemCrankNicolson(const arma::cx_vec &initialState,
              const arma::vec &xaxis, double (*potential)(double, double),
              double v0, double dt) {
  const int N = initialState.n_elem - 2;
  arma::cx_mat A(N, N, arma::fill::zeros);
  arma::cx_mat b(N, N, arma::fill::zeros);
  const arma::cx_double im(0, 1);
  const double dx = xaxis(1) - xaxis(0); // Uniform x-axis
  const double dx2 = dx * dx;
  // Fill A and b
  for (int i = 0; i < N; i++) {
    double v = potential(xaxis(i + 1), v0);
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
  // Ensure boundary conditions
  arma::cx_vec rhs = b * initialState.subvec(1, N);
  arma::cx_vec solution = arma::solve(A, rhs);
  arma::cx_vec final(initialState.n_elem, arma::fill::zeros);
  final.subvec(1, N) = solution;
  return final;
}

// Evolve system by Crank-Nicolson
arma::cx_mat evolveSystemCrankNicolson(const arma::cx_vec &initialState,
              const arma::vec &xaxis, double (*potential)(double, double),
              double v0, double t) {

  const double dx = xaxis(1) - xaxis(0);
  const double dt = dx * dx;
  const int tSteps = t/dt;
  std::cout << "tSteps: " << tSteps << std::endl;
  arma::cx_mat states(initialState.n_elem, tSteps, arma::fill::zeros);
  states.col(0) = initialState;
  for (int t = 1; t < tSteps; t++) {
    states.col(t) = advanceSystemCrankNicolson(states.col(t - 1), xaxis, potential, v0, dt);
    if (t % 1000 == 0) std::cout << "t: " << t << std::endl;
  }
  return states;
}
