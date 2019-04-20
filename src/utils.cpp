#include <iostream>
#include <armadillo>
#include <cmath>
#include <omp.h>

// 1D Laplace operator for FDM
arma::sp_mat generateLaplaceOperator(int n, double value, bool periodic) {
  arma::sp_mat mat(n, n);
  // #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    mat(i, i) = 2 * value;
    if (i > 0) {
      mat(i, i - 1) = -1 * value;
    }
    if (i < (n - 1)) {
      mat(i, i + 1) = -1 * value;
    }
  }
  if (periodic) {
    mat(0, n - 1) = -1 * value;
    mat(n - 1, 0) = -1 * value;
  }
  return mat;
}

// Potential energy operator: v(x_i) on diagonal
arma::sp_mat generateEnergyOperator(int n, arma::vec xaxis, double (*function)(double)) {
  arma::sp_mat mat(n, n);
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    mat(i, i) = function(xaxis(i));
  }
  return mat;
}

// Full FDM matrix with both Laplace and Energy operator
arma::sp_mat generateFDMMatrix(int n, double value, arma::vec xaxis, double (*potential)(double), bool periodic) {
  arma::sp_mat mat = generateLaplaceOperator(n , value, periodic);
  mat += generateEnergyOperator(n, xaxis, potential);
  return mat;
}

// Generate initial state vector from function
arma::vec generateInitialState(double (*function)(double, int), arma::vec xaxis) {
  arma::vec initialState(xaxis.n_elem);
  for (int i = 0; i < xaxis.n_elem; i++) {
    initialState(i) = function(xaxis(i), xaxis.n_elem);
  }
  return initialState;
}

// Calculate inner product of two vectors
double innerProduct(arma::vec eigen, arma::vec initial) {
  arma::rowvec eigenRow = arma::conv_to<arma::rowvec>::from(eigen);
  double result = arma::as_scalar(eigenRow * initial);
  return result;
}

// Check orthogonality of eigenvectors
bool checkOrthogonality(arma::mat eigvec) {
  bool correct = true;
  #pragma omp parallel for schedule(static) reduction(min:correct) collapse(2)
  for (int i = 0; i < eigvec.row(0).n_elem; i++) {
    for (int j = 0; j < eigvec.row(0).n_elem; j++) {
      double val = innerProduct(eigvec.col(i), eigvec.col(j));
      if (j != i) {
        if (val > 1E-9) {
          correct = false;
          // std::cout << "Break! val: " << val << std::endl;
          break;
        }
      }
      else if (j == i) {
        if (abs(val - 1.0f) > 1E-10) {
          correct = false;
          // std::cout << "Break2! val: " << val - 1.0f << std::endl;
          break;
        }
      }
    }
  }
  return correct;
}

// Get normalization of one eigenvector
double getNormalization(arma::vec eigen) {
  double norm = 0;
  #pragma omp parallel for schedule(static) reduction(+:norm)
  for (int i = 0; i < eigen.n_elem; i++) {
    norm += pow(abs(eigen(i)), 2);
  }
  return norm;
}

// Get normalization of one eigenvector
double getNormalization(arma::cx_vec eigen) {
  double norm = 0;
  #pragma omp parallel for schedule(static) reduction(+:norm)
  for (int i = 0; i < eigen.n_elem; i++) {
    norm += std::pow(std::abs(eigen(i)), 2);
  }
  return norm;
}

// Check normalization of eigenvectors
bool checkNormalization(arma::mat eigvec) {
  bool correct = true;
  #pragma omp parallel for schedule(static) reduction(min:correct)
  for (int i = 0; i < eigvec.row(0).n_elem; i++) {
    double norm = getNormalization(eigvec.col(i));
    if (abs(norm - 1.0f) > 1E-10) {
      correct = false;
      // std::cout << "Break norm! val: " << abs(norm - 1.0) << std::endl;
      // break;
    }
  }
  return correct;
}

// Get alpha coefficients for time evolution
arma::vec getAlphaCoefficients(arma::vec initial, arma::mat eigvec) {
  arma::vec alphas(eigvec.row(0).n_elem);
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < eigvec.row(0).n_elem; i++) {
    alphas(i) = innerProduct(eigvec.col(i), initial);
  }
  return alphas;
}

arma::cx_vec getSystemState(arma::mat eigvec, arma::vec eigenEnergy, arma::vec initialState, arma::vec alphas, double t) {
  arma::cx_vec state(eigvec.col(0).n_elem);
  state.fill(0);
  const arma::cx_double i(0, 1);
  for (int n = 0; n < alphas.n_elem; n++) {
    state += alphas(n) * std::exp(- i * eigenEnergy(n) * t) * eigvec.col(n);
    // std::cout << std::exp(- i * eigenEnergy(n) * t) << std::endl;
  }
  return state;
}
