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
  for (int i = 0; i < n; i++) {
    mat(i, i) = function(xaxis(i));
  }
  return mat;
}

// Full FDM matrix with both Laplace and Energy operator
arma::sp_mat generateFDMMatrix(int n, double value, arma::vec xaxis, double (*potential)(double), bool periodic) {
  arma::sp_mat mat(n, n);
  mat += generateLaplaceOperator(n , value, periodic);
  mat += generateEnergyOperator(n, xaxis, potential);
  return mat;
}
