#include <iostream>
#include <armadillo>
#include <cmath>
#include <chrono>
#include <omp.h>

#include "utils.hpp"

double testPotential(double x) {
  return 0;
}

double initialStateFunction(double x, int n) {
  return sqrt(2) * sin(M_PI * x); // /sqrt(n);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Requires argument(s): " << std::endl;
    std::cout << "\tint n: problem size. " << std::endl;
    return 1;
  }

  int n = atoi(argv[1]);
  double dx = 1/n;
  arma::vec xaxis = arma::linspace<arma::vec>(0, 1, n);

  auto system = generateFDMMatrix(n, 1, xaxis, testPotential, false);
  // std::cout << mat << std::endl;

  arma::vec eigenergy;
  arma::mat eigvec;
  arma::eigs_sym(eigenergy, eigvec, system, n - 1);

  bool orthogonal = checkOrthogonality(eigvec);
  std::cout << "Orthogonal: " << (orthogonal ? "success! " : "fail! ") << std::endl;
  bool normalized = checkNormalization(eigvec);
  std::cout << "Normalized: " << (normalized ? "success! " : "fail! ") << std::endl;

  arma::vec initialState = sqrt(0.5f) * eigvec.col(0) + sqrt(0.5f) * eigvec.col(1); // generateInitialState(initialStateFunction, xaxis);
  arma::vec alphas = getAlphaCoefficients(initialState, eigvec);
  arma::cx_vec state = getSystemState(eigvec, eigenergy, initialState, alphas, 0.5);
  arma::cx_vec state2 = getSystemState(eigvec, eigenergy, initialState, alphas, 1);
  double normalization = getNormalization(state);
  std::cout << "Normalization after time evolution: " << std::scientific << normalization << std::endl;

  state.save("../output/state.txt", arma::raw_ascii);
  state2.save("../output/state2.txt", arma::raw_ascii);
  // eigvec.save("../output/eigvecs.txt", arma::raw_ascii);
  // eigenergy.save("../output/eigvals.txt", arma::raw_ascii);

  return 0;
}
