#include <iostream>
#include <armadillo>
#include <cmath>
#include <chrono>
#include <omp.h>

#include "utils.hpp"
#include "potentials.hpp"

double initialStateFunction(double x, int n) {
  return sqrt(2) * sin(M_PI * x); // /sqrt(n);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Requires argument(s): " << std::endl;
    std::cout << "\tint n: problem size. " << std::endl;
    std::cout << "Voluntary argument(s): " << std::endl;
    std::cout << "\tbool save: save files (0 or 1, defaults to 0). " << std::endl;

    throw std::logic_error("Program arguments are wrong! ");
  }

  int n = atoi(argv[1]);
  double dx = 1/n;
  arma::vec xaxis = arma::linspace<arma::vec>(0, 1, n);
  bool save = false;
  if (argc > 2) {
    save = atoi(argv[2]);
  }

  auto system = generateFDMMatrix(n, 1, xaxis, potentialBarrierMiddleThird, 1000, false);
  // auto system = generateFDMMatrix(n, 1.0f/(dx*dx), xaxis, testPotential, 1, false);
  // std::cout << system << std::endl;

  arma::vec eigenergy;
  arma::mat eigvec;
  solveSystem(&eigenergy, &eigvec, system, n);

  bool orthogonal = checkOrthogonality(eigvec);
  std::cout << "Orthogonal: " << (orthogonal ? "success! " : "fail! ") << std::endl;
  bool normalized = checkNormalization(eigvec);
  std::cout << "Normalized: " << (normalized ? "success! " : "fail! ") << std::endl;

  arma::vec initialState = eigvec.col(1); // sqrt(0.5f) * eigvec.col(0) + sqrt(0.5f) * eigvec.col(1); // generateInitialState(initialStateFunction, xaxis);
  arma::vec alphas = getAlphaCoefficients(initialState, eigvec);
  arma::cx_mat states = getSystemStateEvolution(eigvec, eigenergy, initialState, alphas, 0, 1000);
  arma::cx_mat states2 = getSystemStateEvolution(eigvec, eigenergy, initialState, alphas, 1, 1000);
  auto state = states.col(states.n_cols - 1);
  double normalization = getNormalization(state);
  std::cout << "Normalization after time evolution: " << std::scientific << normalization << std::endl;

  if (save) {
    states.save("../output/state.txt", arma::raw_ascii);
    states2.save("../output/state2.txt", arma::raw_ascii);
    eigvec.save("../output/eigvecs.txt", arma::raw_ascii);
    eigenergy.save("../output/eigvals.txt", arma::raw_ascii);
  }
  return 0;
}
