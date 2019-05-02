#include <iostream>
#include <armadillo>
#include <cmath>
#include <chrono>
#include <omp.h>

#include "utils.hpp"
#include "potentials.hpp"
#include "timeEvolution.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Requires argument(s): " << std::endl;
    std::cout << "\tint n: problem size. " << std::endl;
    std::cout << "Voluntary argument(s): " << std::endl;
    std::cout << "\tbool save: save files (0 or 1, defaults to 0). " << std::endl;

    throw std::logic_error("Program arguments are wrong! ");
  }

  int n = atoi(argv[1]);
  int k = n - 3;
  double dx = 1.0/(n - 1);
  double dx2 = dx*dx;
  arma::vec xaxis = arma::linspace<arma::vec>(0, 1, n);
  bool save = false;
  if (argc > 2) {
    save = atoi(argv[2]);
  }
  double v0 = 1000;
  auto system = generateFDMMatrix(n - 2, dx2, xaxis, potentialBarrier, v0, false);
  // std::cout << arma::mat(system) << std::endl;

  arma::vec eigenergy;
  arma::mat eigvec;
  solveSystem(&eigenergy, &eigvec, system, n, dx2);
  eigenergy.save("../output/eigvals.txt", arma::raw_ascii);
  bool orthogonal = checkOrthogonality(eigvec);
  std::cout << "Orthogonal: " << (orthogonal ? "success! " : "fail! ") << std::endl;
  bool normalized = checkNormalization(eigvec);
  std::cout << "Normalized: " << (normalized ? "success! " : "fail! ") << std::endl;

  arma::vec initialState = sqrt(0.5) * eigvec.col(0) + sqrt(0.5) * eigvec.col(1);
  arma::vec alphas = getAlphaCoefficients(initialState, eigvec);
  double time = M_PI/(eigenergy(1)-eigenergy(0));
  //arma::cx_mat states = getSystemStateEvolution(eigvec, eigenergy, initialState, alphas, dx, time, 1000);
  arma::cx_vec initialStateCX(initialState, arma::vec(initialState.n_elem).zeros());
  arma::cx_mat states = evolveSystemCrankNicolson(initialStateCX, xaxis, potentialBarrier, v0, time);


  if (save) {
    // states.save("../output/state.txt", arma::raw_ascii);
    states.save("../output/state.txt", arma::raw_ascii);
    eigvec.save("../output/eigvecs.txt", arma::raw_ascii);
    //eigenergy.save("../output/eigvals.txt", arma::raw_ascii);
  }
  return 0;
}
