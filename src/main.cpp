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
    std::cout << "\tint job: which time evolution scheme to use (0 for none, 1 for alphas, 2 for CN, default none)." << std::endl;


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
  int job = 0;
  if (argc > 3) {
    job = atoi(argv[3]);
  }

  double v0 = 1000;
  auto system = generateFDMMatrix(n - 2, dx2, xaxis, zeroPotential, v0, false);

  arma::vec eigenergy;
  arma::mat eigvec;
  solveSystem(&eigenergy, &eigvec, system, n, dx2);
  bool orthogonal = checkOrthogonality(eigvec);
  bool normalized = checkNormalization(eigvec);
  std::cout << "Eigenvectors orthogonal: " << (orthogonal ? "success! " : "fail! ") << std::endl;
  std::cout << "Eigenvectors normalized: " << (normalized ? "success! " : "fail! ") << std::endl;

  arma::vec initialState = generateDeltaInitialState(n); //sqrt(0.5) * eigvec.col(0) + sqrt(0.5) * eigvec.col(1);
  arma::vec alphas = getAlphaCoefficients(initialState, eigvec);
  double time = M_PI/(eigenergy(1) - eigenergy(0));
  arma::cx_mat states;

  if (job == 0) {
    std::cout << "No system time evolving. " << std::endl;
  }

  if (job == 1) {
    std::cout << "Evolving system by alphas. " << std::endl;
    states = getSystemStateEvolution(eigvec, eigenergy, initialState, alphas, dx, time, 1000);
  }
  else if (job == 2)
  {
    std::cout << "Running Crank-Nicolson. " << std::endl;
    arma::cx_vec initialStateCX(initialState, arma::vec(initialState.n_elem).zeros());
    states = evolveSystemCrankNicolson(initialStateCX, xaxis, potentialBarrier, v0, time);
  }

  if (save) {
    std::cout << "Saving files... " << std::endl;
    states.save("../output/state.txt", arma::raw_ascii);
    eigvec.save("../output/eigvecs.txt", arma::raw_ascii);
    eigenergy.save("../output/eigvals.txt", arma::raw_ascii);
  }
  else {
    std::cout << "No files saved. " << std::endl;
  }
  std::cout << "Program finished! " << std::endl;
  return 0;
}
