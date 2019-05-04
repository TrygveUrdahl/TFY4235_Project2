#include <iostream>
#include <armadillo>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <hdf5.h>

#include "utils.hpp"
#include "potentials.hpp"
#include "timeEvolution.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Requires argument(s): " << std::endl;
    std::cout << "\tint n: problem size. " << std::endl;
    std::cout << "Voluntary argument(s): " << std::endl;
    std::cout << "\tbool save: save files (0 or 1, defaults to 0). " << std::endl;
    std::cout << "\tint job: which time evolution scheme to use " << std::endl;
    std::cout << "\t  (0 for none, 1 for alphas, 2 for CN, 3 for Forward Euler, default none)." << std::endl;
    std::cout << "\tint initial: which initial state to use " << std::endl;
    std::cout << "\t  (0 for ground state, 1 for superposition, 2 for delta, default superposition)." << std::endl;

    throw std::logic_error("Program arguments are wrong! ");
  }

  int n = atoi(argv[1]);
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
  int initial = 1;
  if (argc > 3) {
    initial = atoi(argv[4]);
  }
  double v0 = 500;
  auto system = generateFDMMatrix(n - 2, dx2, xaxis, potentialBarrier, v0, false);

  arma::vec eigenergy;
  arma::mat eigvec;
  solveSystem(&eigenergy, &eigvec, system, n, dx2);
  bool orthogonal = checkOrthogonality(eigvec);
  bool normalized = checkNormalization(eigvec);
  std::cout << "Eigenvectors orthogonal: " << (orthogonal ? "success! " : "fail! ") << std::endl;
  std::cout << "Eigenvectors normalized: " << (normalized ? "success! " : "fail! ") << std::endl;

  arma::vec initialState;
  if (initial == 0) {
    std::cout << "Using ground state. " << std::endl;
    initialState = eigvec.col(0);
  }
  if (initial == 1) {
    std::cout << "Using superposition. " << std::endl;
    initialState = sqrt(0.5) * eigvec.col(0) + sqrt(0.5) * eigvec.col(1);
  }
  if (initial == 2) {
    std::cout << "Using delta state. " << std::endl;
    initialState = generateDeltaInitialState(n);
  }
  double time = M_PI/(eigenergy(1) - eigenergy(0));
  arma::cx_mat states;

  if (job == 0) {
    std::cout << "No system time evolving. " << std::endl;
  }
  else if (job == 1) {
    std::cout << "Evolving system by alphas. " << std::endl;
    arma::vec alphas = getAlphaCoefficients(initialState, eigvec);
    states = getSystemStateEvolution(eigvec, eigenergy, initialState, alphas, dx, time, 500);
  }
  else if (job == 2) {
    std::cout << "Running Crank-Nicolson. " << std::endl;
    arma::cx_vec initialStateCX(initialState, arma::vec(initialState.n_elem).zeros());
    states = evolveSystemCrankNicolson(initialStateCX, xaxis, potentialBarrier, v0, time);
  }
  else if (job == 3) {
    std::cout << "Running Forward Euler. " << std::endl;
    arma::cx_vec initialStateCX(initialState, arma::vec(initialState.n_elem).zeros());
    states = evolveSystemForwardEuler(initialStateCX, xaxis, potentialBarrier, v0, 1, 100);
  }

  if (save) {
    std::cout << "Saving files... " << std::endl;
    arma::mat statesReal = arma::real(states);
    arma::mat statesImag = arma::imag(states);
    statesReal.save("../output/stateReal.h5", arma::hdf5_binary);
    statesImag.save("../output/stateImag.h5", arma::hdf5_binary);
    eigvec.save("../output/eigvecs.txt", arma::raw_ascii);
    eigenergy.save("../output/eigvals.txt", arma::raw_ascii);
  }
  else {
    std::cout << "No files saved. " << std::endl;
  }
  std::cout << "Program finished! " << std::endl;
  return 0;
}
