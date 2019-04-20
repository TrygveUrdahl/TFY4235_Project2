#include <iostream>
#include <armadillo>
#include <cmath>
#include <chrono>
#include <omp.h>

#include "utils.hpp"

double testPotential(double x) {
  return 0;
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

  arma::vec eigval;
  arma::mat eigvec;
  arma::eigs_sym(eigval, eigvec, system, n - 1);


  bool orthogonal = checkOrthogonality(eigvec, n);
  std::cout << "Orthogonal: " << (orthogonal ? "success! " : "fail! ") << std::endl;
  bool normalized = checkNormalization(eigvec, n);
  std::cout << "Normalized: " << (normalized ? "success! " : "fail! ") << std::endl;




  // eigvec.save("../output/testMat.txt", arma::raw_ascii);
  // eigval.save("../output/testVec.txt", arma::raw_ascii);

  return 0;
}
