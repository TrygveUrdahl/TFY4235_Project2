#pragma once

arma::sp_mat generateFDMMatrix(int n, double value, const arma::vec &xaxis,
                                double (*potentialF)(double, double), double v0, bool periodic);

arma::vec generateInitialState(double (*function)(double, int), const arma::vec &xaxis);

void solveSystem(arma::vec *eigenergy, arma::mat *eigvec, const arma::sp_mat &system, int n, double dx2);

double innerProduct(const arma::vec &eigen, const arma::vec &initial);

bool checkOrthogonality(const arma::mat &eigvec);

double getNormalization(const arma::vec &eigen);
double getNormalization(const arma::cx_vec &eigen);

bool checkNormalization(const arma::mat &eigvec);

arma::vec getAlphaCoefficients(const arma::vec &initial, const arma::mat &eigvec);

arma::vec generateDeltaInitialState(int n);
