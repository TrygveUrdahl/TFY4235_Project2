#pragma once

arma::sp_mat generateFDMMatrix(int n, double value, arma::vec xaxis,
                                double (*potential)(double), bool periodic);

arma::vec generateInitialState(double (*function)(double, int), arma::vec xaxis);

double innerProduct(arma::vec eigen, arma::vec initial);

bool checkOrthogonality(arma::mat eigvec);

double getNormalization(arma::vec eigen);
double getNormalization(arma::cx_vec eigen);


bool checkNormalization(arma::mat eigvec);

arma::vec getAlphaCoefficients(arma::vec initial, arma::mat eigvec);

arma::cx_vec getSystemState(arma::mat eigvec, arma::vec eigenEnergy, arma::vec initialState, arma::vec alphas, double t);
