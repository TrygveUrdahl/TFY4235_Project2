#pragma once

arma::sp_mat generateFDMMatrix(int n, double value, arma::vec xaxis,
                                double (*potential)(double), bool periodic);

double innerProduct(arma::vec eigen, arma::vec initial);

bool checkOrthogonality(arma::mat eigvec, int n);

double getNormalization(arma::vec eigen);

bool checkNormalization(arma::mat eigvec, int n);
