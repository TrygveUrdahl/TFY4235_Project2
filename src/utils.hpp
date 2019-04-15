#pragma once

arma::sp_mat generateFDMMatrix(int n, double value, arma::vec xaxis,
                                double (*potential)(double), bool periodic);
