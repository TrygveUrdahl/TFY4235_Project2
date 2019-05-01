#pragma once

double potentialBarrier(double x, double v0) {
  if (x < 0 || x > 1) {
    return std::numeric_limits<double>::max();
  }
  else if ((x > (1.0/3.0)) && (x < (2.0/3.0))) {
    return v0;
  }
  else {
    return 0;
  }
}
double zeroPotential(double x, double v0) {
  if (x < 0 || x > 1) {
    return std::numeric_limits<double>::max();
  }
  return 0;
}

double variablePotential(double x, double v0) {
  if (x < 0 || x > 1) {
    return std::numeric_limits<double>::max();
  }
  else if (x < (1.0/3.0)) {
    return 0;
  }
  else if ((x > (1.0/3.0)) && (x < (2.0/3.0))) {
    return v0;
  }
  else {
    return 0;
  }
}
