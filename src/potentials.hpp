#pragma once

double potentialBarrierMiddleThird(double x, double v0) {
  if (x < 0 || x > 1) {
    return std::numeric_limits<double>::max();
  }
  else if ((x > (1.0f/3.0f)) && (x < (2.0f/3.0f))) {
    return v0;
  }
  else {
    return 0;
  }
}
double zeroPotential(double x, double v0) {
  return 0;
}
