#pragma once

double potentialBarrierMiddleThird(double x) {
  if ((x > (1.0f/3.0f)) && (x < (2.0f/3.0f))) {
    return 1;
  }
  else {
    return 0;
  }
}
double zeroPotential(double x) {
  return 0;
}
