#!/usr/local/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.optimize as opt
from math import sqrt, sin, cos, pi, exp


def f(lam):
    barrier = 1000 #v0
    k = sqrt(lam)
    kappa = sqrt(barrier-lam)

    value = exp(kappa/3) * (kappa * sin(k/3) + kappa*cos(k/3))**2 - exp(-kappa/3) * (kappa * sin(k/3) - k * cos(k/3))**2

    return value

xaxis = np.linspace(0,1000, 1000000)
yaxis = [f(x) for x in xaxis]

plt.plot(xaxis, yaxis)
plt.plot(xaxis, [0 for x in xaxis])
plt.show()

# print(opt.ridder(f, 20, 49.965))
