#!/usr/local/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from math import sqrt, sin, cos, pi, exp

def f(lam):
    """
    Returns function value corresponding to equation (3.4)
    """
    barrier = 1000 #v0
    if (lam > barrier):
        return (lam - barrier)**2
    k = sqrt(lam)
    kappa = sqrt(barrier-lam)

    value = exp(kappa/3) * (kappa * sin(k/3) + k*cos(k/3))**2 - exp(-kappa/3) * (kappa * sin(k/3) - k * cos(k/3))**2

    return value

xaxis = np.linspace(0,1000, 100000)
yaxis = [f(x) for x in xaxis]

roots = []
for x in [73.8663877,73.8683378,293.2169877,293.2410202,647.6674955,648.1536705]:
    root = opt.fsolve(f, x)
    if root not in roots:
        roots.append(np.float64(root))
print("Roots:", roots)

plt.plot(xaxis, yaxis)
for i in range(len(roots)):
    root = roots[i]
    plt.plot([root, root], [0, max(yaxis)], label="Root " + str(i))
plt.legend(loc='best')
plt.xlabel("$\lambda$")
plt.ylabel("$f(\lambda)$")
plt.savefig("./output/roots.png")
plt.show()
