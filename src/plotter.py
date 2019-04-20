#!/usr/local/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import math

def LoadComplexData(file,**genfromtext_args):
    """
    Load complex data in the Armadillo C++ format in numpy.
    """
    array_as_strings = np.genfromtxt(file,dtype=str,**genfromtext_args)
    complex_parser = np.vectorize(lambda x: complex(*eval(x)))
    return complex_parser(array_as_strings)

"""
eigvec = np.loadtxt("./output/eigvecs.txt", dtype=np.float64)
eigval = np.loadtxt("./output/eigvals.txt", dtype=np.float64)
dim = eigvec.shape[0]
xaxis = np.linspace(0, 1, num=dim)


# Plot eigenvectors / states
# plt.subplot(2,1,1)
for i in range(1, 4):
    plt.subplot(3,1,1)
    plt.plot(xaxis, eigvec[:,i - 1], ".", label="Eigenvector " + str(i))
    state = [math.sqrt(2) * math.sin((i + 1) * math.pi * x)/math.sqrt(dim) for x in xaxis]
    plt.plot(xaxis, state, '-', label="Analytic " + str(i))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    error = sum([abs(state[j] - eigvec[j,i - 1]) for j in range(dim)])
    print("Error " + str(i) + ": ", error)
    normalizedEig = [abs(i)**2 for i in eigvec[:,i - 1]]
    normalizedEigSum = sum(normalizedEig)
    normalizedState = [abs(i)**2 for i in state]
    normalizedStateSum = sum(normalizedState)
    print("NormalizedEig: ", normalizedEigSum, " NormalizedState: ", normalizedStateSum)
    plt.subplot(3,1,2)
    plt.plot(xaxis, normalizedEig, '.', label="Eigenvector " + str(i))
    plt.plot(xaxis, normalizedState, '-', label="Analytic " + str(i))
    plt.xlabel("$x$")
    plt.ylabel("$p(x)$")
    plt.legend(loc="best")

# Plot eigenvalues / energies
plt.subplot(3,1,3)
for i in range(5):
    analytic = ((math.pi*(i + 2))**2) / dim
    plt.plot([0, 1], [analytic, analytic], '-', label="Analytic " + str(i))
    plt.plot([0, 1], [eigval[i] * dim, eigval[i] * dim], '--', label="FDM " + str(i))
    # print("Analytic: ", analytic, " FDM: ", eigval[i])
plt.xlabel("")
plt.ylabel("Energy")
plt.legend(loc="best")
"""

# Plot one state
state = LoadComplexData("./output/state2.txt")
X = [val.real for val in state]
Y = [val.imag for val in state]
#print(Y)
#plt.ylim(min(Y), max(Y))
#plt.scatter(X, Y, label="State")
dim = state.shape[0]
xaxis = np.linspace(0, 1, num=dim)
normalizedState = [abs(i)**2 for i in state]
#plt.plot(xaxis, normalizedState, '-', label="Probability")

plt.subplot(2,1,1)
plt.plot(xaxis, X, '-', label="Real")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")

plt.subplot(2,1,2)
plt.plot(xaxis, Y, '-', label="Imaginary")

plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")

plt.show()
