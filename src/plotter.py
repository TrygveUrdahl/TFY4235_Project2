#!/usr/local/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import math

eigvec = np.loadtxt("./output/testMat.txt", dtype=np.float64)
eigval = np.loadtxt("./output/testVec.txt", dtype=np.float64)
xaxis = np.linspace(0, 1, eigvec.shape[0])


# Plot eigenvectors
# plt.subplot(2,1,1)
for i in range(3):
    plt.plot(xaxis, 16*eigvec[:,i], ".", label="Eigenvector " + str(i))
    state = [math.sqrt(2) * math.sin((i + 2) * math.pi * x) for x in xaxis]
    plt.plot(xaxis, state, '-', label="Analytic " + str(i))
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")

"""
# Plot eigenvalues
plt.subplot(2,1,2)
for i in range(5):
    plt.plot([0, 1], [eigval[i], eigval[i]], '--', label="FDM " + str(i))
    analytic = (math.pi*i)**2
    plt.plot([0, 1], [analytic, analytic], '-', label="Analytic " + str(i))
    print("Analytic: ", analytic, " FDM: ", eigval[i])
plt.xlabel("")
plt.ylabel("Energy")
plt.legend(loc="best")
"""
plt.show()
