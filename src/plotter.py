#!/usr/local/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import h5py
import pandas as pd
"""
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'CMU Serif Roman 2'
"""

def LoadComplexData(fileReal, fileImag,**genfromtext_args):
    """
    Load complex data from the C++ format as HDF5 in numpy.
    """
    hfReal = h5py.File(fileReal, 'r')
    hfImag = h5py.File(fileImag, 'r')

    statesReal = hfReal.get('dataset').value
    statesImag = hfImag.get('dataset').value

    states = statesReal + statesImag * 1j
    states = states.transpose()
    return states


def CompareErrors(eigvalfile):
    """
    Compare calculated eigenvalues versus analytic eigenvalues, and how the error
    changes with increasing n.
    """
    eigval = np.loadtxt(eigvalfile, dtype=np.float64)
    dim = eigval.shape[0]
    analytic = [((math.pi*(i + 1))**2) for i in range(dim)]
    xaxis = [i for i in range(dim)]
    error = [abs(analytic[i]-eigval[i])**2 for i in range(dim) ]

    fig, ax = plt.subplots()
    plt.subplot(2,1,1)
    plt.semilogx(xaxis, eigval, label="Calculated")
    plt.semilogx(xaxis, analytic, label="Analytic")
    plt.xlabel("$n$")
    plt.ylabel("$\lambda_n$")
    plt.legend(loc="best")
    fig.tight_layout()

    plt.subplot(2,1,2)
    plt.semilogx(xaxis, error, label="Rel. error")
    plt.xlabel("$n$")
    plt.ylabel("$|{analytic}_n - \lambda_n|^2$")
    plt.legend(loc="best")
    fig.tight_layout()
    plt.savefig("./output/errors.png")
    plt.show()


def PlotVecAndEnergy(eigvecfile, eigvalfile):
    """
    Plot eigenstates and eigenenergies and compare to a known analytic solution
    (for the zeroPotential well).
    """
    eigvec = np.loadtxt(eigvecfile, dtype=np.float64)
    eigval = np.loadtxt(eigvalfile, dtype=np.float64)
    dim = eigvec.shape[0]
    xaxis = np.linspace(0, 1, num=dim)

    fig, ax = plt.subplots()
    # Plot eigenvectors / states
    # plt.subplot(2,1,1)
    for i in range(0, 3):
        plt.subplot(2,1,1)
        plt.plot(xaxis, eigvec[:,i] * math.sqrt(dim), ".", label="Eigenvector " + str(i))
        state = [math.sqrt(2) * math.sin((i + 1) * math.pi * x) for x in xaxis]
        plt.plot(xaxis, state, '-', label="Analytic " + str(i))
        plt.xlabel("$x/L$")
        plt.ylabel("y")
        plt.legend(loc="best")
        error = sum([abs(state[j] - eigvec[j,i]) for j in range(dim)])
        print("Error " + str(i) + ": ", error)
        normalizedEig = [abs(i)**2 for i in eigvec[:,i]]
        normalizedEigSum = sum(normalizedEig)
        normalizedState = [abs(i/math.sqrt(dim))**2 for i in state]
        normalizedStateSum = sum(normalizedState)
        print("NormalizedNumeric: ", normalizedEigSum, " NormalizedAnalytic: ", normalizedStateSum)
        plt.subplot(2,1,2)
        plt.plot(xaxis, normalizedEig, '.', label="Eigenvector " + str(i))
        plt.plot(xaxis, normalizedState, '-', label="Analytic " + str(i))
        plt.xlabel("$x/L$")
        plt.ylabel("$|\Psi(x,t)|^2$")
        plt.legend(loc="best")
    fig.tight_layout()
    plt.savefig("./output/eigenstates.png")
    plt.show()
    # Plot eigenvalues / energies
    #plt.subplot(3,1,3)
    for i in range(6):
        analytic = ((math.pi*(i + 1))**2)
        plt.plot([0, 1], [analytic, analytic], '-', label="Analytic " + str(i))
        plt.plot([0, 1], [eigval[i], eigval[i]], '--', label="FDM " + str(i))
        print("Analytic: ", analytic, " FDM: ", eigval[i])
    plt.xlabel("")
    plt.ylabel("Energy $\lambda_n$")
    plt.xticks([])
    plt.legend(loc="best")
    fig.tight_layout()
    plt.savefig("./output/eigenenergies.png")
    plt.show()

def PlotOneStateComplex(stateFileReal, stateFileImag, t):
    """
    Plot one state from a complex state file. The state to plot can be chosen.
    """
    # Plot one state
    state = LoadComplexData(stateFileReal, stateFileImag)[:,t]
    X = [val.real for val in state]
    Y = [val.imag for val in state]
    #print(Y)
    #plt.ylim(min(Y), max(Y))
    #plt.scatter(X, Y, label="State")
    dim = state.shape[0]
    xaxis = np.linspace(0, 1, num=dim)
    normalizedState = [abs(i)**2 for i in state]
    #plt.plot(xaxis, normalizedState, '-', label="Probability")
    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].tick_params(axis='x', bottom=False)
    axes[0].plot(xaxis, X, '-', label="Real")
    axes[0].set_ylabel("$Re(\Psi(x,1))$")
    axes[0].legend(loc="best")

    axes[1].tick_params(axis='x', bottom=False)
    axes[1].plot(xaxis, Y, '-', label="Imaginary")
    axes[1].set_ylabel("$Im(\Psi(x,1))$")
    axes[1].legend(loc="best")

    axes[2].plot(xaxis, normalizedState, label="Absolute square")
    axes[2].set_xlabel("x/L")
    axes[2].set_ylabel("$|\Psi(x,1)|^2$")
    axes[2].legend(loc="best")
    print("Normalization: ", sum(normalizedState))
    fig.tight_layout(h_pad=0.01)
    # plt.savefig("./output/deltafunct1.png")
    plt.show()

def PlotState(eigvecfile, eigvalfile):
    """
    Plot eigenvectors and eigenenergies for a solved system.
    """
    barrierHeight = 0.005
    eigvec = np.loadtxt(eigvecfile, dtype=np.float64)
    eigval = np.loadtxt(eigvalfile, dtype=np.float64)
    dim = eigvec.shape[0]
    xaxis = np.linspace(0, 1, num=dim)
    fig, ax = plt.subplots()
    # Plot eigenvectors / states
    # plt.subplot(2,1,1)
    for i in range(1, 5):
        plt.subplot(2,1,1)
        plt.plot(xaxis, eigvec[:,i - 1], ".", label="Eigenvector " + str(i - 1))
        plt.xlabel("$x/L$")
        plt.ylabel("$y$")
        plt.legend(loc="best")
        normalized = [abs(i)**2 for i in eigvec[:, i-1]]
        plt.subplot(2,1,2)
        plt.plot(xaxis, normalized, '.', label="Eigenvector " + str(i - 1))
        plt.xlabel("$x/L$")
        plt.ylabel("$|\Psi(x,t)|^2$")
        plt.legend(loc="best")
    #plt.plot([1/3, 1/3], [0, barrierHeight], '-', color='black')
    #plt.plot([2/3, 2/3], [0, barrierHeight], '-', color='black')
    #plt.plot([1/3, 2/3], [barrierHeight, barrierHeight], '-', color='black')
    fig.tight_layout()
    plt.savefig("./output/eigenstatesbarrier.png")
    plt.show()

    # Plot eigenvalues / energies
    #plt.subplot(3,1,3)
    for i in range(0,6):
        plt.plot([0, 1], [eigval[i], eigval[i]], '--', label="Eigenenergy " + str(i))
        print("Eigenenergy: ", eigval[i])
    plt.title("")
    plt.xlabel("")
    plt.xticks([])
    plt.ylabel("Energy $\lambda_n$")
    plt.legend(loc="best")
    fig.tight_layout()
    plt.savefig("./output/eigenenergiesbarrier.png")

    plt.show()

def initAnim():
    barrierHeight = 0.005
    barrier = True
    if (barrier):
        plt.plot([1/3, 1/3], [0, barrierHeight], '-', color='black')
        plt.plot([2/3, 2/3], [0, barrierHeight], '-', color='black')
        plt.plot([1/3, 2/3], [barrierHeight, barrierHeight], '-', color='black')

def updateAnim(i, fig, line, states, xaxis, text, dt):
    line.set_ydata([abs(ii)**2 for ii in states[:,i]])
    text.set_text("t' = %f" % (i*dt))
    # print("Norm:", sum([abs(ii)**2 for ii in states[:,i]]))
    return line, text

def AnimatePlot(stateFileReal, stateFileImag):
    """
    Animate time evolution of a system with a barrier from x = 1/3 to x = 2/3
    (barrier can be togled by variable "barrier" in initAnim).
    """
    states = LoadComplexData(stateFileReal, stateFileImag)
    # states = states[:,0:-1:100]
    dt = 1/states.shape[1]
    fig, ax = plt.subplots()
    dim = states.shape[0]
    xaxis = np.linspace(0, 1, num=dim)
    line, = ax.plot(xaxis, [abs(ii)**2 for ii in states[:,0]])
    #line, = ax.plot(xaxis, [i.real for i in states[:,0]])
    plt.title("Probability at time $t'= t/2mL/\hbar$")
    ax.set_xlabel("Position $x/L$")
    ax.set_ylabel("$|\Psi(x,t)|^2$")
    plt.xlim(0, 1)
    plt.ylim(0, 0.15)
    text = ax.text(0.5, 0.95, "t' = %f" % 0, transform=ax.transAxes, va="top", ha="center")
    ani = animation.FuncAnimation(fig, updateAnim, init_func=initAnim, fargs=(fig, line, states, xaxis, text, dt), frames=states.shape[1], interval=200, repeat=True)

    plt.show()

def ImshowPlot(stateFileReal, stateFileImag, eigvalfile):
    """
    Plot time evolution of a system
    """
    states = LoadComplexData(stateFileReal, stateFileImag)
    energy = np.loadtxt(eigvalfile, dtype=np.float64)
    dim = states.shape[0]
    tSteps = states.shape[1]
    dt = math.pi/(energy[1] - energy[0])/tSteps
    t = 2 * math.pi/(energy[1] - energy[0])# tSteps * dt
    xaxis = np.linspace(0, 1, num=dim)
    extent = [0, t, xaxis[0], xaxis[-1]]
    statesRe = [ii.real for ii in states]
    statesIm = [ii.imag for ii in states]
    states = [abs(ii)**2 for ii in states]

    fig, ax = plt.subplots()
    plt.imshow(states, extent=extent, aspect="auto")
    plt.title("Time evolution of $|\Psi(x,t)|^2$")
    plt.xlabel("Time $t/(2mL^2/\hbar)$")
    plt.ylabel("Position $x/L$")
    plt.colorbar()
    fig.tight_layout()

    # plt.savefig("./output/cntunneling.png")
    # plt.savefig("./output/deltaevolve.png")
    plt.show()

def ImshowPlotSubplots(stateFileReal, stateFileImag, eigvalfile):
    """
    Plot time evolution of a system with real and imaginary values.
    """
    states = LoadComplexData(stateFileReal, stateFileImag)

    energy = np.loadtxt(eigvalfile, dtype=np.float64)
    dim = states.shape[0]
    tSteps = states.shape[1]
    dt = math.pi/(energy[1] - energy[0])/tSteps
    t = 2 * math.pi/(energy[1] - energy[0])# tSteps * dt
    xaxis = np.linspace(0, 1, num=dim)
    extent = [0, t, xaxis[0], xaxis[-1]]
    statesRe = [ii.real for ii in states]
    statesIm = [ii.imag for ii in states]
    states = [abs(ii)**2 for ii in states]

    fig, ax = plt.subplots()
    plt.subplot(3,1,1)
    plt.imshow(statesRe, extent=extent, aspect="auto")
    plt.title("Time evolution of $Re|\Psi(x,t)|^2$")
    plt.xlabel("Time $t/(2mL^2/\hbar)$")
    plt.ylabel("Position $x/L$")
    plt.colorbar()

    plt.subplot(3,1,2)
    plt.imshow(statesIm, extent=extent, aspect="auto")
    plt.title("Time evolution of $Im|\Psi(x,t)|^2$")
    plt.xlabel("Time $t/(2mL^2/\hbar)$")
    plt.ylabel("Position $x/L$")
    plt.colorbar()

    plt.subplot(3,1,3)
    plt.imshow(states, extent=extent, aspect="auto")
    plt.title("Time evolution of $|\Psi(x,t)|^2$")
    plt.xlabel("Time $t/(2mL^2/\hbar)$")
    plt.ylabel("Position $x/L$")
    plt.colorbar()
    fig.tight_layout()

    # plt.savefig("./output/alphatunneling.png")
    # plt.savefig("./output/deltaevolve.png")
    plt.show()

# CompareErrors("./output/eigvals.txt")
# PlotVecAndEnergy("./output/eigvecs.txt", "./output/eigvals.txt")
# PlotOneStateComplex("./output/stateReal.h5", "./output/stateImag.h5", 2)
# PlotState("./output/eigvecs.txt", "./output/eigvals.txt");
# AnimatePlot("./output/stateReal.h5", "./output/stateImag.h5")
# ImshowPlotSubplots("./output/stateReal.h5", "./output/stateImag.h5", "./output/eigvals.txt")
ImshowPlot("./output/stateReal.h5", "./output/stateImag.h5", "./output/eigvals.txt")
