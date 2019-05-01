#!/usr/local/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

def LoadComplexData(file,**genfromtext_args):
    """
    Load complex data from the Armadillo C++ format in numpy.
    """
    array_as_strings = np.genfromtxt(file,dtype=str,**genfromtext_args)
    complex_parser = np.vectorize(lambda x: complex(*eval(x)))
    return complex_parser(array_as_strings)

def PlotVecAndEnergy(eigvecfile, eigvalfile):

    eigvec = np.loadtxt(eigvecfile, dtype=np.float64)
    eigval = np.loadtxt(eigvalfile, dtype=np.float64)
    dim = eigvec.shape[0]
    xaxis = np.linspace(0, 1, num=dim)


    # Plot eigenvectors / states
    # plt.subplot(2,1,1)
    for i in range(1, 4):
        plt.subplot(2,1,1)
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
        print("NormalizedNumeric: ", normalizedEigSum, " NormalizedAnalytic: ", normalizedStateSum)
        plt.subplot(2,1,2)
        plt.plot(xaxis, normalizedEig, '.', label="Eigenvector " + str(i))
        plt.plot(xaxis, normalizedState, '-', label="Analytic " + str(i))
        plt.xlabel("$x$")
        plt.ylabel("$|\Psi(x,t)|^2$")
        plt.legend(loc="best")
    plt.savefig("./output/eigenstates.png")
    plt.show()
    # Plot eigenvalues / energies
    #plt.subplot(3,1,3)
    for i in range(5):
        analytic = ((math.pi*(i + 2))**2)
        plt.plot([0, 1], [analytic, analytic], '-', label="Analytic " + str(i))
        plt.plot([0, 1], [eigval[i] * dim, eigval[i] * dim], '--', label="FDM " + str(i))
        # print("Analytic: ", analytic, " FDM: ", eigval[i])
    plt.xlabel("")
    plt.ylabel("Energy $\lambda_n$")
    plt.xticks([])
    plt.legend(loc="best")
    plt.savefig("./output/eigenenergies.png")
    plt.show()

def PlotOneStateComplex(stateFile):
    # Plot one state
    state = LoadComplexData(stateFile)[:,0]
    X = [val.real for val in state]
    Y = [val.imag for val in state]
    #print(Y)
    #plt.ylim(min(Y), max(Y))
    #plt.scatter(X, Y, label="State")
    dim = state.shape[0]
    xaxis = np.linspace(0, 1, num=dim)
    normalizedState = [abs(i)**2 for i in state]
    #plt.plot(xaxis, normalizedState, '-', label="Probability")

    plt.subplot(3,1,1)
    plt.plot(xaxis, X, '-', label="Real")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")

    plt.subplot(3,1,2)
    plt.plot(xaxis, Y, '-', label="Imaginary")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")

    plt.subplot(3,1,3)
    plt.plot(xaxis, normalizedState, label="Absolute square")
    plt.xlabel("x")
    plt.ylabel("$|\Psi(x,t)|^2$")
    plt.legend(loc="best")
    print("Normalization: ", sum(normalizedState))
    plt.show()

def PlotState(eigvecfile, eigvalfile):
    barrierHeight = 0.005
    eigvec = np.loadtxt(eigvecfile, dtype=np.float64)
    eigval = np.loadtxt(eigvalfile, dtype=np.float64)
    dim = eigvec.shape[0]
    xaxis = np.linspace(0, 1, num=dim)
    # Plot eigenvectors / states
    # plt.subplot(2,1,1)
    for i in range(1, 6):
        plt.subplot(2,1,1)
        plt.plot(xaxis, eigvec[:,i - 1], ".", label="Eigenvector " + str(i))
        plt.xlabel("$x/L$")
        plt.ylabel("$y$")
        plt.legend(loc="best")
        normalized = [abs(i)**2 for i in eigvec[:, i-1]]
        plt.subplot(2,1,2)
        plt.plot(xaxis, normalized, '.', label="Eigenvector " + str(i))
        plt.xlabel("$x/L$")
        plt.ylabel("$|\Psi(x,t)|^2$")
        plt.legend(loc="best")
    #plt.plot([1/3, 1/3], [0, barrierHeight], '-', color='black')
    #plt.plot([2/3, 2/3], [0, barrierHeight], '-', color='black')
    #plt.plot([1/3, 2/3], [barrierHeight, barrierHeight], '-', color='black')
    plt.savefig("./output/eigenstatesbarrier.png")
    plt.show()

    # Plot eigenvalues / energies
    #plt.subplot(3,1,3)
    for i in range(1,7):
        plt.plot([0, 1], [eigval[i] * dim, eigval[i] * dim], '--', label="Eigenenergy " + str(i))
        print("Eigenenergy: ", eigval[i] * dim)
    plt.title("")
    plt.xlabel("")
    plt.xticks([])
    plt.ylabel("Energy $\lambda_n$")
    plt.legend(loc="best")
    plt.savefig("./output/eigenenergiesbarrier.png")

    plt.show()

def initAnim():
    barrierHeight = 0.005
    plt.plot([1/3, 1/3], [0, barrierHeight], '-', color='black')
    plt.plot([2/3, 2/3], [0, barrierHeight], '-', color='black')
    plt.plot([1/3, 2/3], [barrierHeight, barrierHeight], '-', color='black')

def updateAnim(i, fig, line, states, xaxis, text, dt):
    line.set_ydata([abs(ii)**2 for ii in states[:,i]])
    text.set_text("t' = %f" % (i*dt))
    # print("Norm:", sum([abs(ii)**2 for ii in states[:,i]]))
    return line, text

def AnimatePlot(stateFile):
    states = LoadComplexData(stateFile)
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
    plt.ylim(0, 0.03)
    text = ax.text(0.5, 0.95, "t' = %f" % 0, transform=ax.transAxes, va="top", ha="center")
    ani = animation.FuncAnimation(fig, updateAnim, init_func=initAnim, fargs=(fig, line, states, xaxis, text, dt), frames=states.shape[1], interval=50, repeat=True)

    # ani.save("./output/animation.mp4")
    plt.show()

def ImshowPlot(stateFile):
    states = LoadComplexData(stateFile)
    dim = states.shape[0]
    tSteps = states.shape[1]
    dt = 1.0/tSteps
    xaxis = np.linspace(0, 1, num=dim)
    extent = [0, tSteps*dt, xaxis[0], xaxis[-1]]
    states = [abs(ii)**2 for ii in states]
    #states = [ii.real for ii in states]
    #states = [ii.imag for ii in states]

    fig, ax = plt.subplots()
    plt.imshow(states, extent=extent, aspect="auto")
    plt.title("Time evolution of $|\Psi(x,t)|^2$")
    plt.xlabel("Time $t/(2mL^2/\hbar)$")
    plt.ylabel("Position $x/L$")
    plt.colorbar()
    fig.tight_layout()

    plt.show()

# PlotVecAndEnergy("./output/eigvecs.txt", "./output/eigvals.txt")
# PlotOneStateComplex("./output/state.txt")
# PlotState("./output/eigvecs.txt", "./output/eigvals.txt");
# AnimatePlot("./output/state.txt")
ImshowPlot("./output/state.txt")
