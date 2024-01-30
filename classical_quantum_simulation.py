import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from matrix_utils import generate_m, generate_k, generate_f, generate_b
from visualization_utils import create_animation


def simulate(M, K, x0, tf):
    dim = len(M)

    M = generate_m(M, dim)
    K = generate_k(K, dim)
    F = generate_f(K, dim)
    A = -np.matmul(np.linalg.inv(M), F)
    A = generate_b(A, dim)

    def f(t, x):  # /!\ changed the order of the arguments for solve_ivp
        return A.dot(x)

    # compute analytical solution for the linear system
    # dx/dt = Ax  --> x(t) = exp(t*A) * x(0)
    dt_exp = 0.05
    t_exp = np.arange(0, tf, dt_exp)  # time points
    exp_dtA = scipy.linalg.expm(dt_exp * A)  # used to compute the solution during one time step
    sol_exp = [x0]
    for t in t_exp[1:]:
        sol_exp.append(exp_dtA.dot(sol_exp[-1]))
    sol_exp = np.array(sol_exp).T

    # position plotting
    plt.figure()
    for i in range(len(M)):
        label = i + 1
        plt.plot(t_exp, sol_exp[2 * i, :], label=label, linestyle='-', marker=None)

    plt.ylabel('position (m)')
    plt.xlabel('t (s)')
    plt.grid()
    plt.title('Positions')
    plt.legend()
    plt.show()

    # velocity plotting
    plt.figure()
    for i in range(len(M)):
        label = i + 1
        plt.plot(t_exp, sol_exp[2 * i + 1, :], label=label, linestyle='-', marker=None)

    plt.ylabel('velocity (m/s)')
    plt.xlabel('t (s)')
    plt.grid()
    plt.title('Velocities')
    plt.legend()
    plt.show()

    #  animation showing how the two masses move
    t_vals = np.linspace(0, 10, 200)
    x_1_vals = sol_exp[0, :]
    x_2_vals = sol_exp[2, :]

    create_animation(t_vals, [x_1_vals, x_2_vals], "quantum_classical.gif")


def example():
    print("Simulation of 2 masses, each coupled to each other and to a wall")
    print("This simulation uses classical algorithms to perfectly calculate what the quantum simulation does")
    M = [1, 1]
    K = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]  # [node 1, node 2, spring constant], equal nodes = wall connection
    x0 = np.array([1, 1, 2, -1])  # initial conditions in order: (x1, v1, x2, v2.....x_n, v_n)
    tf = 10  # final time

    simulate(M, K, x0, tf)
