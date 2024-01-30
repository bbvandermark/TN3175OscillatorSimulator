# Heavily adapted from
# https://medium.com/quantum-untangled/hamiltonian-simulation-with-quantum-computation-fedc7cdc02e0


import numpy as np
from qiskit import Aer, execute
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms import TimeEvolutionProblem
from qiskit_algorithms.time_evolvers import TrotterQRTE
from qiskit.quantum_info import state_fidelity
import scipy
import matplotlib.pyplot as plt

from visualization_utils import create_animation


def simulate(N, trotter_steps, quantum_results=True, return_results=False):
    # Amount of values needed to encode all values of the spring constants
    _M = int(N * (N + 1) / 2)

    # Energy constant
    E = 1

    # Amount of qubits to use
    qubits = int(np.log2(2 * N ** 2))

    # The time interval and length to run the simulation
    n_timesteps = 100
    times = np.linspace(0, 10, n_timesteps)

    # Whether to print the values in case of simulation at
    # a single point in time
    print_values = (n_timesteps == 1)

    # Spring constants
    K = np.empty((N, N))
    if N == 2:
        K = [
            [1, 1],
            [1, 1]
        ]
    elif N == 4:
        K = [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ]
    else:
        print("N must be 2 or 4")
        exit()

    # Weights
    if N == 2:
        M = [
            [1, 0],
            [0, 1]
        ]
    elif N == 4:
        M = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]

    # Construct the Hamiltonian

    basis = []
    for i in range(N):
        for j in range(i, N):
            basis.append((i, j))

    B = np.zeros((N, _M))

    for i in range(_M):
        index = basis[i]
        k = index[0]  # First oscillator
        j = index[1]  # Second oscillator
        spring = K[k][j]
        mass = M[k][k]

        B[k][i] = np.sqrt(spring / mass)
        if k != j:
            mass = M[j][j]
            B[j][i] = -1 * np.sqrt(spring / mass)

    pad_width = N ** 2 - _M
    # Pad B to be N x N^2
    B = np.pad(B, [(0, 0), (0, pad_width)], mode='constant', constant_values=0)

    pad_width = N ** 2 - N
    # B padded to N^2 x N^2 to easily construct the Hamiltonian
    B_block = np.pad(B, [(0, pad_width), (0, 0)], mode='constant', constant_values=0)

    H = -1 * np.block([
        [np.zeros((N ** 2, N ** 2)), B_block],
        [B_block.T.conj(), np.zeros((N ** 2, N ** 2))]
    ])

    pauli_hamiltonian = SparsePauliOp.from_operator(H)

    # Construct the initial state
    state_label = ""
    for _ in range(qubits - 1):
        state_label += "0"
    state_label += "-"

    # Initial state with initial positions
    # state_label = "+0-"

    initial_state = Statevector.from_label(state_label)

    # Hamiltonian evolution
    results_quantum = []

    for time in times:
        evolution_problem = TimeEvolutionProblem(pauli_hamiltonian, time, initial_state)
        estimator = Estimator()
        trotter = TrotterQRTE(estimator=estimator, num_timesteps=trotter_steps)
        evolved_state = trotter.evolve(evolution_problem).evolved_state

        backend = Aer.get_backend('statevector_simulator')
        result = execute(evolved_state, backend).result()
        vec = result.get_statevector()
        qvec = vec / np.linalg.norm(vec)  # normalize

        results_quantum.append(qvec)
        if print_values:
            print(qvec)



    start = np.zeros(2 * N ** 2, dtype=complex)
    start[0] = 1 / np.sqrt(2 * E)
    start[1] = -1 * start[0]
    results_classical = []

    for time in times:
        simul = scipy.linalg.expm(-1j * H * time)
        vec = simul @ start
        cvec = vec / np.linalg.norm(vec)  # normalize

        results_classical.append(cvec)
        if print_values:
            print(cvec)

    if print_values:
        state_fidelity(qvec, cvec)

    if quantum_results:
        results = results_quantum
    else:
        results = results_classical

    # Get the velocities

    final_velocities = []
    for time in range(n_timesteps):
        velocities = np.zeros(N)

        for i in range(N):
            velocities[i] = results[time][i].real

        velocities = np.matmul(np.linalg.inv(np.sqrt(M)), velocities)
        velocities *= np.sqrt(2 * E)
        final_velocities.append(velocities)
        if print_values:
            for i in range(N):
                print("x'_{} = {}".format(i, velocities[i]))

    # Get the positions

    final_positions = []
    for time in range(n_timesteps):
        solved = dict()
        unsolved = []

        for i in range(_M):
            index = basis[i]
            k = index[0]  # First oscillator
            j = index[1]  # Second oscillator
            i += N ** 2

            if K[k][j] != 0:
                result = -1j * np.sqrt(2 * E) * (results[time][i] / np.sqrt(K[k][j]))
                result = result.real
                if k == j:
                    # Add new entry to the dictionairy
                    solved[k] = result
                else:
                    unsolved.append((k, j, result))

        # Solve all the equations of the form x_k - x_j
        while len(unsolved) != 0:
            for i in range(len(unsolved)):
                problem = unsolved[i]
                k = problem[0]
                j = problem[1]
                result = problem[2]
                # k - j = result
                if k in solved:
                    solved[j] = solved[k] - result
                    unsolved.remove(problem)
                    break
                elif j in solved:
                    solved[k] = solved[j] + result
                    unsolved.remove(problem)
                    break

        positions = []
        for problem in sorted(solved):
            positions.append(solved[problem])
            if print_values:
                print("x_{} = {}".format(problem, solved[problem]))

        final_positions.append(positions)

    if return_results:
        return final_positions, final_velocities

    plot_label = []
    for i in range(N):
        plot_label.append(i + 1)

    plot_title = f"Trotter steps = {trotter_steps}"

    plt.plot(times, final_velocities, label=plot_label)
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.legend(title="Oscillators")
    plt.title(plot_title)
    plt.show()

    plt.clf()

    plt.plot(times, final_positions, label=plot_label)
    plt.xlabel("Time")
    plt.ylabel("Displacement")
    plt.legend(title="Oscillators")
    plt.title(plot_title)
    plt.show()

    #  animation showing how the two masses move
    t_vals = times
    x_vals = [[] for _ in range(N)]
    for i in range(N):
        for j in range(len(final_positions)):
            x_vals[i].append(final_positions[j][i])

    create_animation(t_vals, x_vals, f"quantum_{N}.gif")


def example():
    print("Simulation of 2 masses, each coupled to each other and to a wall")
    simulate(2, 10)
    print("Simulation of 4 masses, each coupled to each other. Only the first mass is coupled to a wall")
    simulate(4, 10)
