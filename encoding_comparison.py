import matplotlib.pyplot as plt
import numpy as np
from qiskit import Aer, transpile
from qiskit.quantum_info import Statevector

from matrix_utils import next_power_of_2, generate_m, generate_k, construct_hamiltonian, hamiltonian_evolution
from visualization_utils import create_animation


def simulate_encoding_one(masses, spring_constants, initial_pos, initial_vel):
    qubits = next_power_of_2(len(masses))
    dim = 2 ** qubits

    M = generate_m(masses, dim)
    K = generate_k(spring_constants, dim)

    F = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i != j:
                F[i][j] = -K[i][j]
            else:
                for l in range(dim):
                    F[i][i] += K[i][l]

    A = np.matmul(np.linalg.inv(np.sqrt(M)), np.matmul(F, np.linalg.inv(np.sqrt(M))))
    B = np.linalg.cholesky(A)
    B_dagger = np.conjugate(B).T

    initial_pos = np.matmul(np.sqrt(M), initial_pos)

    initial_vel = np.matmul(np.sqrt(M), initial_vel)

    def create_evolution(time):
        first_comp = initial_vel
        second_comp = 1j * np.matmul(B_dagger, initial_pos)

        initial_state = np.concatenate((first_comp, second_comp))

        norm = np.linalg.norm(initial_state)

        initial_vec = Statevector(initial_state / np.linalg.norm(initial_state))

        [hamiltonian, qubits] = construct_hamiltonian(masses, spring_constants)
        evolution_one = hamiltonian_evolution(hamiltonian, qubits, initial_state=initial_vec, time=time)
        return [evolution_one, norm]

    times = np.linspace(0, 10, 100)
    counts = []
    res1 = []
    res2 = []
    vel1 = []
    vel2 = []
    for i in times:
        [qc, norm] = create_evolution(i)

        result_one = np.array(Statevector(qc))

        l = int(len(result_one) / 2)

        result_one = result_one * norm
        vel = result_one[:l]
        pos = result_one[l:]

        # reverting the encoding to normal positions/velocities
        pos = pos * (-1j)
        pos = np.matmul(np.linalg.inv(B_dagger), pos)

        pos = np.matmul(np.linalg.inv(np.sqrt(M)), pos)
        vel = np.matmul(np.linalg.inv(np.sqrt(M)), vel)
        res1.append(pos[0])
        res2.append(pos[1])
        vel1.append(vel[0])
        vel2.append(vel[1])

    plt.plot(times, res1)
    plt.plot(times, vel1)
    plt.show()
    plt.plot(times, res2)
    plt.plot(times, vel2)
    plt.show()

    create_animation(times, [res1, res2], "quantum_encoding_1.gif")


# second type of encoding

def simulate_encoding_two(masses, spring_constants, initial_pos, initial_vel):
    qubits = next_power_of_2(len(masses))
    dim = 2 ** qubits

    M = generate_m(masses, dim)
    K = generate_k(spring_constants, dim)

    F = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i != j:
                F[i][j] = -K[i][j]
            else:
                for l in range(dim):
                    F[i][i] += K[i][l]

    A = np.matmul(np.linalg.inv(np.sqrt(M)), np.matmul(F, np.linalg.inv(np.sqrt(M))))
    B = np.linalg.cholesky(A)

    A_pseudoinv = np.linalg.pinv(A)
    B_pseudoinv = np.linalg.pinv(B)
    P = np.eye(dim) - np.matmul(A_pseudoinv, A)

    def evolve2(time):
        first_comp = np.matmul(P, initial_pos)
        second_comp = -1j * np.matmul(np.matmul(B_pseudoinv, P), initial_vel)

        initial_state = np.concatenate((first_comp, second_comp))

        norm_two = np.linalg.norm(initial_state)

        initial_vec = Statevector(initial_state / np.linalg.norm(initial_state))

        [hamiltonian, qubits] = construct_hamiltonian(masses, spring_constants)
        evolution_two = hamiltonian_evolution(hamiltonian, qubits, initial_state=initial_vec, time=time)

        return [evolution_two, norm_two]

    # plot results second encoding

    times = np.linspace(0, 10, 100)
    counts = []
    res = []
    res_v = []
    for i in times:
        [qc, norm_two] = evolve2(i)
        result_two = np.array(Statevector(qc))
        l = int(len(result_two) / 2)

        result_two = result_two * norm_two

        vel = result_two[l:]
        pos = result_two[:l]

        pos = np.matmul(np.linalg.inv(P), pos)

        vel = vel * 1j
        vel = np.matmul(np.linalg.inv(P), np.matmul(np.linalg.inv(B_pseudoinv), vel))

        pos = np.matmul(np.linalg.inv(np.sqrt(M)), pos)
        vel = np.matmul(np.linalg.inv(np.sqrt(M)), vel)

        qc.measure_all()
        simulator = Aer.get_backend('aer_simulator')
        circ = transpile(qc, simulator)
        result = simulator.run(circ).result()
        counts = result.get_counts(qc)
        res.append(pos[0])
        res_v.append(vel[0])

    plt.plot(times, res)
    plt.plot(times, res_v)
    plt.show()

    # x -> y
    initial_pos = np.matmul(np.sqrt(M), initial_pos)

    initial_vel = np.matmul(np.sqrt(M), initial_vel)

    # second type of encoding

    A_pseudoinv = np.linalg.pinv(A)
    B_pseudoinv = np.linalg.pinv(B)
    P = np.eye(dim) - np.matmul(A_pseudoinv, A)


    times = np.linspace(0, 10, 100)
    res1 = []
    res2 = []
    vel1 = []
    vel2 = []
    for i in times:
        [qc, norm_two] = evolve2(i)
        result_two = np.array(Statevector(qc))
        l = int(len(result_two) / 2)

        result_two = result_two * norm_two

        vel = result_two[l:]
        pos = result_two[:l]

        pos = np.matmul(np.linalg.inv(P), pos)

        vel = vel * (1j)
        vel = np.matmul(np.linalg.inv(P), np.matmul(np.linalg.inv(B_pseudoinv), vel))

        pos = np.matmul(np.linalg.inv(np.sqrt(M)), pos)
        vel = np.matmul(np.linalg.inv(np.sqrt(M)), vel)

        qc.measure_all()
        simulator = Aer.get_backend('aer_simulator')
        circ = transpile(qc, simulator)
        result = simulator.run(circ).result()
        counts = result.get_counts(qc)
        res1.append(pos[0])
        res2.append(pos[1])
        vel1.append(vel[0])
        vel2.append(vel[1])

    plt.plot(times, res1)
    plt.plot(times, vel1)
    plt.show()
    plt.plot(times, res2)
    plt.plot(times, vel2)
    plt.show()

    create_animation(times, [res1, res2], "quantum_encoding_2.gif")


def example():
    print("Simulation of a system using the first encoding provided in the paper")
    simulate_encoding_one([1, 1], [[0, 0, 1], [0, 1, 1], [1, 1, 1]], [1, 2], [1, -1])
    print("Simulation of the same system using the encoding provided in the appendix")
    simulate_encoding_two([1, 1], [[0, 0, 1], [0, 1, 1], [1, 1, 1]], [1, 2], [1, -1])
