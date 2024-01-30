from matplotlib import pyplot as plt
from qiskit import Aer, transpile
from qiskit.quantum_info import Statevector

from matrix_utils import next_power_of_2, generate_m, generate_k, generate_f, hamiltonian_evolution
import numpy as np

from visualization_utils import create_animation


def construct_hamiltonian(masses, spring_constants, B, B_dagger):
    qubits = next_power_of_2(len(masses))
    dim = 2**qubits

    h = -1*np.block([[np.zeros((dim,dim)),B],[B_dagger,np.zeros((dim,dim))]])
    qubits = qubits + 1  # One more qubit needed because hamiltonian was doubled during block encoding
    return [h, qubits]


def encoding_one(masses, spring_constants, initial_pos, initial_vel):
    qubits = next_power_of_2(len(masses))
    dim = 2**qubits
    times = np.linspace(0, 10, 100)

    M = generate_m(masses, dim)
    K = generate_k(spring_constants, dim)
    F = generate_f(K, dim)

    A = np.matmul(np.linalg.inv(np.sqrt(M)), np.matmul(F, np.linalg.inv(np.sqrt(M))))
    B = np.linalg.cholesky(A)
    B_dagger = np.conjugate(B).T

    A_pseudoinv = np.linalg.pinv(A)
    B_pseudoinv = np.linalg.pinv(B)
    P = np.eye(dim) - np.matmul(A_pseudoinv,A)

    initial_pos = np.matmul(np.sqrt(M), initial_pos)
    initial_vel = np.matmul(np.sqrt(M), initial_vel)

    def evolve_encoding_one(time):
        first_comp = initial_vel

        second_comp = 1j * np.matmul(B_dagger, initial_pos)

        initial_state = np.concatenate((first_comp,second_comp))

        norm = np.linalg.norm(initial_state)

        normalized_initial_state = initial_state / np.linalg.norm(initial_state)
        initial_vec = Statevector(normalized_initial_state)

        [hamiltonian,qubits] = construct_hamiltonian(masses,spring_constants, B, B_dagger)
        evolution_one = hamiltonian_evolution(hamiltonian, qubits, initial_state=initial_vec, time=time, num_timesteps=50)
        return [evolution_one, norm]

    # plot results

    counts = []

    position1_first = []
    position2_first = []
    velocity1_first = []
    velocity2_first = []
    for i in times:
        [qc, norm] = evolve_encoding_one(i)

        result_one = np.array(Statevector(qc))

        l = int(len(result_one)/2)

        result_one = result_one*norm
        vel = result_one[:l]
        pos = result_one[l:]

        # reverting the encoding to normal positions/velocities
        pos = pos * (-1j)
        pos = np.matmul(np.linalg.inv(B_dagger), pos)

        pos = np.matmul(np.linalg.inv(np.sqrt(M)),pos)
        vel = np.matmul(np.linalg.inv(np.sqrt(M)),vel)
        qc.measure_all()
        simulator = Aer.get_backend('aer_simulator')
        circ = transpile(qc, simulator)
        result = simulator.run(circ).result()
        counts = result.get_counts(qc)
        position1_first.append(pos[0])
        position2_first.append(pos[1])

        velocity1_first.append(vel[0])
        velocity2_first.append(vel[1])

    plt.plot(times, position1_first)
    plt.plot(times, position2_first)
    plt.show()

    create_animation(times, [position1_first, position2_first], "quantum_encoding_1.gif")

    return position1_first, position2_first, velocity1_first, velocity2_first, times

def encoding_two(masses, spring_constants, initial_pos, initial_vel):
    qubits = next_power_of_2(len(masses))
    dim = 2 ** qubits
    times = np.linspace(0, 10, 100)

    M = generate_m(masses, dim)
    K = generate_k(spring_constants, dim)
    F = generate_f(K, dim)

    A = np.matmul(np.linalg.inv(np.sqrt(M)), np.matmul(F, np.linalg.inv(np.sqrt(M))))
    B = np.linalg.cholesky(A)
    B_dagger = np.conjugate(B).T

    A_pseudoinv = np.linalg.pinv(A)
    B_pseudoinv = np.linalg.pinv(B)
    P = np.eye(dim) - np.matmul(A_pseudoinv, A)

    initial_pos = np.matmul(np.sqrt(M), initial_pos)
    initial_vel = np.matmul(np.sqrt(M), initial_vel)

    def evolve_encoding_two(time):
        first_comp = -np.matmul(B, np.matmul(B_dagger, initial_pos))

        second_comp = 1j * np.matmul(B_dagger, initial_vel)

        initial_state = np.concatenate((first_comp, second_comp))

        norm_two = np.linalg.norm(initial_state)

        initial_vec = Statevector(initial_state / np.linalg.norm(initial_state))

        [hamiltonian, qubits] = construct_hamiltonian(masses, spring_constants, B, B_dagger)
        evolution_two = hamiltonian_evolution(hamiltonian, qubits, initial_state=initial_vec, time=time, num_timesteps=50)

        return [evolution_two, norm_two]

    # plot results

    counts = []

    position1_second = []
    position2_second = []
    velocity1_second = []
    velocity2_second = []

    check = np.matmul(np.matmul(B, B_pseudoinv), P)

    for i in times:
        [qc, norm_two] = evolve_encoding_two(i)
        result_two = np.array(Statevector(qc))
        l = int(len(result_two) / 2)

        result_two = result_two * norm_two

        vel = result_two[l:]
        pos = result_two[:l]

        pos = -np.matmul(np.linalg.inv(B_dagger), np.matmul(np.linalg.inv(B), pos))

        vel = -1j * np.matmul(np.linalg.inv(B_dagger), vel)

        pos = np.matmul(np.linalg.inv(np.sqrt(M)), pos)
        vel = np.matmul(np.linalg.inv(np.sqrt(M)), vel)

        qc.measure_all()
        simulator = Aer.get_backend('aer_simulator')
        circ = transpile(qc, simulator)
        result = simulator.run(circ).result()
        counts = result.get_counts(qc)
        position1_second.append(pos[0])
        position2_second.append(pos[1])
        velocity1_second.append(vel[0])
        velocity2_second.append(vel[1])

    plt.plot(times, position1_second)
    plt.plot(times, position2_second)
    plt.show()

    create_animation(times, [position1_second, position2_second], "quantum_encoding_2.gif")

    return position1_second, position2_second, velocity1_second, velocity2_second, times


def get_difference(
        position1_first, position2_first, velocity1_first, velocity2_first,
        position1_second, position2_second, velocity1_second, velocity2_second,
        times
):
    pos_diff = []
    vel_diff = []
    for i in range(len(position1_first)):
        pos_diff.append(np.abs(position1_first[i]-position1_second[i]) + np.abs(position2_first[i]-position2_second[i]))
        vel_diff.append(np.abs(velocity1_first[i]-velocity1_second[i]) + np.abs(velocity2_first[i]-velocity2_second[i]))

    plt.plot(times, pos_diff)
    plt.plot(times, vel_diff)
    plt.xlabel('Time')
    plt.ylabel('Difference between first and second encoding')
    plt.legend(['position differences', 'velocity differences'])
    plt.show()


def example():
    masses = [1, 1]
    spring_constants = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
    initial_pos = [1, 2]
    initial_vel = [1, -1]

    print("Simulating 2 masses, each coupled to each other and to a wall using the first encoding from the paper")
    position1_first, position2_first, velocity1_first, velocity2_first, times = (
        encoding_one(masses, spring_constants, initial_pos, initial_vel))
    print("Simulating the same system with the second encoding from the paper")
    position1_second, position2_second, velocity1_second, velocity2_second, times = (
        encoding_two(masses, spring_constants, initial_pos, initial_vel))
    get_difference(
        position1_first, position2_first, velocity1_first, velocity2_first,
        position1_second, position2_second, velocity1_second, velocity2_second,
        times
    )
