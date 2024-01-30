import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_algorithms import TimeEvolutionProblem, TrotterQRTE


# mass matrix
def generate_m(masses, dim):
    m = [[0 for _ in range(dim)] for _ in range(dim)]
    for i in range(len(masses)):
        m[i][i] = masses[i]
    return m


# spring constant matrix
def generate_k(spring_constants, dim):
    k = [[0 for _ in range(dim)] for _ in range(dim)]
    for i in range(len(spring_constants)):
        curr = spring_constants[i]
        idx_1 = curr[0]
        idx_2 = curr[1]
        k[idx_1][idx_2] = curr[2]
        k[idx_2][idx_1] = curr[2]
    return k


def generate_f(K, dim):
    F = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i != j:
                F[i][j] = -K[i][j]
            else:
                for l in range(dim):
                    F[i][i]+=K[i][l]
    return F


# 2-nd order ODE to 1-st order ODE decomposition of matrix A
def generate_b(A, dim):
    B = np.zeros((2*dim, 2*dim))
    for i in range(dim):
        for j in range(dim):
            if i != j:
                B[2*i : 2*i+2, 2*j : 2*j+2] = [[0, 0], [A[i][j], 0]]
            else:
                B[2*i : 2*i+2, 2*j : 2*j+2] = [[0, 1], [A[i][j], 0]]
    return B


# See https://quantumcomputing.stackexchange.com/questions/11899/how-can-i-decompose-a-matrix-in-terms-of-pauli-matrices
def get_coefficient(matrix, gate_list, dimension):
    gate = Pauli(gate_list[0])
    for i in range(1, len(gate_list)):
        gate = gate.tensor(Pauli(gate_list[i]))
    trace = gate.to_matrix().dot(matrix).trace()
    return trace / 2 ** dimension


def construct_sparse_pauli_op(matrix, dimension):
    operators = list(product('XYZI', repeat=dimension))
    result = []
    for operator in operators:
        operator = ''.join(operator)
        coefficient = get_coefficient(matrix, operator, dimension)
        if coefficient != 0:
            result.append((operator, coefficient))
    return SparsePauliOp.from_list(result)


def hamiltonian_evolution(hamiltonian, qubit_count, time=1, initial_state=None, num_timesteps=1):
    pauli_hamiltonian = construct_sparse_pauli_op(hamiltonian, qubit_count)

    # Hamiltonian evolution library from Qiskit
    if initial_state is None:
        initial_state = QuantumCircuit(qubit_count)
    evolution_problem = TimeEvolutionProblem(pauli_hamiltonian, time, initial_state)
    estimator = Estimator()
    trotter = TrotterQRTE(estimator=estimator, num_timesteps=num_timesteps)
    return trotter.evolve(evolution_problem).evolved_state


def next_power_of_2(x):
    return math.ceil(math.log2(x))


def generate_h(m, k):
    h = [[0 for _ in range(len(m))] for _ in range(len(m))]
    for i in range(len(m)):
        if m[i][i] == 0:
            continue
        for j in range(len(m)):
            mass = m[i][i]
            spring_constant = k[i][j]
            sign = 1 if spring_constant > 0 else -1
            spring_constant = abs(spring_constant)
            h[i][j] = sign * math.sqrt(spring_constant / mass)
    return h


def block_encode(matrix):
    length = len(matrix)
    result = [[0 for _ in range(length * 2)] for _ in range(length * 2)]
    for i in range(length):
        for j in range(length):
            result[i][j+length] = matrix[i][j]
            result[i+length][j] = matrix[j][i]
    return result


def construct_hamiltonian(masses, spring_constants):
    # masses is a list of numbers
    # spring_constants is 2D list, where each nested list is as follows:
    # [idx_1, idx_2, spring_constant] to create a spring with constant spring_constant between idx_1 and idx_2
    # returns [hamiltonian, qubits], where qubits is the number of qubits needed to represent the hamiltonian
    qubits = next_power_of_2(len(masses))
    dim = 2**qubits
    m = generate_m(masses, dim)
    k = generate_k(spring_constants, dim)
    h = generate_h(m, k)
    hamiltonian = block_encode(h)
    qubits = qubits + 1  # One more qubit needed because hamiltonian was doubled during block encoding
    return [hamiltonian, qubits]