from classical_simulation import example as classical_example
from classical_quantum_simulation import example as classical_quantum_example
from hamiltonian_circuit import example as quantum_example
from encoding_comparison import example as encoding_example
from error_simulation import compare as error_example

print("Simulation using equations from classical mechanics")
print("--------------------------------------------------")
classical_example()
print("--------------------------------------------------")
print("Simulating the same system with classical computations of the quantum algorithm")
print("--------------------------------------------------")
classical_quantum_example()
print("--------------------------------------------------")
print("Simulating the same system with (simulated) quantum computations of the quantum algorithm")
print("--------------------------------------------------")
quantum_example()
print("--------------------------------------------------")
print("Visualization of the error between the classical and quantum simulation")
print("--------------------------------------------------")
error_example()
print("--------------------------------------------------")
print("Comparisons of different encodings")
encoding_example()
print("--------------------------------------------------")
