import concurrent

import numpy as np
import matplotlib.pyplot as plt
from hamiltonian_circuit import simulate
import multiprocessing.dummy as mp
from multiprocessing import cpu_count


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def compare():
    trott = [1, 5, 10, 15, 20, 30]

    rmse1 = np.zeros(len(trott))
    rmse2 = np.zeros(len(trott))

    times = np.linspace(0, 10, 100)

    def simulate_wrapper(element):
        quant_pos, _ = simulate(2, element, True, True)  # run full simulation
        quant1 = [item[0] for item in quant_pos]  # separate solution for each mass
        quant2 = [item[1] for item in quant_pos]
        class_pos, _ = simulate(2, element, False, True)
        class1 = [item[0] for item in class_pos]
        class2 = [item[1] for item in class_pos]

        plt.plot(times, quant1)
        plt.plot(times, quant2)
        plt.plot(times, class1)
        plt.plot(times, class2)
        plt.legend(['Quantum mass 1', 'Quantum mass 2', 'Classical mass 1', 'Classical mass 2'])
        plt.title(f"Simulation for {element} Trotter steps")
        plt.show()

        plt.figure()

        class1 = np.array(class1)
        class2 = np.array(class2)
        quant1 = np.array(quant1)
        quant2 = np.array(quant2)
        index = trott.index(element)
        rmse1[index] = rmse(class1, quant1)
        rmse2[index] = rmse(class2, quant2)

    p = mp.Pool(cpu_count() - 1)
    p.map(simulate_wrapper, trott)
    p.close()
    p.join()

    plt.plot(trott, rmse1, label='Mass 1', marker='o')
    plt.plot(trott, rmse2, label='Mass 2', marker='o')
    plt.grid()
    plt.legend()
    plt.title('RMSE classical and quantum simulation for $ x_o = (0, 0)$ and $v_o = (1, -1)$')
    plt.xlabel('Trotter steps')
    plt.ylabel('RMSE displacement')
    plt.show()
