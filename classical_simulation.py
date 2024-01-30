#  simulation of 2 masses, each coupled to each other and to a wall
#  | --- [] --- [] --- |
#  see https://ocw.mit.edu/courses/res-8-009-introduction-to-oscillations-and-waves-summer-2017/mitres_8_009su17_lec6.pdf

import math
import matplotlib.pyplot as plt
import numpy as np

from visualization_utils import create_animation


def simulate(k, m, omega_0, A_plus, A_minus, phi_plus, phi_minus):
    def x_1(t):
        return (A_plus * math.cos(omega_0 * t - phi_plus) +
                A_minus * math.cos(math.sqrt(3) * omega_0 * t - phi_minus))

    def x_2(t):
        return (A_plus * math.cos(omega_0 * t - phi_plus) -
                A_minus * math.cos(math.sqrt(3) * omega_0 * t - phi_minus))

    def plot_function(func):
        t_vals = np.linspace(0, 10, 1000)
        x_vals = [func(t) for t in t_vals]
        plt.plot(t_vals, x_vals)
        plt.xlabel("t")
        plt.ylabel(func.__name__ + "(t)")
        plt.title("Classical simulation of " + func.__name__ + "(t)")
        plt.show()

    plot_function(x_1)
    plot_function(x_2)

    #  animation showing how the two masses move
    t_vals = np.linspace(0, 10, 200)
    x_1_vals = [x_1(t) for t in t_vals]
    x_2_vals = [x_2(t) for t in t_vals]
    create_animation(t_vals, [x_1_vals, x_2_vals], "classical.gif")


def example():
    print("Simulation of 2 masses, each coupled to each other and to a wall")
    print("This simulation uses equations from classical mechanics")
    k = 1  # spring constant of the springs
    m = 1  # mass of the masses
    omega_0 = math.sqrt(k / m)
    A_plus = 2  # arbitrary constant
    A_minus = 2  # arbitrary constant
    phi_plus = 0  # arbitrary constant
    phi_minus = 0  # arbitrary constant

    simulate(k, m, omega_0, A_plus, A_minus, phi_plus, phi_minus)
