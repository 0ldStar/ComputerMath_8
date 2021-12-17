import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.optimize import linprog

A, B = -10, 15
GLOBAL_A, GLOBAL_B = -100, 100
GLOBAL_PIECE_COUNT = 4
STEP = 0.001
EPS = 1e-3
SOL1 = -5


def func(x):
    return (x + 5) ** 4


def der_func(x):
    return 4 * (x + 5) ** 3


def f(params):
    # print(params)  # <-- you'll see that params is a NumPy array
    x1, x2 = params  # <-- for readability you may wish to assign names to the component variables
    return 4 * x1 ** 2 + 4 * x1 + x2 ** 2 - 8 * x2 + 5


# linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])


def chord(start, end, eps):
    x = start
    a = start
    b = end
    n = 0
    x_arr = []
    if der_func(a) * der_func(b) > 0:
        return None, None, None
    while abs(der_func(x)) > eps:
        x = b - der_func(b) * (b - a) / (der_func(b) - der_func(a))
        x_arr.append(abs(x - SOL1))
        if der_func(a) * der_func(x) < 0:
            b = x
        else:
            if der_func(x) * der_func(b) < 0:
                a = x
        n += 1
    return x, n, x_arr


def simple_chord():
    X = np.arange(A, B, STEP)
    x1, n, n_arr = chord(A, B, EPS)
    plt.plot(X, func(X), label='function graph')
    print(x1, n)
    plt.scatter(x1, func(x1), label='minimize by chord method')
    plt.scatter(SOL1, func(SOL1), label='true minimum')
    plt.grid()
    plt.legend()
    plt.show()
    N = np.arange(0, n, 1)
    plt.plot(N, n_arr, label='error on ones iteration')
    plt.xlabel("iteration")
    plt.ylabel("abs error")
    plt.grid()
    plt.legend()
    plt.show()


def global_chord():
    sol = []
    sol_y = []
    h = (GLOBAL_B - GLOBAL_A) / GLOBAL_PIECE_COUNT
    for i in range(GLOBAL_PIECE_COUNT):
        x1, n, n_arr = chord(GLOBAL_A + i * h, GLOBAL_A + (i + 1) * h, EPS)
        if x1 is not None:
            sol.append(x1)
            sol_y.append(func(x1))
    X = np.arange(GLOBAL_A, GLOBAL_B, STEP)
    plt.plot(X, func(X), label='function graph')
    plt.plot(sol, sol_y, label='minimize by chord method')
    plt.scatter(SOL1, func(SOL1), label='true minimum')
    plt.grid()
    plt.legend()
    plt.show()


def two_sol():
    initial_guess = [1, 1]
    result = optimize.minimize(f, initial_guess)
    if result.success:
        fitted_params = result.x
        print(fitted_params)
    else:
        raise ValueError(result.message)


# def shraf(eps):
#     i = 1
#
#     while i < 1000:
#         if curr_func(x_c) < eps:
#             break
#         curr_func = lambda x: rz(x) + r * (1.0 / (h_1(x) ** 2 + h_2(x) ** 2 + h_3(x) ** 2))
#         x_c = minimize(curr_func, x_c).x;
#         i += 1
#         r *= b;
#
global_chord()
two_sol()
