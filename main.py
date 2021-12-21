import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import LinearConstraint

A, B = -10, 15
GLOBAL_A, GLOBAL_B = -100, 100
GLOBAL_PIECE_COUNT = 10
STEP = 0.001
EPS = 1e-3
SOL1 = -5

linear_constrain = LinearConstraint([2, -1], 6, 6)


def func(x):
    return (x + 5) ** 4


def der_func(x):
    return 4 * (x + 5) ** 3


def fine_func(params) -> float:
    x1, x2 = params
    return (2 * x1 - x2 - 6) ** 2


def f(params):
    x1, x2 = params
    return 4 * x1 ** 2 + 4 * x1 + x2 ** 2 - 8 * x2 + 5


def F(params, m):
    return f(params) + m * fine_func(params)


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


def fib(n):
    SQRT5 = np.sqrt(5)
    PHI = (SQRT5 + 1) / 2
    return int(PHI ** n / SQRT5 + 0.5)


def fibonacci(start, end, eps):
    n = 10
    for i in range(n):
        lam = start + (fib(n - i - 1) / fib(n - i + 1)) * (end - start)
        mu = start + (fib(n - i) / fib(n - i + 1)) * (end - start)
        if func(lam) < func(mu):
            end = mu
        else:
            start = lam
    x_min = (start + end) / 2
    return x_min


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
        x1 = fibonacci(GLOBAL_A + i * h, GLOBAL_A + (i + 1) * h, EPS)
        if x1 is not None:
            sol.append(x1)
            sol_y.append(func(x1))
    X = np.arange(GLOBAL_A, GLOBAL_B, STEP)
    plt.plot(X, func(X), label='function graph')
    plt.scatter(sol, sol_y, label='minimize by chord method', color='red')
    plt.scatter(SOL1, func(SOL1), label='true minimum')
    plt.grid()
    plt.legend()
    plt.show()


def twoD_sol(x1, m, b, eps):
    while m * fine_func(x1) > eps:
        m = b * m
        result = optimize.minimize(F, x1, m)
        x1 = result.x
        # print(m * fine_func(x1))
    print(x1)
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter3D(X, Y, f([X, Y]), c=f([X, Y ]))
    # ax.contour3D(X, Y, Z, 50, cmap='binary')
    # plt.title('3D func graph')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()


def two_sol():
    initial_guess = [1, 1]
    result = optimize.minimize(f, initial_guess, constraints=linear_constrain)
    if result.success:
        fitted_params = result.x
        print(fitted_params)
    else:
        raise ValueError(result.message)


simple_chord()
global_chord()
two_sol()
twoD_sol([1, 1], 0.001, 6, 1e-3)
twoD_sol([1, 1], 0.001, 6, 1e-5)
twoD_sol([1, 1], 0.001, 6, 1e-7)

