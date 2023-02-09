from scipy.integrate import solve_ivp
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import math as m


def Henon_Heiles(t, u):
       return (u[3], u[2], -1 / 12 * (
                m.e ** (2 * u[1] + 2 * np.sqrt(3) * u[0]) + m.e ** (2 * u[1] - 2 * np.sqrt(3) * u[0]) - 2 * m.e ** (
                    -4 * u[1])),
            -np.sqrt(3) / 12 * (m.e ** (2 * u[1] + 2 * np.sqrt(3) * u[0]) - m.e ** (2 * u[1] - 2 * np.sqrt(3) * u[0])))


def PSS(x, u):
      return (1, u[2] / u[3], -1 / 12 * (
                m.e ** (2 * u[1] + 2 * np.sqrt(3) * u[0]) + m.e ** (2 * u[1] - 2 * np.sqrt(3) * u[0]) - 2 * m.e ** (
                    -4 * u[1])) / u[3],
            -np.sqrt(3) / 12 * (m.e ** (2 * u[1] + 2 * np.sqrt(3) * u[0]) - m.e ** (2 * u[1] - 2 * np.sqrt(3) * u[0])) /
            u[3], 1 / u[3])


figure, axes = plt.subplots(figsize=(6, 12))
for j in np.linspace(-2, 2, 10):
    for k in np.linspace(-23, 23, 75):
        if 2 * (10 + 1 / 8 - 1 / 24 * (
                m.e ** (2 * (j) + 2 * np.sqrt(3) * (0)) + m.e ** (2 * (j) - 2 * np.sqrt(3) * (0)) + m.e ** (
                -4 * (j)))) - (k ** 2) >= 0:
            u0 = (0, j, k, m.sqrt(2 * (10 + 1 / 8 - 1 / 24 * (
                        m.e ** (2 * (j) + 2 * np.sqrt(3) * (0)) + m.e ** (2 * (j) - 2 * np.sqrt(3) * (0)) + m.e ** (
                            -4 * (j)))) - (k ** 2)))
            sol = solve_ivp(Henon_Heiles, [0, 2000], u0, method='DOP853', rtol=1e-6, atol=1e-6)

            plot_y = []

            plot_py = []
            print(j, k)
            for i in np.arange(0, len(sol.y[0]) - 1):
                if sol.y[3][i] >= 0:
                    ut = (sol.y[0][i], sol.y[1][i], sol.y[2][i], sol.y[3][i], sol.t[i])
                    temp = solve_ivp(PSS, [sol.y[0][i], 0], ut, method='DOP853', rtol=1e-6, atol=1e-6)
                    plot_y.append(temp.y[1][-1])
                    plot_py.append(temp.y[2][-1])

            plt.scatter(plot_y, plot_py, s=0.005)

plt.title("Q1 PSS for H = 10")
plt.xlabel('y', fontsize=16)
plt.ylabel('py', fontsize=16)

plt.savefig('AMM1_Assignment2_1_figure10')