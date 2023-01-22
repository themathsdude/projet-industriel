import numpy as np
import matplotlib.pyplot as plt


def interpolation(A, B, penteA, penteB):
    vandermonde = np.array([[A[0] ** 3, A[0] ** 2, A[0], 1],
                            [B[0] ** 3, B[0] ** 2, B[0], 1],
                            [3 * A[0] ** 2, 2 * A[0], 1, 0],
                            [3 * B[0] ** 2, 2 * B[0], 1, 0]])
    sndmembre = np.array([A[1], B[1], penteA, penteB])
    sol = np.linalg.solve(vandermonde, sndmembre)
    return sol


A = np.array((0, 0))
B = np.array((1, 1))
penteA = 0
penteB = 1

coeffs = interpolation(A, B, penteA, penteB)

x = np.linspace(A[0], B[0], 100)


def f(x, coeffs):
    y = np.empty_like(x)
    for k, xval in enumerate(x):
        xarr = np.array((xval ** 3, xval ** 2, xval, 1))
        y[k] = np.dot(xarr, coeffs)
    return y


y = f(x, coeffs)

plt. figure()
plt.plot(x, y)
plt.plot(x, penteA*(x - A[0]) + A[1])
plt.plot(x, penteB*(x - B[0]) + B[1])
plt.show()
