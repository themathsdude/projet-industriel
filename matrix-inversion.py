import numpy as np
import matplotlib.pyplot as plt


def interpolation(A, B, penteA, penteB):
    vandermonde = np.array([[A[0] ** 3, A[0] ** 2, A[0], 1],
                            [B[0] ** 3, B[0] ** 2, B[0], 1],
                            [3 * A[0] ** 2, 2 * A[0], 1, 0],
                            [3 * B[0] ** 2, 2 * B[0], 1, 0]])
    sndmembre = np.array([A[1], B[1], penteA, penteB])
    sol = np.linalg.solve(vandermonde, sndmembre)
    return sol, vandermonde


diametreTube = 1e-1
rayonTube = diametreTube / 2
alpha = np.pi/4
longueurTubeY = 1
longueurTube1 = 1
d0 = diametreTube / np.sin(alpha) - rayonTube / np.tan(alpha)
hauteurOscillations = 2 * rayonTube
nbOscillations = 5
longueurOscillations = 1
k = 2 * np.pi * nbOscillations / longueurOscillations
pente = hauteurOscillations * k

A = np.array((longueurTubeY + d0 + longueurTube1, rayonTube))
B = np.array((longueurTubeY + d0 + longueurTube1 + rayonTube, 2*rayonTube))
penteA = 0
penteB = pente*(B[1]-A[1])/(B[0]-A[0])

C = np.array((0, 0))
D = np.array((1, 1))

coeffs, vand = interpolation(C, D, penteA, penteB)

x = np.linspace(C[0], D[0], 100)


def f(x, coeffs):
    y = np.empty_like(x)
    for k, xval in enumerate(x):
        xarr = np.array((xval ** 3, xval ** 2, xval, 1))
        y[k] = np.dot(xarr, coeffs)
    return y


y = f(x, coeffs)

plt. figure()
plt.plot(x, y)
# plt.plot(x, penteA*(x - A[0]) + A[1])
# plt.plot(x, penteB*(x - B[0]) + B[1])
plt.show()
