import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

def solution(A, y0, tgrid):
    return np.array([np.dot(expm(A*t), y0) for t in tgrid])


def eulerstep(A, uold, h):
    return uold + h*np.dot(A, uold)


def ieulerstep(A, uold, h):
    N = len(A)
    I = np.eye(N)

    return np.dot(np.linalg.inv(I - h*A), uold)


def TRstep(A, uold, h):
    N = len(A)
    I = np.eye(N)

    return np.dot(np.dot(np.linalg.inv(I - h/2*A), (I + h/2*A)), uold)


def eulerint(A, y0, t0, tf, N):
    tgrid = np.linspace(t0, tf, N)
    h = (tf - t0)/N
    approx = np.array([y0])
    yn = y0

    for _ in range(N - 1):
        yn = eulerstep(A, yn, h)
        approx = np.append(approx, yn[np.newaxis], axis=0)

    sol = solution(A, y0, tgrid)
    err = sol - approx

    return [tgrid, approx, err]


def ieulerint(A, y0, t0, tf, N):
    tgrid = np.linspace(t0, tf, N)
    h = (tf - t0)/N
    approx = np.array([y0])
    yn = y0

    for _ in range(N - 1):
        yn = ieulerstep(A, yn, h)
        approx = np.append(approx, yn[np.newaxis], axis=0)

    sol = solution(A, y0, tgrid)
    err = sol - approx

    return [tgrid, approx, err]


def TRint(A, y0, t0, tf, N):
    tgrid = np.linspace(t0, tf, N)
    h = (tf - t0)/N
    approx = np.array([y0])
    yn = y0

    for _ in range(N - 1):
        yn = TRstep(A, yn, h)
        approx = np.append(approx, yn[np.newaxis], axis=0)

    sol = solution(A, y0, tgrid)
    err = sol - approx

    return [tgrid, approx, err]


def errVSh(A, y0, t0, tf):
    H = []
    E = []

    for k in range(1, 14):
        print(f'Simulating error for N = 2^{k} = {2**k}')
        N = 2**k
        h = (tf - t0)/N
        _, _, err = eulerint(A, y0, t0, tf, N)

        H.append(h)
        E.append(np.linalg.norm(err[-1]))

    plt.loglog(H, E)
    plt.show()


def ierrVSh(A, y0, t0, tf):
    H = []
    E = []

    for k in range(1, 14):
        print(f'Simulating error for N = 2^{k} = {2**k}')
        N = 2**k
        h = (tf - t0)/N
        _, _, err = ieulerint(A, y0, t0, tf, N)

        H.append(h)
        E.append(np.linalg.norm(err[-1]))

    plt.loglog(H, E)
    plt.show()


def TRerrVSh(A, y0, t0, tf):
    H = []
    E = []

    for k in range(1, 14):
        print(f'Simulating error for N = 2^{k} = {2**k}')
        N = 2**k
        h = (tf - t0)/N
        _, _, err = TRint(A, y0, t0, tf, N)

        H.append(h)
        E.append(np.linalg.norm(err[-1]))

    plt.loglog(H, E)
    plt.show()


if __name__ == '__main__':
    l = 1
    A = np.array([[l]])
    y0 = np.array([1])
    # A = np.array([[-1, 10], [0, -3]])
    # A = np.array([[-1, 100], [0, -3]])
    # y0 = np.array([1, 1])

    # tg, app, err = ieulerint(A, y0, 0, 10, 1000)

    # tg, app, err = eeuler.int(A, y0, 0, 10, 1000)

    ierrVSh(A, y0, 0, 5)

    # plt.plot(tg, solution(A, y0, tg), 'k')
    # plt.plot(tg, app, 'r--')
    # plt.plot(tg, np.log(abs(err)))
    # plt.show()

    # print(err)
    # print(f'Approximation: {app}, error: {err}')

    # errVSh(A, y0, 0, 5)
    # ierrVSh(A, y0, 0, 5)