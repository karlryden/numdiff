import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

def solution(A, y0, tgrid):
    return np.array([np.dot(expm(A*(t - tgrid[0])), y0) for t in tgrid])

class ThetaMethod:
    def __init__(self, theta=0): 
        self.theta = theta

    def step(self, A, uold, h):
        if type(A) is np.ndarray:
            I = np.eye(len(A)) 
            # U is the transition matrix; y_(n+1) = U*y_n = (I-h0A)^(-1)*(I+h(1-0)A)*y_n, (0 is theta)
            U = np.dot(  np.linalg.inv(  I - h*self.theta*A  ), I + h*(1 - self.theta)*A  )

        else:
            U = (1 + h*(1 - self.theta)*A)/(1 - h*self.theta*A)

        return np.dot(U, uold)

    def int(self, A, y0, t0, tf, N):
        tgrid = np.linspace(t0, tf, N)
        h = (tf - t0)/N

        approx = np.zeros((N, len(A)))
        # print(approx)
        # print(y0)
        approx[0] = y0
        yn = y0

        for i, _ in enumerate(tgrid[1:]):
            yn = self.step(A, yn, h)
            approx[i+1] = yn

        sol = solution(A, y0, tgrid)
        err = approx - sol

        return [tgrid, approx, err]

    def errVSh(self, A, y0, t0, tf):
        H = []
        E = []

        for k in range(1, 13):
            print(f'Simulating error for N = 2^{k} = {2**k}')
            N = 2**k
            h = (tf - t0)/N
            _, _, err = self.int(A, y0, t0, tf, N)

            H.append(h)
            E.append(np.linalg.norm(err[-1]))

        plt.loglog(H, E, label=f'Global error vs stepsize, h={h}, theta={self.theta}')
        # plt.show()


if __name__ == '__main__':
    l = 1
    # A = np.array([[l]])
    # y0 = np.array([1])
    A = np.array([[-1, 10], [0, -3]])
    # A = np.array([[-1, 100], [0, -3]])
    y0 = np.array([1, 1])

    eeuler = ThetaMethod(0)
    ieuler = ThetaMethod(1)
    TR = ThetaMethod(0.5)

    tg, app, err = TR.int(A, y0, 0, 10, 100)
    sol = solution(A, y0, tg)

    # plt.plot(tg, sol, 'k')
    # plt.plot(tg, app, 'r--')
    # plt.show()

    for method in [eeuler, ieuler, TR]:
        method.errVSh(A, y0, 0, 5)
        
    plt.legend()
    plt.show()

    # eeuler.errVSh(A, y0, 0, 5)

    # plt.plot(tg, solution(A, y0, tg), 'k')    
    # plt.plot(tg, app, 'r--')
    # plt.plot(tg, np.log(abs(err)))
    # plt.show()

    # print(err)
    # print(f'Approximation: {app}, error: {err}')

    # errVSh(A, y0, 0, 5)
    # ierrVSh(A, y0, 0, 5)