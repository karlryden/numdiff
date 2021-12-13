import numpy as np
import matplotlib.pyplot as plt
from numpy import diag, exp, ones, zeros, eye, dot, linspace, meshgrid, vectorize

def eulerstep(Tdx, uold, dt):
    return uold + dt*dot(Tdx, uold)

def eulerint(Tdx, u0, tf, M):
    dt = tf/M
    approx = zeros((M + 1, len(u0)))
    approx[0] = u0
    un = u0

    for i in range(1, M + 1):
        un = eulerstep(Tdx, un, dt)
        approx[i] = un

    return approx

def TRstep(Tdx, uold, dt):
    N = len(uold)
    A = eye(N) - dt/2*Tdx
    b = dot(eye(N) + dt/2*Tdx, uold)
    return np.linalg.solve(A, b)

def TRint(Tdx, u0, tf, M):
    dt = tf/M
    approx = zeros((M + 1, len(u0)))
    approx[0] = u0
    un = u0

    for i in range(1, M + 1):
        un = TRstep(Tdx, un, dt)
        approx[i] = un

    return approx

def LaxWen(u, amu):
    N = len(u)
    d1 = amu/2*(1 + amu)
    d2 = 1 - amu**2
    d3 = -amu/2*(1 - amu)
    Tamu = d1*diag(ones(N - 1), -1) + d3*diag(ones(N - 1), 1) + d2*eye(N)
    Tamu[0][-1] = d1
    Tamu[-1][0] = d3

    return dot(Tamu, u)

def LaxWenint(a, u0, tf, N, M):
    dt = tf/M
    dx = 1/N

    amu = a*dt/dx

    approx = zeros((M + 1, len(u0)))
    approx[0] = u0
    un = u0

    for i in range(1, M + 1):
        un = LaxWen(un, amu)
        approx[i] = un

    return approx

if __name__ == '__main__':
    def task11():
        # Method becomes stable when dt/dx^2 = (N + 1)/M >=~ 20 (???!!!) 
        N = 100
        M = 2050
        M_v = 2000
        tf = 0.1
        dx = 1/(N + 1)
        Tdx = 1/(dx**2)*(diag(ones(N + 1), 1) + diag(ones(N + 1), -1) - 2*eye(N + 2))
        xgrid = linspace(0, 1, N + 2)
        tgrid = linspace(0, tf, M + 1)
        tgrid_v = linspace(0, tf, M_v + 1)
        [T, X] = meshgrid(xgrid, tgrid)
        [T_v, X_v] = meshgrid(xgrid, tgrid_v)

        # Definitely not an eigenfunction
        g = vectorize(lambda x: x*(1 - x)*np.exp(-(x**2)))
        U = eulerint(Tdx, g(xgrid), tf, M)
        U_v = eulerint(Tdx, g(xgrid), tf, M_v)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(T, X, U)

        fig_v = plt.figure()
        ax_v = fig_v.add_subplot(projection='3d')
        ax_v.plot_surface(T_v, X_v, U_v)

        plt.show()

    def task12():
        N = 100
        M = 100
        tf = 0.1
        dx = 1/(N + 1)
        Tdx = 1/(dx**2)*(diag(ones(N + 1), 1) + diag(ones(N + 1), -1) - 2*eye(N + 2))
        xgrid = linspace(0, 1, N + 2)
        ingrid = xgrid[1:-1]
        tgrid = linspace(0, tf, M + 1)
        [T, X] = meshgrid(xgrid, tgrid)

        g = vectorize(lambda x: x*(1 - x)*np.exp(-(x**2)))
        U = TRint(Tdx, g(xgrid), tf, M)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(T, X, U)
        plt.show()

    def task21():
        N = 99
        M = 2499
        tf = 0.1
        a = M/N

        xgrid = linspace(0, 1, N + 1)
        tgrid = linspace(0, tf, M + 1)

        g = vectorize(lambda x: exp(-100*((x - 0.5)**2)))
        #g = vectorize(lambda x: 0.5 - abs(x - 0.5) if (abs(x - 0.5) <= 0.25) else 0.25) # -^-

        [T, X] = meshgrid(xgrid, tgrid)
        U = LaxWenint(a, g(xgrid), tf, N, M)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(T, X, U)
        ax.set(xlabel='x')
        ax.set(ylabel='t')
        # ax.zlabel('u')
        plt.show()

    def task31():
        N = 99
        M = 2499
        tf = 0.1
        a = 1
        d = 1

        xgrid = linspace(0, 1, N + 1)
        tgrid = linspace(0, tf, M + 1)

        g = vectorize(lambda x: exp(-100*((x - 0.5)**2)))
        #g = vectorize(lambda x: 0.5 - abs(x - 0.5) if (abs(x - 0.5) <= 0.25) else 0.25) # -^-

        [T, X] = meshgrid(xgrid, tgrid)
        U = LaxWenint(a, g(xgrid), tf, N, M)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_wireframe(T, X, U)
        ax.set(xlabel='x')
        ax.set(ylabel='t')
        # ax.zlabel('u')
        plt.show()

    # task11()
    task12()
    # task21()
