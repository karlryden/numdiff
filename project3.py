import numpy as np
import matplotlib.pyplot as plt
from numpy import diag, exp, ones, zeros, eye, dot, linspace, meshgrid, vectorize
from numpy.linalg.linalg import _multi_dot_matrix_chain_order

def RMS(v, dx):
    return np.sqrt(dx)*np.linalg.norm(v)

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
    # Tamu is a Toeplitz matrix
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
        tf = 5
        dx = 1/N
        dt = tf/M
        mu = dt/dx

        a1 = 1
        a2 = -0.5
        a3 = 1/mu
        a4 = 0.9/mu

        xgrid = linspace(0, 1, N + 1)
        tgrid = linspace(0, tf, M + 1)

        g = vectorize(lambda x: exp(-100*((x - 0.5)**2)))
        L2 = lambda v: RMS(v, dx)

        [T, X] = meshgrid(xgrid, tgrid)
        U1 = LaxWenint(a1, g(xgrid), tf, N, M)
        U2 = LaxWenint(a2, g(xgrid), tf, N, M)
        U3 = LaxWenint(a3, g(xgrid), tf, N, M)
        U4 = LaxWenint(a4, g(xgrid), tf, N, M)
        
        print(T)
        print(X)
        L2U3 = np.apply_along_axis(L2, 1, U3)
        L2U4 = np.apply_along_axis(L2, 1, U4)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(projection='3d')
        ax1.plot_wireframe(T, X, U1)
        ax1.set(xlabel='x')
        ax1.set(ylabel='t')
        ax1.set(title=f'a = {a1}')

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(projection='3d')
        ax2.plot_wireframe(T, X, U2)
        ax2.set(xlabel='x')
        ax2.set(ylabel='t')
        ax2.set(title=f'a = {a2}')
        
        figL2 = plt.figure()
        plt.plot(tgrid, L2U3)
        plt.plot(tgrid, L2U4)
        plt.xlabel('t')
        plt.legend(['amu = 1', 'amu = 2'])
        plt.title('L2 norm of solutions')

        plt.show()
        
        # The norm decreases with time when amu < 1. This is an information leak caused by going below the CLF-limit. 
        # Not sure how to motivate further.

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
    # task12()
    # task21()
    task31()