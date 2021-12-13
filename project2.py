import numpy as np
from numpy import diag, ones, eye, dot, zeros
from numpy.linalg import inv, norm
import scipy.sparse.linalg
import matplotlib.pyplot as plt

# Two point Dirichlet boundary value problem
def twopBVP(fvec, alpha, beta, L, N):
    dx = L/(N + 1)
    fvec[0] += -alpha/(dx**2)
    fvec[-1] += -beta/(dx**2)

    Tdx = scipy.sparse.csr_matrix(1/(dx**2)*(diag(ones(N - 1), 1) + diag(ones(N - 1), -1) - 2*eye(N)))
    y = scipy.sparse.linalg.spsolve(Tdx, fvec)
    approx = np.array([alpha, *y, beta])

    return approx

# Dirichlet eigenvalue problem
def Schrödinger(V, M, xgrid):
    L = xgrid[-1] - xgrid[0]
    N = len(xgrid) - 2
    ingrid = xgrid[1:-1]
    Vize = np.vectorize(V)

    dx = L/(N + 1)
    Tdx = 1/(dx**2)*(diag(ones(N - 1), 1) + diag(ones(N - 1), -1) - 2*eye(N))
    T = Tdx - 2*dot(Vize(ingrid), np.eye(N))
    v, w = np.linalg.eig(T)   # v is array of eigenvalues, w is matrix of eigenvectors
    z = np.zeros((1, M))
    # Sorting eigenv. by ascending absolute value of eigenvalues
    idx = np.argsort(np.abs(v)) 
    v = v[idx]
    w = w[:,idx]

    # Returns information with added boundary condition u(0) = 0 for all eigenvectors
    return v[:M], np.vstack((z, w[:,:M], z))


# Mixed Dirichlet-Neumann eigenvalue problem
def DNEVP(N, M):
    dx = 1/N    
    Tdx = 1/(dx**2)*(diag(ones(N - 1), 1) + diag(ones(N - 1), -1) - 2*eye(N))
    Tdx[-1][-2] = 2/(dx**2)   
    v, w = np.linalg.eig(Tdx)   # v is array of eigenvalues, w is matrix of eigenvectors

    # Sorting eigenv. by ascending absolute value of eigenvalues
    idx = np.argsort(np.abs(v)) 
    v = v[idx]
    w = w[:,idx]
    
    # Returns information with added boundary condition u(0) = 0 for all eigenvectors
    return v[:M], np.vstack((np.zeros((1, M)), w[:,:M]))

def RMS(v, dx):
    return np.sqrt(dx)*norm(v)

def errVSdx(f, g, L):
    alpha = g(0)
    beta = g(L)
    
    fize = np.vectorize(f)
    gize = np.vectorize(g)

    D = []
    E = []
        
    for k in range(1, 13):
        print(f'Simulating for N = 2^{k}')
        N = 2**k
        dx = L/(N + 1)
        D.append(dx)

        xgrid = np.linspace(0, L, N + 2)
        ingrid = xgrid[1:-1]

        approx = twopBVP(fize(ingrid), alpha, beta, L, N)[1:-1]
        sol = gize(ingrid)

        err = RMS(sol - approx, dx)
        E.append(err)

    plt.figure()
    plt.loglog(D, E, 'r')
    plt.loglog(D, [dx**2 for dx in D], 'k--')
    plt.title('Error vs dx, with error computed as RMS of global error at every point')
    plt.legend(['Error', 'log(dx) = log(dx^2) = 2log(dx)'])
    plt.xlabel('dx')
    plt.ylabel('Error')
    plt.grid()

def errVSN():
    Ns = []
    E = []

    truel = lambda k: -(2*k - 1)**2*np.pi**2/4


    M = 10

    for k in range(2, 10):
        N = 2**k
        Ns.append(N)
        print(f'Simulating for N = {N}')
        
        l = np.array([truel(i) for i in range(1, 4)])
        ldx, _ = DNEVP(N, 3)
        e = abs(ldx - l)
        E.append(e.tolist())

    E = np.array(E)

    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    for i, ax in enumerate([ax1, ax2, ax3]):
        e = E[:,i]
        ax.loglog(Ns, e, 'r--')
        ax.grid()

    ax1.set(title='Eigenvalue approximation errors for first three eigenvalues \n as functions of N')
    ax1.set(ylabel=r'|$\lambda_{\Delta x, 1} - \lambda_1$|')
    ax2.set(ylabel=r'|$\lambda_{\Delta x, 2} - \lambda_2$|')
    ax3.set(ylabel=r'|$\lambda_{\Delta x, 3} - \lambda_3$|')
    ax3.set(xlabel='N')

if __name__ == '__main__':
    def task11():
        L = 5
        N = 100

        xgrid = np.linspace(0, L, N + 2)
        ingrid = xgrid[1:-1]
        
        g = lambda x: np.exp(-x**2)
        # g = lambda x: 4*x**4 - 3*x**3 + 2*x**2 - x
        alpha = g(0)
        beta = g(L)
        f = lambda x: -2*np.exp(-x**2)*(1 - 2*x**2)
        # f = lambda x, y: 48*x**2 - 18*x + 4

        # gize = np.vectorize(g)
        fize = np.vectorize(f)
        
        fvec = fize(ingrid)
        # sol = g(ingrid)

        y = twopBVP(fvec, alpha, beta, L, N)
        plt.figure()
        plt.plot(xgrid, y, 'r--')
        plt.plot(xgrid, g(xgrid), 'k')
        plt.title(f'Numerical and analytic solution of y\"(x) = (e^(-x^2))\"')
        plt.legend(['Numerical solution with N = {N}', 'Analytic solution'])
        plt.xlabel('x')
        plt.ylabel('y')

        errVSdx(f, g, L)

        plt.show()

    def task12():
        q = -50000
        E = 1.9e11
        L = 10
        N = 999

        fM = lambda x: q

        xgrid = np.linspace(0, L, N + 2)
        ingrid = xgrid[1:-1]
        
        fMize = np.vectorize(fM)
        fMvec = fMize(ingrid)

        alpha = beta = 0
        M = twopBVP(fMvec, alpha, beta, L, N)
        I = 1e-3*(3 - 2*(np.cos(np.pi*ingrid/L)**12))

        fuvec = M[1:-1]/(E*I)

        u = twopBVP(fuvec, alpha, beta, L, N)

        plt.figure()
        plt.plot(xgrid, 1000*u)
        plt.title('Deflection of bridge')
        plt.xlabel('x (m)')
        plt.ylabel('Deflection (mm)')
        plt.show()

        print(f'u({xgrid[500]}) = {u[500]}')
        # Midpoint deflection: -11.741059085875718 mm with N = 999

    def task21():
        N = 499
        M = 3

        xgrid = np.linspace(0, 1, N + 1)
        d, S = DNEVP(N, M)
        print(d)

        errVSN()

        plt.figure()
        for i in range(M):
            l = d[i]
            v = S[:,i]
            plt.plot(xgrid, v, '-')

        plt.title(
            f'Approximations of first {M} eigenfunctions sorted by ascending absolute eigenvalue, with N = {N}',
            fontsize=20.0    
        )        
        plt.legend([f'k = {k}' for k in range(1, M + 1)], fontsize=15.0)
        plt.xlabel('x', fontsize=20.0)
        plt.ylabel(r'$\phi_k$', fontsize=20.0)

        plt.show()

    def task22():
        N = 999
        xgrid = np.linspace(0, 1, N + 2)
        # V = lambda x: 0
        # V = lambda x: 700*(0.5 - abs(x  - 0.5))
        # V = lambda x: 800*np.sin(np.pi*x)**2
        k = 6
        # V = lambda x: 1000*np.sin(np.pi*k*x)**2
        eps = 0.1
        V = lambda x: N**2*(abs(x - 0.25) <= eps)

        Vize = np.vectorize(V)
        M = 6

        d, S = Schrödinger(V, M, xgrid)
        print(d)
        P = 1/N

        _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)        
        ax1.plot(xgrid, Vize(xgrid), 'k')
        
        for i in range(M):
            E = -d[i]
            psi = S[:,i]
            psi_hat = 1/P*psi
            p_hat = np.power(np.abs(psi_hat), 2)

            ax2.plot(xgrid, psi_hat + E)
            ax3.plot(xgrid, p_hat + E)

        ax1.set_title(
            f'First {M} normalized wave functions and probability densities \n sorted by ascending absolute energy level, with N = {N}',
            fontsize=24.0
        )
        ax1.set_ylabel(r'$V$', fontsize=20.0)
        ax2.set_ylabel(r'$\hat{\psi}_k + E_k$', fontsize=20.0)
        ax3.set_ylabel(r'$|\hat{\psi}_k^2| + E_k$', fontsize=20.0)        
        ax3.set_xlabel(r'$x$', fontsize=20.0)

        plt.show()

    # task11()
    # task12()
    # task21()
    task22()
