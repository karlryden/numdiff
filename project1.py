import time
import numpy as np
from numpy.linalg import norm, inv
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def solution(A, y0, tgrid):
    return np.array([np.dot(expm(A*(t - tgrid[0])), y0) for t in tgrid])

def RK4step(f, told, uold, h):
    A = np.array([[0, 0, 0, 0],
                  [1/2, 0, 0, 0], 
                  [0, 1/2, 0, 0],
                  [0, 0, 1, 0]])
    b = np.array([1/6, 1/3, 1/3, 1/6])
    c = np.sum(A, axis=1)

    Ydot = np.ndarray(4)

    for i in range(4):
        # Note: This only works if A is strictly lower triangluar, i.e for explicit RK
        Ydot[i] = f(told + h*c[i], uold + h*sum([A[i][j]*Ydot[j] for j in range(i)]))

    unew = uold + h*sum([b[j]*Ydot[j] for j in range(4)])

    return unew

def RK4int(f, y0, t0, tf, h):
    N = int(np.ceil((tf - t0) // h))  # + ((tf - t0) % h != 0)) 
    tgrid = np.ndarray(N)
    approx = np.zeros((N, len(y0)))
    
    tgrid[0] = t0
    approx[0] = y0
    
    tn = t0
    yn = y0
    
    for i in range(1, N - 1):
        yn = RK4step(f, tn, yn, h)
        approx[i] = yn
        tn = tn + h
        tgrid[i] = tn

    tgrid[-1] = tf
    approx[-1] = RK4step(f, tn, yn, tf - tn)

    return [tgrid, approx]

def RK34step(f, told, uold, h):
    A4 = np.array([[0, 0, 0, 0],
                  [1/2, 0, 0, 0], 
                  [0, 1/2, 0, 0],
                  [0, 0, 1, 0]])
    b4 = np.array([1/6, 1/3, 1/3, 1/6])
    c4 = np.sum(A4, axis=1)

    Ydot = np.zeros((4, len(uold)))

    for i in range(4):
        Ydot[i] = f(told + h*c4[i], uold + h*sum([A4[i][j]*Ydot[j] for j in range(i)]))

    # Zdot3 = Zdot[2]
    Zdot3 = f(told + h, uold + h*(-Ydot[0] + 2*Ydot[1]))

    # Why does comparing with a worse approximation approximate the error?? 
    err = h/6*(2*Ydot[1] + Zdot3 - 2*Ydot[2] - Ydot[3])

    unew = uold + h*sum([b4[j]*Ydot[j] for j in range(4)])

    return [unew, err]

def errVSh(f, A, y0, t0, tf):
    H = []
    E = []

    for k in range(1, 13):
        N = 2**k
        tg = np.linspace(t0, tf, N)
        h = (tf - t0)/N

        sol = solution(A, y0, tg)
        _, approx = RK4int(f, y0, t0, tf, h)
        err = approx[-1] - sol[-1]

        H.append(h)
        E.append(norm(err))

    plt.figure()
    plt.loglog(H, [h**4 for h in H], 'k--')
    plt.loglog(H, E, 'r')
    plt.title('Global error (e) vs step size (h) for ERK4-method on test equation')
    plt.legend(['log(e) = log(h^4)', 'e vs h'])


def newstep(tol, err, errold, hold, k):
    return (tol/abs(err))**(2/(3*k)) * (tol/abs(errold))**(-1/(3*k)) * hold
    # return np.power(tol/err, 2/(3*k))*np.power(tol/errold, -1/(3*k))*hold

def adaptiveRK34(f, t0, tf, y0, tol):
    t = [t0]
    y = [y0]

    tn = t0
    hn = abs(tf-t0)*tol**(1/4)/(100*(1 + norm(f(None, y0))))
    # hn = abs(tf - t0)*np.power(tol, 1/4)/(100*(1 + norm(f(None, y0))))
    yn = y0
    rn = tol

    while hn < tf - tn:
        yn, err = RK34step(f, tn, yn, hn)
        tn = tn + hn

        t.append(tn)
        y.append(yn)

        rold = rn
        rn = norm(err)
        hn = newstep(tol, rn, rold, hn, 4)

    t.append(tf)
    y.append(RK34step(f, tn, yn, tf - tn)[0])

    return [np.array(t), np.array(y)]


def adaptiveRK34pro(f, t0, tf, y0, tol):
    h0 = abs(tf-t0)*tol**(1/4)/(100*(1 + norm(f(None, y0))))
    N = int(2*(tf - t0)//h0)

    tgrid = np.zeros(N)
    approx = np.zeros((N, len(y0)))

    tn = t0
    hn = h0
    # hn = abs(tf - t0)*np.power(tol, 1/4)/(100*(1 + norm(f(None, y0))))
    yn = y0
    rn = tol

    i = 0
    while hn < tf - tn:
        if i >= N:
            print('Extending vectors')
            tgrid = np.hstack((tgrid, np.zeros(N)))
            approx = np.vstack((approx, np.zeros((N, len(y0)))))

            N = 2*N

        tgrid[i] = tn
        approx[i] = yn

        yn, err = RK34step(f, tn, yn, hn)
        tn = tn + hn

        rold = rn
        rn = norm(err)
        hn = newstep(tol, rn, rold, hn, 4)

        i += 1

    tgrid = np.trim_zeros(tgrid, 'b')
    approx = approx[approx != np.zeros((1, len(y0)))]

    tgrid = np.append(tgrid, tf)
    approx = np.append(approx, RK34step(f, tn, yn, tf - tn)[0])

    return [tgrid, approx]


def H(u, *parameters):
    x, y = u
    a, b, c, d = parameters
    return c*x + b*y - d*np.log(x) - a*np.log(y)


def BDF2steptesteq(A, yprev, ypreev, h):
    alpha0, alpha1, alpha2 = 1/2, -2, 3/2    
    ynow = -np.dot(inv((alpha2*np.eye(len(A)) - h*A)), alpha1*yprev + alpha0*ypreev)

    return ynow

def BDF2inttesteq(A, t0, tf, ystart, h):
    # y0, y1 = ystart
    N = int(np.ceil((tf - t0)/h))

    tgrid = np.linspace(t0, tf, N)
    approx = np.zeros((N, len(A)))
    approx[0], approx[1] = ystart

    for i in range(2, N):
        approx[i] = BDF2steptesteq(A, approx[i - 1], approx[i - 2], h)

    return [tgrid, approx]


if __name__ == '__main__':
    # Setting up test equation
    l = 1
    A = np.array([[l]])
    y0 = np.array([1])
    testeq = lambda t, y: np.dot(A, y)

    B = np.array([[-1, 10], [0, -3]])
    z0 = np.array([1, 1])

    # Not sure if e vs h should be plotted using RK4int or RK4step
    def task11():
        t0 = 0
        tf = 1

        h = 1e-3
        N = int(np.ceil((tf - t0) // h))
        tg = np.linspace(t0, tf, N)
        sol = solution(A, y0, tg)
        t, y = RK4int(testeq, y0, t0, tf, h)
        plt.figure()
        plt.plot(t, sol, 'k')
        plt.plot(t, y, 'r--')
        plt.title('RK4 applied to test equation')
        plt.legend(['Analytic solution to test equation', f'Numerical solution using ERK4 (h = {h})'])
        plt.xlabel('t (Time)')
        plt.ylabel('yi')

        errVSh(testeq, A, y0, t0, tf)
        plt.show()

    def task12():
        told = 0
        uold = np.array([1])
        h = 1e-3

        u, e = RK34step(testeq, told, uold, h)
        print(f'unew = {u}')
        print(f'Local error ~= {e}')
    

    def task13():
        # No testing required
        pass

    def task14():
        t0 = 0
        tf = 1
        tol = 1e-3

        N = 100
        tg = np.linspace(t0, tf, N)
        sol = solution(A, y0, tg)

        t, y = adaptiveRK34pro(testeq, t0, tf, y0, tol)
        print(t)
        print(y)
        plt.plot(tg, sol, 'k')
        plt.plot(t, y, 'r--')
        plt.title('Applying adaptive RK34-method to test equation')
        plt.legend(['Analytic solution to test equation', 'Numerical solution'])
        plt.xlabel('t (Time)')
        plt.ylabel('yi')

        plt.show()

        # Lower tol => shorter t, bigger errors

    # Change initial conditions
    def task21():
        u0 = np.array([1, 1])
        t0 = 0
        tf = 100
        tol = 1e-6
        a, b, c, d = 3, 9, 15, 15
        # u = [x, y]
        lv = lambda t, u: np.array([a*u[0] - b*u[0]*u[1], c*u[0]*u[1] - d*u[1]])

        t, u = adaptiveRK34(lv, t0, tf, u0, tol)
        plt.figure()
        plt.plot(t, u[:, 0], 'b')
        plt.plot(t, u[:, 1], 'r')
        plt.title(f'Solution to Lotka-Volterra equations using adaptive RK34-method given \n a, b, c, d = {a, b, c, d}, [x0 y0] = {u0}')
        plt.legend(['Prey (rabbits)', 'Predators (foxes)'])
        plt.xlabel('t (Time)')
        plt.ylabel('Amount of animals in some unit')

        plt.figure()
        plt.plot(u[:, 0], u[:, 1], 'purple')
        plt.title(f'Approximative phase portrait of Lotka-Volterra equations using adaptive RK34-method given \n a, b, c, d = {a, b, c, d}, [x0 y0] = {u0}')
        plt.legend([f'(x(t), y(t)), {t0} < t < {tf}'])
        plt.xlabel('Prey (rabbits)')
        plt.ylabel('Predators (foxes)')
        
        plt.figure()
        plt.semilogy(t, [abs(H(xy, a, b, c, d)/H(u0, a, b, c, d) - 1) for xy in u], 'k')
        plt.title('Absolute relative change in H over time')
        plt.legend(['z = |H(x(t), y(t))/(H(x0, y0) - 1)|'])
        plt.xlabel('t (Time)')
        plt.ylabel('ln(z)')

        plt.show()

        # I chose linlog but I don't know why linlin is wrong?

    # Not sure I understand the 'limit cycle'-part?
    def task31():
        tol = 1e-6
        mu = 100
        y0 = np.array([1, 1])
        t0 = 0
        tf = 2*mu
        # y = [y1, y2]
        vdP = lambda t, y: np.array([y[1], mu*(1 - y[0]**2)*y[1] - y[0]])

        t, y = adaptiveRK34(vdP, t0, tf, y0, tol)
        plt.figure()
        plt.plot(t, y[:,1], 'r')
        plt.title(f'Numerical approximation of y2(t) from van der Pol equations given \n mu =  {mu}, [y1(0) y2(0)] = {y0}')
        plt.legend(['y ~= y2(t)'])
        plt.xlabel('t (Time)')
        plt.ylabel('y')

        plt.figure()
        plt.plot(y[:,0], y[:,1], 'purple')
        plt.title(f'Approximative phase portrait of van der Pol equations given \n mu =  {mu}, [y1(0) y2(0)] = {y0}')
        plt.legend([f'(y1(t), y2(t)), {t0} < t < {tf}'])
        plt.show()

    def task32():
        y0 = np.array([2, 0])
        t0 = 0
        tol = 1e-7
        E6_series = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470]#, 680]#, 1000]

        N = []

        for mu in E6_series:
            print(f'Simulating for mu = {mu}')
            vdP = lambda t, y: np.array([y[1], mu*(1 - y[0]**2)*y[1] - y[0]])
            tf = 0.7*mu            
            t, _ = adaptiveRK34(vdP, t0, tf, y0, tol)
            # print(y)
            N.append(len(t) - 1)
        
        plt.figure()
        plt.loglog(E6_series, N, 'r')
        plt.loglog(E6_series, [mu**2 for mu in E6_series], 'k--')
        plt.title('Steps needed for adaptive RK34-method for different \n values of mu from E6 series')
        plt.legend(['Steps recorded for different values of mu', 'ln(N) = ln(N^2) = 2*ln(N)'])
        plt.xlabel('mu')
        plt.ylabel('N (steps)')
        plt.show()

        # q = 2
        # Stiffness increases as mu increases because more steps are needed <=> 
        # smaller steps are taken

    def task33():
        y0 = np.array([2, 0])
        t0 = 0
        tol = 1e-7
        E6_series = [10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]

        N = []

        for mu in E6_series:
            print(f'Simulating for mu = {mu}')
            vdP = lambda t, y: np.array([y[1], mu*(1 - y[0]**2)*y[1] - y[0]])
            tf = 0.7*mu            
            t = solve_ivp(vdP, (t0, tf), y0, method='BDF')['t']#, options=[None, None, (1, tol)])
            print(t)
            N.append(len(t) - 1)
        
        plt.figure()
        plt.loglog(E6_series, N, 'r')
        plt.title('Steps needed for scipy.integrate.solve_ivp for different \n values of mu from E6 series')
        plt.legend(['Steps recorded for different values of mu'])
        plt.xlabel('mu')
        plt.ylabel('N (steps)')
        plt.show()

        # solve_ivp is much faster because it uses implicit methods, which have a higher order
        # meaning that larger steps can be taken without diverging?
        # In this case N does not even seem to increase with mu

    def bonus():
        t0 = 0
        tf = 1
        h = 1e-3
        ystart = np.array([1, 1])

        t, y = BDF2inttesteq(B, t0, tf, ystart, h)
        plt.plot(t, y)
        plt.show()

    # task11()
    # task12()
    # task13()
    # task14()
    # task21()
    # task31()
    task32()
    # task33()
    # bonus()