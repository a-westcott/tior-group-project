import numpy as np
from math import log

K = np.longdouble(10**5)
INF = np.longdouble(10**40)


def f1(xi, k):
    return 0.008*xi**3 - 0.03*xi**2 + 0.2*xi - 1/(k*xi)

def f2(xi, k):
    return 0.024*xi**2 - 0.06*xi + 0.2 + 1/(k*xi**2)

def sum_xi_n(x, n):
    return sum([xi**n for xi in x])

def p(x, k=K):
    # print(x)
    for element in x:
        if element < 0:
            return INF
    return np.longdouble(0.002*sum_xi_n(x, 4) - 0.01*sum_xi_n(x, 3) + 0.1*sum_xi_n(x, 2) \
        + 50*x[0] + 400*x[1] + 80*x[2] + 900*x[3] + 50*x[4] + 30*x[5] + 204 \
        - 1/k * sum([log(xi) for xi in x]) + k/2 *((x[3] + x[5] - 30)**2 \
        + (x[5] + 40 - x[4])**2 + (x[2] + x[3] - x[0])**2 + (x[1] + x[2] - x[4])**2))

def grad_p(x, k=K):
    return np.array([
        f1(x[0], k) + 50 - k*(x[2] + x[3] - x[0]),
        f1(x[1], k) + 400 + k*(x[1] + x[2] - x[4]),
        f1(x[2], k) + 80 + k*(2*x[2] + x[1] + x[3] - x[0] - x[4]),
        f1(x[3], k) + 900 + k*(2*x[3] - x[0] +x[2] + x[5] -30),
        f1(x[4], k) + 50 - k*(x[1] + x[2] - 2*x[4] + x[5] + 40),
        f1(x[5], k) + 30 + k*(2*x[5] + x[3] - x[4] + 10)
    ], dtype=np.longdouble)

def hess_p(x, k=K):
    return np.array([
        [f2(x[0], k) + k, 0, -k, -k, 0, 0],
        [0, f2(x[1], k) + k, k, 0, -k, 0],
        [-k, k, f2(x[2], k) + 2*k, k, -k, 0],
        [-k, 0, k, f2(x[3], k) + 2*k, 0, k],
        [0, -k, -k, 0, f2(x[4], k) + 2*k, -k],
        [0, 0, 0, k, -k, f2(x[5], k) + 2*k]
    ], dtype=np.longdouble)


def main():
    from descent import BFGS, NewtonMethod
    
    print(BFGS(p, 
               grad_p, 
               np.array([1,10,10,1,10,1], dtype=np.longdouble),
               np.longdouble(10**(-2)), 
               np.longdouble(10**(-8)), 
               1, 
               np.eye(6, dtype=np.longdouble)
               ))
    '''
    print(NewtonMethod(p, 
                       grad_p, 
                       np.array([1,10,10,1,10,1], dtype=np.longdouble),
                       np.longdouble(10**(-2)), 
                       np.longdouble(10**(-15)), 
                       1, 
                       hess_p
                       ))

    '''

def main2():
    from numpy.linalg import norm
    
    x= [ 6.82941785, 38.9133082,   8.89436118, 16.13798671, 25.34035365, 13.86359013]

    print(grad_p(x))
    print(norm(grad_p(x)))

    x = [ 5.99062093, 39.74355186,  8.90424708, 15.29402952, 26.17682906, 14.70637969]

    print(grad_p(x))
    print(norm(grad_p(x)))

    x = [10.63159508, 17.56418622,  3.67600254, 24.57398467, 31.21537927,  5.42601509]

    print(grad_p(x))
    print(norm(grad_p(x)))

if __name__ == '__main__':
    main()