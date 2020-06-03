import numpy as np
from math import log

K = np.longdouble(10**7)
INF = np.longdouble(10**40)

delta = 5

dem1 = 30 
dem2 = 40 + delta
b_o1v1 = 20 
b_o2v2 = 20 
b_v1v2 = 30 
b_v1v3 = 900  
b_v2v4 = 50 
b_v4v3 = 80
b_v3d  = 50 
b_v4d  = 400

def f1(xi, k):
    return 0.008*xi**3 - 0.03*xi**2 + 0.2*xi - 1/(k*xi)

def f2(xi, k):
    return 0.024*xi**2 - 0.06*xi + 0.2 + 1/(k*xi**2)

def sum_xi_n(x, n):
    return sum([xi**n for xi in x])

def constant():
    return (sum([a*(0.002*a**3 - 0.01*a**2 + 0.1*a) for a in (dem1, dem2)]) + dem1*b_o1v1 + dem2*b_o2v2)

def p(x, k=K):
    #print(x)
    for element in x:
        if element < 0:
            return INF
    return np.longdouble(0.002*sum_xi_n(x, 4) - 0.01*sum_xi_n(x, 3) + 0.1*sum_xi_n(x, 2) \
        + b_v3d*x[0] + b_v4d*x[1] + b_v4v3*x[2] + b_v1v3*x[3] + b_v2v4*x[4] + b_v1v2*x[5] + constant() \
        - 1/k * sum([log(xi) for xi in x]) + k/2 *((x[3] + x[5] - dem1)**2 \
        + (x[5] + dem2 - x[4])**2 + (x[2] + x[3] - x[0])**2 + (x[1] + x[2] - x[4])**2))

def grad_p(x, k=K):
    return np.array([
        f1(x[0], k) + b_v3d - k*(x[2] + x[3] - x[0]),
        f1(x[1], k) + b_v4d + k*(x[1] + x[2] - x[4]),
        f1(x[2], k) + b_v4v3 + k*(2*x[2] + x[1] + x[3] - x[0] - x[4]),
        f1(x[3], k) + b_v1v3 + k*(2*x[3] - x[0] +x[2] + x[5] - dem1),
        f1(x[4], k) + b_v2v4 - k*(x[1] + x[2] - 2*x[4] + x[5] + dem2),
        f1(x[5], k) + b_v1v2 + k*(2*x[5] + x[3] - x[4] + (dem2 - dem1))
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
               np.array([35,30,9,29,47,7], dtype=np.longdouble),
               np.longdouble(10**(-3)), 
               np.longdouble(10**(-8)), 
               1, 
               np.eye(6, dtype=np.longdouble)
               ))
    


if __name__ == '__main__':
    main()