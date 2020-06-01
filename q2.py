import numpy as np
from math import log

K = 10**3

def f1(xi, k):
    return 0.008*xi**3 - 0.03*xi**2 + 0.2*xi - 1/(k*xi)

def f2(xi, k):
    return 0.024*xi**2 - 0.06*xi + 0.2 + 1/(k*xi**2)

def sum_xi_n(x, n):
    return sum([xi**n for xi in x])

def p(x, k=K):
    print(x)
    return 0.002*sum_xi_n(x, 4) - 0.01*sum_xi_n(x, 3) + 0.1*sum_xi_n(x, 2) \
        + 50*x[0] + 400*x[1] + 80*x[2] + 900*x[3] + 50*x[4] + 30*x[5] + 204 \
        - 1/k * sum([log(xi) for xi in x]) + k/2 *((x[3] + x[5] - 30)**2) \
        + (x[5] + 40 - x[4])**2 + (x[2] + x[3] - x[0])**2 + (x[1] + x[2] - x[4])**2

def grad_p(x, k=K):
    return np.array([
        f1(x[0], k) + 50 - k*(x[2] + x[3] - x[0]),
        f1(x[1], k) + 400 + k*(x[1] + x[2] - x[4]),
        f1(x[2], k) + 80 + k*(2*x[2] + x[1] + x[3] - x[4]),
        f1(x[3], k) + 900 + k*(2*x[3] - x[0] +x[2] + x[5] -30),
        f1(x[4], k) + 50 - k*(x[1] + x[2] - 2*x[4] + x[5] + 40),
        f1(x[5], k) + 30 + k*(2*x[5] + x[3] - x[4] + 10)
    ])

def hess_p(x, k=K):
    return np.array([
        [f2(x[0], k), 0, -k, -k, 0, 0],
        [0, f2(x[1], k), k, 0, -k, 0],
        [0, k, f2(x[2], k) + 2*k, k, -k, 0],
        [-k, 0, k, f2(x[3], k) + 2*k, 0, k],
        [0, -k, -k, 0, f2(x[4], k) + 2*k, -k],
        [0, 0, 0, k, -k, f2(x[5], k) + 2*k]
    ])


def main():
    from descent import BFGS

    print(BFGS(p, grad_p, [1,1,1,1,1,1], 10**(-2), 10**(-5), 10, np.eye(6)))



if __name__ == '__main__':
    main()