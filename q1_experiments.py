from scipy.optimize import minimize
import pandas as pd
import numpy as np
from scipy.linalg import eigh, eig
from itertools import product
from tqdm import tqdm
from time import time
from descent import NewtonMethod, BFGS

N_DIMENSIONS = 4
X,Y = 0,1

data = np.genfromtxt('traffic_data.csv', delimiter=',', skip_header=1).T


s_xi = data[X].sum()
s_yi = data[Y].sum()

s_xi0 = sum([xi**0 for xi in data[X]])
s_xi2 = sum([xi**2 for xi in data[X]])
s_xi3 = sum([xi**3 for xi in data[X]])
s_xi4 = sum([xi**4 for xi in data[X]])
s_xi5 = sum([xi**5 for xi in data[X]])
s_xi6 = sum([xi**6 for xi in data[X]])

s_yi2 = sum([yi**2 for yi in data[Y]])

s_yixi  = sum([data[Y][i]*(data[X][i]**1) for i in range(len(data[X]))])
s_yixi2 = sum([data[Y][i]*(data[X][i]**2) for i in range(len(data[X]))])
s_yixi3 = sum([data[Y][i]*(data[X][i]**3) for i in range(len(data[X]))])



def f(x):
    return  np.longdouble(s_yi2 - 2*x[0]*s_yixi3 - 2*x[1]*s_yixi2 - 2*x[2]*s_yixi - 2*x[3]*s_yi + \
            x[0]**2 * s_xi6 + x[1]**2 * s_xi4 + x[2]**2 * s_xi2 + x[3]**2 * s_xi0 + \
            2*x[0]*x[1]*s_xi5 + 2*x[0]*x[2]*s_xi4 + 2*x[0]*x[3]*s_xi3 + \
            2*x[1]*x[2]*s_xi3 + 2*x[1]*x[3]*s_xi2 + 2*x[2]*x[3]*s_xi)
  
def gradf(x):
    return np.array([
        -2*s_yixi3 + 2*x[0]*s_xi6 + 2*x[1]*s_xi5 + 2*x[2]*s_xi4 + 2*x[3]*s_xi3,
        -2*s_yixi2 + 2*x[1]*s_xi4 + 2*x[0]*s_xi5 + 2*x[2]*s_xi3 + 2*x[3]*s_xi2,
        -2*s_yixi  + 2*x[2]*s_xi2 + 2*x[0]*s_xi4 + 2*x[1]*s_xi3 + 2*x[3]*s_xi ,
        -2*s_yi    + 2*x[3]*s_xi0 + 2*x[0]*s_xi3 + 2*x[1]*s_xi2 + 2*x[2]*s_xi  ], dtype=np.longdouble)

def hessf(x):
    return np.array([
    [2*s_xi6, 2*s_xi5, 2*s_xi4, 2*s_xi3],
    [2*s_xi5, 2*s_xi4, 2*s_xi3, 2*s_xi2],
    [2*s_xi4, 2*s_xi3, 2*s_xi2, 2*s_xi ],
    [2*s_xi3, 2*s_xi2, 2*s_xi , 2*s_xi0]], dtype=np.longdouble)


def scipy_experiment(algorithms, x_range, tolerance1s, Ts, trials):
    '''
    Can provide list of descent algorithms, tolerances (for descent and stepsize), Ts and will try every combination.
    Each combination is trialled 'trials' number of times, and points are chosen from x_range (x_min, x_max) in Rn space 
    Returns results in a pandas dataframe
    '''
    columns = ['algorithm', 'tolerance1', 'T', 'x0', 'xmin', 'fmin', 'iterations', 'time']
    results = {column: [] for column in columns}
    
    for algorithm, tolerance1, T in tqdm(list(product(algorithms, tolerance1s, Ts))*trials):
        # Choose a random point in the x_range
        x0 = np.random.rand(N_DIMENSIONS)*(x_range[1]-x_range[0]) + x_range[0]
        
        if algorithm == 'newtons':
            start = time()
            run = NewtonMethod(f, gradf, x0, tolerance1, 10**(-5), T, hessf)
            end = time()
            t = end - start
            xmin, fmin, k = run

        elif algorithm == 'bfgs':
            start = time()
            run = BFGS(f, gradf, x0, tolerance1, 10**(-5), T, np.eye(4, dtype=np.longdouble))
            end = time()
            t = end - start
            xmin, fmin, k = run

        elif algorithm == 'Nelder-mead':
            start = time()
            run = minimize(f, x0, method=algorithm, tol=tolerance1, options={'xatol': 1e-8, 'disp': False})
            end = time()
            t = end - start
            xmin, fmin, k = run.x, run.fun, run.nit

        else:
            start = time()
            run = minimize(f, x0, method=algorithm, jac=gradf, hess=hessf, 
                           tol=tolerance1, options={'xatol': 1e-8, 'disp': False})
            end = time()
            t = end - start
            xmin, fmin, k = run.x, run.fun, run.nit
        
        # Record trial
        
        for column, result in zip(columns, [algorithm, tolerance1, T, 
                                            str(np.round(x0, 2)), str(np.round(xmin, 2)), round(fmin, 2), k, t]):
            results[column].append(result)
    return pd.DataFrame(results)


def main():
    algs = ['Nelder-mead', 'Newton-CG', 'trust-ncg', 'newtons', 'bfgs']
    x_range = (-5,5)
    t1s = [10**(x-3) for x in range(10)]
    Ts = [10**((x-4)/2) for x in range(9)]
    trials = 100
    results = scipy_experiment(algs, x_range, t1s, Ts, trials)
    results.to_csv('res2.csv')

def main2():
    algs = ['Nelder-mead', 'Newton-CG', 'trust-ncg']
    x_range = (-5,5)
    t1s = [10**(6)]
    Ts = [1]
    trials = 1
    results = scipy_experiment(algs, x_range, t1s, Ts, trials)
    results.to_csv('res2.csv')

if __name__ == '__main__':
    main()