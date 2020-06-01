import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh, inv


def multiVariableHalfOpen(f, x, d, T):
    '''
    INPUT
        f: multivariable function to minimise
        x: starting point
        d: direction vector
        T: upper bound increment parameter
    OUTPUT
        a: lower bound on the location of minimum of f in direction d from x
        b: upper bound on the location of minimum of f in direction d from x
    '''
    k = 1
 
    p = x
    q = x + T*d

    fp = f(p)
    fq = f(q)

    
    while fp > fq:
        k += 1
        p = q
        fp = fq
        q = p + (2**(k-1))*T*d
        fq = f(q)
            
    if k == 1:
        a = 0
        b = T

    elif k == 2:
        a = 0
        b = 3*T

    else:
        u = np.arange(0,k,1)
        v = np.arange(0,k-2,1)

        a = T*sum((2*np.ones(k-2))**v)       
        b = T*sum((2*np.ones(k))**u) 
        
    return a, b

def multiVariableGoldenSectionSearch(f, a, b, tolerance, x, d):
    '''
    performs golden section search for finding minimum of f along the
    direction d, starting at x, where the minimum has upper and lower bound [a, b]
    '''
    if b <= a:
        raise ValueError('b must be strictly greater than a')
    if tolerance <= 0:
        raise ValueError('tolerance must be strictly positive')

    # Begin the Golden Search algorithm

    gamma = (np.sqrt(5) - 1)/2 
    k = 1

    p = b - gamma*(b-a)
    q = a + gamma*(b-a)

    fp = f(x + p*d)
    fq = f(x + q*d)

    while b-a >= 2*tolerance:
        k += 1

        if fp <= fq:
            b = q
            q = p
            fq = fp
            p = b - gamma*(b-a)
            fp = f(x + p*d)

        else:
            a = p
            p = q
            fp = fq
            q = a + gamma*(b-a)
            fq = f(x + q*d)

    # Midpoint of the final interval
    minEstimate = (a+b)/2
    fminEstimate = f(x + minEstimate*d)  
    return minEstimate, fminEstimate

def BFGS(f, gradf, x0, tolerance1, tolerance2, T, H0, **kwargs):
    '''
    INPUT
        f:          the multivariable function to minimise
        gradf:      function which returns the gradient vector of f evaluated at x
        x0:         the starting iterate
        tolerance1: tolerance for stopping criterion of steepest descent method
        tolerance2: tolerance for stopping criterion of line minimisation
        T:          parameter used by the "improved algorithm for finding an upper bound for the minimum" along 
                    each given descent direction
    OUTPUT
        xminEstimate: estimate of the minimum
        fminEstimate: the value of f at xminEstimate
        k:            iteration counter
    '''
    k = 0
    iteration_number = 0

    xk = np.copy(x0)
    xk_old = np.copy(x0)
    H_old = H0
    
    while norm(gradf(xk)) >= tolerance1:
        iteration_number += 1  

        # this is definitely a bad idea
        '''
        # Correction if det H_old gets too large or small
        H_old /= np.amax(H_old) 
        '''

        # Get dk as a row vector
        dk = -H_old.dot(gradf(xk))
        # minimise f with respect to t in the direction dk

        # (1) find upper and lower bound, [a,b], for the stepsize t
        a, b = multiVariableHalfOpen(f, xk, dk, T)

        # (2) use golden section algorithm to estimate the stepsize t in [a,b] which minimises f in the direction dk from xk
        tmin, fmin = multiVariableGoldenSectionSearch(f, a, b, tolerance2, xk, dk)

        k += 1

        xk += tmin*dk
        xk_new = xk_old +tmin*dk
        
        sk=(xk_new - xk_old).T
        gk= (gradf(xk_new)-gradf(xk_old)).T
        rk=(H_old.dot(gk))/(sk.dot(gk))
        
        H_new = H_old + (1 + rk.dot(gk))/(sk.dot(gk))*np.outer(sk, sk) - np.outer(sk, rk) - np.outer(rk, sk)
        
        xk_old = xk_new
        H_old = H_new

    xminEstimate = xk
    fminEstimate = f(xminEstimate)
    return xminEstimate, fminEstimate, k

def NewtonMethod(f, gradf, x0, tolerance1, tolerance2, T, hessf, **kwargs):
    '''
    INPUT
        f:          the multivariable function to minimise
        gradf:      function which returns the gradient vector of f evaluated at x 
        x0:         the starting iterate
        tolerance1: tolerance for stopping criterion of steepest descent method
        tolerance2: tolerance for stopping criterion of line minimisation
        T:          parameter used by the "improved algorithm for finding an upper bound for the minimum" along 
                    each given descent direction
    OUTPUT
        xminEstimate: estimate of the minimum
        fminEstimate: the value of f at xminEstimate
        k:            iteration counter
    '''
    k = 0
    xk = x0
    while norm(gradf(xk)) >= tolerance1:
        gradf(xk)
        Hessian = hessf(xk)
        
        # the following seemed to break the algorithm, so its gone now
        '''
        # Correction if det Hessian gets too large or small
        Hessian /= np.amax(Hessian) 
        '''

        # Checks to see if the Hessian is positive definite
        if np.all(eigh(Hessian, eigvals_only=True) > 0):
            # the Newton direction - as a row vector
            dk = -(inv(Hessian).dot(gradf(xk).T)).T
        else:
            # the steepest descent direction.
            dk = -gradf(xk)
                   
        # Minimise f with respect to t in the direction dk
                   
        # (1) find upper and lower bound,[a,b],for the stepsize t 
        a, b = multiVariableHalfOpen(f, xk, dk, T)
        
        # (2) use golden section algorithm to estimate the stepsize t in [a,b] which minimises f in the direction dk from xk
        tmin, fmin = multiVariableGoldenSectionSearch(f, a, b, tolerance2, xk, dk)
                   
        k += 1
        xk += tmin*dk
                   
    xminEstimate = xk
    fminEstimate = f(xminEstimate)
    return xminEstimate, fminEstimate, k

def steepestDescentMethod(f, gradf, x0, tolerance1, tolerance2, T, **kwargs):
    '''
    INPUT
        f:          the multivariable function to minimise
        gradf:      function which returns the gradient vector of f evaluated at x 
        x0:         the starting iterate
        tolerance1: tolerance for stopping criterion of steepest descent method
        tolerance2: tolerance for stopping criterion of line minimisation
        T:          parameter used by the "improved algorithm for finding an upper bound for the minimum" along 
                    each given descent direction
    OUTPUT
        xminEstimate: estimate of the minimum
        fminEstimate: the value of f at xminEstimate
        k:            iteration counter
    '''
    k = 0
    xk = x0
    
    while norm(gradf(xk)) >= tolerance1:
        # Steepest descent direction
        dk = -gradf(xk)
        
        # Minimise f with respect to t in the direction dk
        
        # (1) find upper and lower bound, [a,b], for the stepsize t
        a, b = multiVariableHalfOpen(f, xk, dk, T)

        # (2) use golden section algorithm  to estimate the stepsize t in [a,b] which minimises f in the direction dk from xk
        tmin, fmin = multiVariableGoldenSectionSearch(f, a, b, tolerance2, xk, dk)

        k += 1
        xk += tmin*dk
        
    xminEstimate = xk
    fminEstimate = f(xminEstimate)
    return xminEstimate, fminEstimate, k