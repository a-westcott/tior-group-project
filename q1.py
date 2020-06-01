import numpy as np
from scipy.linalg import eigh, eig
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


def main():
    from descent import BFGS, NewtonMethod

    x0 = np.array([-6,200,-30,2], dtype=np.longdouble)

    print(BFGS(f, 
               gradf, 
               x0, 
               np.longdouble(10**(-5)), 
               np.longdouble(10**(-10)), 
               1, 
               np.eye(4, dtype=np.longdouble)
               )
    )

    print(NewtonMethod(f, 
                       gradf, 
                       x0, 
                       np.longdouble(10**(-5)), 
                       np.longdouble(10**(-10)), 
                       1, 
                       hessf
                   )
    )




if __name__ == '__main__':
    main()