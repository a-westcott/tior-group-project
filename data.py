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
    return  s_yi2 - 2*x[0]*s_yixi3 - 2*x[1]*s_yixi2 - 2*x[2]*s_yixi - 2*x[3]*s_yi + \
            x[0]**2 * s_xi6 + x[1]**2 * s_xi4 + x[2]**2 * s_xi2 + x[3]**2 * s_xi0 + \
            2*x[0]*x[1]*s_xi5 + 2*x[0]*x[2]*s_xi4 + 2*x[0]*x[3]*s_xi3 + \
            2*x[1]*x[2]*s_xi3 + 2*x[1]*x[3]*s_xi2 + 2*x[2]*x[3]*s_xi
  
def gradf(x):
    return np.array([
        -2*s_yixi3 + 2*x[0]*s_xi6 + 2*x[1]*s_xi5 + 2*x[2]*s_xi4 + 2*x[3]*s_xi3,
        -2*s_yixi2 + 2*x[1]*s_xi4 + 2*x[0]*s_xi5 + 2*x[2]*s_xi3 + 2*x[3]*s_xi2,
        -2*s_yixi  + 2*x[2]*s_xi2 + 2*x[0]*s_xi4 + 2*x[1]*s_xi3 + 2*x[3]*s_xi ,
        -2*s_yi    + 2*x[3]*s_xi0 + 2*x[0]*s_xi3 + 2*x[1]*s_xi2 + 2*x[2]*s_xi  ])

def hessf(x):
    return np.array([
    [2*s_xi6, 2*s_xi5, 2*s_xi4, 2*s_xi3],
    [2*s_xi5, 2*s_xi4, 2*s_xi3, 2*s_xi2],
    [2*s_xi4, 2*s_xi3, 2*s_xi2, 2*s_xi ],
    [2*s_xi3, 2*s_xi2, 2*s_xi , 2*s_xi0]])


def main():
    #print(s_xi)
    # currently we have a negative eigenvalue, and that is bad
    #print(eigh(hessf(1), eigvals_only=True))
    #print(eig(hessf(1), right=False))

    b1 = 1.041251333*10**-3
    b2 = -2.28569*10**-2
    b3 = 2.616989*10**-1
    b4 = 3.94403

    sse0 = f([b1, b2, b3, b4])
    print(sse0)


if __name__ == '__main__':
    main()