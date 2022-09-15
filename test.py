import numpy as np
import src.cubic_reg as utils
import matplotlib.pyplot as plt

def polynomial(x):
    n = len(x)
    term1, term2 = 0, 0
    for i in range(1,n):
        term1 += (x[i]-x[i-1]*x[0])
    for i in range(0,n):
        for j in range(0,n):
            term2 += (A[i,j]*x[i]*x[j]-c)
    return term1**2+term2**2

def f(x0):
    x = np.power(np.ones(n)*x0, np.arange(1,n+1))
    term2 = 0
    for i in range(0,n):
        for j in range(0,n):
            term2 += (A[i,j]*x[i]*x[j]-c)
    return term2**2

#np.random.seed(3)
n = 3
a = np.random.uniform(-1,1,size=(n,n))
A = (a + a.T)/2
A[n-1, n-1] = 0
c = 0
x0 = np.random.randn(n)

T = np.arange(-1, 1, 0.01)
f_t = np.array([f(t) for t in T])

cr = utils.CubicRegularization(x0, f=polynomial, conv_tol=1e-8, L0=0.00001, aux_method="monotone_norm", verbose=0, conv_criterion='gradient')
x_opt, intermediate_points, n_iter, flag = cr.cubic_reg()
print(x_opt, f(x_opt))
print(A)
plt.figure()
plt.axis([-0.7, 0.7, -0.005, 0.015])
plt.plot(T, f_t)
plt.scatter(x_opt[0], f(x_opt))
plt.show()