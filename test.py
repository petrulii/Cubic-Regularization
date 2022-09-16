import numpy as np
import src.cubic_reg as utils
import matplotlib.pyplot as plt

def polynomial(x):
    n = len(x)
    result = 0
    for i in range(1,n):
        result += (x[i]-x[i-1]*x[0])**2
    for i in range(0,n):
        for j in range(0,n):
            result += (A[i,j]*x[i]*x[j]-c)**2
    return result

def f(x0):
    x = np.power(np.ones(n)*x0, np.arange(1,n+1))
    result = 0
    for i in range(0,n):
        for j in range(0,n):
            result += (A[i,j]*x[i]*x[j]-c)**2
    return result

np.random.seed(3)
n = 3
a = np.random.uniform(-1,1,size=(n,n))
A = (a + a.T)/2
A[n-1, n-1] = 0
c = 5
x0 = np.random.randn(n)

T = np.arange(-10, 10, 0.1)
f_t1 = np.array([f(t) for t in T])
f_t2 = np.array([np.power((f(t)), 3.5) for t in T])

#cr = utils.CubicRegularization(x0, f=polynomial, conv_tol=1e-8, L0=0.00001, aux_method="monotone_norm", verbose=0, conv_criterion='gradient')
#x_opt, intermediate_points, n_iter, flag, intermediate_hess_cond = cr.cubic_reg()
#print(x_opt, f(x_opt))
print(A)
plt.figure()
plt.axis([-10, 10, -50000000, 450000000])
plt.plot(T, f_t1, 'b', T, f_t2, 'r')
#plt.scatter(x_opt[0], f(x_opt))
plt.show()