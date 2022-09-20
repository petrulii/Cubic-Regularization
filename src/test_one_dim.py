import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cubic_reg as cubic_reg

def f(x):
    """
    A quadratic objective with quadratic constraints.
    """
    n = len(x)
    term1 = 0
    term2 = 0
    for i in range(1,n):
        term1 += (x[i]-x[i-1]*x[0])**2
    for i in range(0,n):
        for j in range(0,n):
            term2 += (A[i,j]*x[i]*x[j]-c)
    return lambd*term1+term2**2

def f_one_dimensional(x0):
    """
    A quadratic objective with quadratic constraints w.r.t. one variable
    for testing the accuraccy of the n-dimensional solution.
    """
    x = np.power(np.ones(n)*x0, np.arange(1,n+1))
    term2 = 0
    for i in range(0,n):
        for j in range(0,n):
            term2 += (A[i,j]*x[i]*x[j]-c)
    return term2**2

np.random.seed(6)

# Dimension of the problem.
n = 4
# Initializing parameters of the quadratic objective.
a = np.random.uniform(-10,10,size=(n,n))
A = (a + a.T)/2
A[n-1, n-1] = 0
lambd = 1
c = np.random.uniform(-10,10)
x0 = np.random.uniform(-10,10,(n,))
# Solution to the one-dimensional form of the qudratic objective.
res = minimize(f_one_dimensional, x0[0], method='Nelder-Mead', tol=1e-6)
# Solution to the n-dimensional form of the qudratic objective.
cr = cubic_reg.CubicRegularization(x0, f=f, conv_tol=1e-8, L0=0.00001, aux_method="monotone_norm", verbose=0, conv_criterion='gradient')
x_opt, intermediate_points, n_iter, flag, intermediate_hess_cond = cr.cubic_reg()

offset = 0.2
T = np.arange(min(res.x-offset, x_opt[0]-offset), max(res.x+offset, x_opt[0]+offset), 0.01)
f_t1 = np.array([f_one_dimensional(t) for t in T])

# Compare the two solutions.
plt.figure()
#plt.axis([res.x-0.5, res.x+0.5, -100, np.max(f_t1)])
plt.plot(T, f_t1, alpha=0.8, label="f(x)")
plt.scatter(x_opt[0], f(x_opt), marker='o', label="n-D solution")
plt.scatter(res.x, f_one_dimensional(res.x), marker='*', label="1-D solution")
plt.xlabel('$x_0$')
plt.ylabel('$f(x)$')
plt.legend(loc='best')
plt.title("$f(x)$ w.r.t. $x_0$")
plt.show()