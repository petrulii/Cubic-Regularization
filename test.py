import numpy as np
import src.cubic_reg as utils

def f(x):
    n = len(x)
    c = 0
    result = 0
    for i in range(1,n):
        result += (x[i]-x[i-1]*x[0])**2
    for i in range(0,n):
        for j in range(0,n):
            result += (A[i,j]*x[i]*x[j]-c)**2
    return result

np.random.seed(3)
n = 7
a = np.random.randint(-100,100,size=(n,n))
A = (a + a.T)/2
x0 = np.random.randn(n)
print("A: ", A)

cr = utils.CubicRegularization(x0, f=f, conv_tol=1e-8, L0=0.00001, aux_method="monotone_norm", verbose=0, conv_criterion='gradient')
x_opt, intermediate_points, n_iter, flag = cr.cubic_reg()
print("Argmin of function:", x_opt)
x_pows = np.ones(n)*x_opt[0]
x_rise = np.arange(1,n+1)
x_pows = np.power(x_pows, x_rise)
print("Power series:", x_pows)
print("Iterations:", n_iter, ", f_opt:", f(x_opt))
