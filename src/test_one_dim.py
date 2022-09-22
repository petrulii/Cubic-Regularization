import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import src.cubic_reg as cubic_reg
from src.quadratic_obj import quadratic_obj
from src.quadratic_obj import quadratic_obj_1D

def test_one_dim(n, A=None, c=None, x0=None, lambd = 1):
    """
    Comparing results between minimizing an n-dimensional
    and a 1-dimensional quadratic objective function.
    :param n: dimension of the problem
    :param A: symmetric matrix
    :param c: scalar
    :param x0: initial point for minimization
    :param lambd: constraint coefficient
    """
    n = n
    if A is None or c is None:
        # Initializing parameters of the quadratic objective.
        a = np.random.uniform(-1,1,size=(n,n))
        A = (a + a.T)/2
        A[n-1, n-1] = 0
        c = np.random.uniform(-10,10)
    if x0 is None:
        x0 = np.random.uniform(-10,10,(n,))
    f = quadratic_obj(n, A, c, lambd = lambd)
    f_1D = quadratic_obj_1D(n, A, c)

    # Solution to the one-dimensional form of the qudratic objective.
    res = minimize(f_1D, x0[0], method='Nelder-Mead', tol=1e-6)
    # Solution to the n-dimensional form of the qudratic objective.
    cr = cubic_reg.CubicRegularization(x0, f=f, conv_tol=1e-8, L0=1e-5, aux_method="monotone_norm", verbose=0, conv_criterion='gradient')
    x_opt, intermediate_points, n_iter, flag, intermediate_hess_cond = cr.cubic_reg()
    print("Objective value of the n-D quad. objective:", f(x_opt), "argmin:", x_opt)
    print("Objective value of the 1-D quad. objective:", f_1D(res.x), "argmin:", res.x)

    offset = 0.2
    T = np.arange(min(res.x-offset, x_opt[0]-offset), max(res.x+offset, x_opt[0]+offset), 0.01)
    f_t1 = np.array([f_1D(t) for t in T])

    # Compare the two solutions.
    plt.figure()
    plt.plot(T, f_t1, alpha=0.8, label="f(x)")
    plt.scatter(x_opt[0], f(x_opt), marker='o', label="n-D solution")
    plt.scatter(res.x, f_1D(res.x), marker='*', label="1-D solution")
    plt.xlabel('$x_0$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='best')
    plt.title("$f(x)$ w.r.t. $x_0$")
    plt.show()