import numpy as np
import src.cubic_reg as cubic_reg
from matplotlib import pyplot as plt

def quadratic_obj(n, A, c, lambd=1):
    """
    An n-dimensional quadratic objective function.
    :param n: dimension of the problem
    :param A: symmetric matrix
    :param c: scalar
    """
    assert(n==A.shape[0])
    def f(x):
        term1 = 0
        term2 = 0
        for i in range(1,n):
            term1 += (x[i]-x[i-1]*x[0])**2
        for i in range(0,n):
            for j in range(0,n):
                term2 += (A[i,j]*x[i]*x[j]-c)
        return lambd*term1+term2**2
    return f

def quadratic_obj_1D(n, A, c):
    """
    A one-dimensional quadratic objective function.
    :param n: length of the power series
    :param A: symmetric matrix
    :param c: scalar
    """
    assert(n==A.shape[0])
    def f_1D(x0):
        """
        A quadratic objective with quadratic constraints w.r.t. x[0]
        for testing the accuraccy of the n-dimensional solution.
        """
        x = np.power(np.ones(n)*x0, np.arange(1,n+1))
        term2 = 0
        for i in range(0,n):
            for j in range(0,n):
                term2 += (A[i,j]*x[i]*x[j]-c)
        return term2**2
    return f_1D

def test_quadratic_obj(n, A=None, c=None, lambd=1, nb_minima=1):
    """
    Test cubic regularization on a quadratic usually non-convex objective
    with the global minimum at 0.
    :param n: dimension of the problem
    :param A: symmetric matrix
    :param c: scalar
    :param lambd: constraint coefficient
    :param nb_minima: number of times to run cubic reg. from diff. initial points
    """
    # Dimension of the problem.
    n = n
    if A is None or c==None:
        # Initializing parameters of the quadratic objective.
        a = np.random.uniform(-1,1,size=(n,n))
        A = (a + a.T)/2
        A[n-1, n-1] = 0
        c = np.random.uniform(-10,10)
    f = quadratic_obj(n, A, c, lambd = 1)
    # All local minima found from different initial points.
    minima = np.zeros(nb_minima)
    # Minimize f using cubic regularization.
    for i in range(nb_minima):
        x0 = np.random.uniform(-10,10,(n,))
        cr = cubic_reg.CubicRegularization(x0, f=f, conv_tol=1e-10, L0=1e-4, aux_method="monotone_norm", verbose=0, conv_criterion='gradient', maxiter=10000)
        x_opt, intermediate_points, n_iter, flag, intermediate_hess_cond = cr.cubic_reg()
        minima[i] = f(x_opt)
        print("Objective value:", minima[i], ", iterations:", n_iter, ", experiment:", i, "\nArgmin x*:", x_opt)
    # Round all minima to a specified accuracy.
    minima = np.around(minima, decimals=2)
    print("Number of local minima found:", len(np.unique(minima)), ", best local minimum:", np.min(minima))

def plot_cond(intermediate_hess_cond, n_iter):
    """
    Visualize the shape of the function at each iteration by plotting
    the condition number of the hessian at first 1000 iterations.
    :param intermediate_hess_cond: hessian condition number at each iteration
    :param n_iter: number of iterations
    """
    k = min(n_iter, 1000)
    X = np.arange(0, k, 1)
    plt.figure()
    plt.plot(X, intermediate_hess_cond[0:k])
    plt.show()