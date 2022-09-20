import numpy as np
import matplotlib.pyplot as plt
import torch

def f_x0(x0):
    x = np.power(np.ones(n)*x0, np.arange(1,n+1))
    term2 = 0
    for i in range(0,n):
        for j in range(0,n):
            term2 += (A[i,j]*x[i]*x[j]-c)
    return term2**2

def f(x):
    n = len(x)
    term1 = 0
    term2 = 0
    for i in range(1,n):
        term1 += 10*(x[i]-x[i-1]*x[0])**2
    for i in range(0,n):
        for j in range(0,n):
            term2 += (A[i,j]*x[i]*x[j]-c)
    return 1000*term1+term2**2

def hess_f(x):
    x_tens = torch.from_numpy(x)
    hess = torch.autograd.functional.hessian(f,x_tens)
    return hess.numpy()

def f_terms(x):
    n = len(x)
    term1 = 0
    term2 = 0
    for i in range(1,n):
        term1 += (x[i]-x[i-1]*x[0])**2
    for i in range(0,n):
        for j in range(0,n):
            term2 += (A[i,j]*x[i]*x[j]-c)
    return term1, term2**2

def plot_f(x_min, x_opt, f_x_opt):
    T = np.arange(-10, 10, 0.05)
    f_t1 = np.array([f_x0(t) for t in T])
    plt.figure()
    plt.axis([-10, 10, -1000, 100000])
    plt.plot(T, f_t1)
    plt.scatter(x_min, f_x0(x_min), marker='*')
    plt.scatter(x_opt[0], f_x_opt, marker='.')
    plt.scatter(x_opt[0], f_x0(x_opt[0]), marker='+')
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    plt.show()

def plot_cond(intermediate_hess_cond):
    k = min(n_iter, 1000)
    X = np.arange(0, k, 1)
    plt.figure()
    plt.plot(X, intermediate_hess_cond[0:k])
    plt.show()