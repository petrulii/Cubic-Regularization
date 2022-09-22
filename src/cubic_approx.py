import numpy as np
import matplotlib.pyplot as plt
import torch
from autograd import grad

def hess(f):
    """
    Hessian of the quadratic objective.
    :param f: function to find the hessian of
    """
    def hessian_f(x):
        x_tens = torch.from_numpy(x)
        hess = torch.autograd.functional.hessian(f,x_tens)
        return hess.numpy()
    return hessian_f

def plot_approximations(f, gradient=None, hessian=None):
    """
    Visualize how the cubic approximation depends on the M term.
    :param f: function to approximate
    :param gradient: gradient oracle of f
    :param hessian: hessian oracle of f
    """
    if gradient==None:
        gradient = grad(f)
    if hessian==None:
        hessian = hess(f)
    x0 = np.ones(1)*(-10)

    f_first = lambda th, y: f(th) + np.dot(gradient(th), y-th)
    f_second = lambda th, y: f(th) + np.dot(gradient(th), y-th) + 0.5*np.dot(hessian(th), (y-th)**2)
    f_cubic = lambda th, y, mu: f(th) + np.dot(gradient(th), y-th) + 0.5*np.dot(hessian(th), (y-th)**2) + mu/6*(np.linalg.norm(y-th)**3)

    T = np.arange(-20, 6, 0.1)
    f_t = np.array([f(t) for t in T])
    dumm = np.ones(1)
    f_t_first = np.array([f_first(x0, t*dumm) for t in T])
    f_t_second = np.array([f_second(x0, t*dumm) for t in T])
    f_t_cubic_1e2 = np.array([f_cubic(x0, t*dumm, 1e2) for t in T])
    f_t_cubic_1e3 = np.array([f_cubic(x0, t*dumm, 1e3) for t in T])

    plt.figure(figsize=(9,6))
    plt.plot(T, f_t, label="$f(x)$", color='r')
    plt.plot(T, f_t_first, label="Lin. approx. of $f(x) around x_0$", color='b', alpha=0.3)
    plt.plot(T, f_t_second, label="Quad. approx. of $f(x) ar. x_0$", color='b', alpha=0.5)
    plt.plot(T, f_t_cubic_1e2, label="Cub. approx. of $f(x) ar. x_0,\ M=100$", color='m', alpha=0.5)
    plt.plot(T, f_t_cubic_1e3, label="Cub. approx. of $f(x) ar. x_0,\ M=1000$", color='m', alpha=0.7)
    plt.scatter(x0, f(x0), label="$f(x_0)$", color='r', marker='*')
    plt.xlabel('x')
    plt.legend(loc='best', prop={'size': 10})
    plt.show()