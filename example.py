import matplotlib.pyplot as plt
import numpy as np

import src.cubic_reg

from scipy.optimize import rosen as banana_f
from scipy.optimize import rosen_der as banana_grad
from scipy.optimize import rosen_hess as banana_hess

class Function:
    def __init__(self, function='bimodal', method='adaptive', hessian_update='broyden'):
        self.method = method
        if function == 'bimodal':
            self.f = lambda x: -(x[0] ** 2 + 3*x[1] ** 2)*np.exp(1-x[0] ** 2 - x[1] ** 2)
            self.grad = None
            self.hess = None
            self.x0 = np.array([1, 0])  # Start at saddle point!
            np.random.rand(1,2)
        elif function == 'simple':
            self.f = lambda x: x[0]**2*x[1]**2 + x[0]**2 + x[1]**2
            self.grad = lambda x: np.asarray([2*x[0]*x[1]**2 + 2*x[0], 2*x[0]**2*x[1] + 2*x[1]])
            self.hess = lambda x: np.asarray([[2*x[1]**2 + 2, 4*x[0]*x[1]], [4*x[0]*x[1], 2*x[0]**2 + 2]])
            self.x0 = np.array([1, 2])
        elif function == 'quadratic':
            self.f = lambda x: x[0]**2+x[1]**2
            self.grad = lambda x: np.asarray([2*x[0], 2*x[1]])
            self.hess = lambda x: np.asarray([[2, 0], [0, 2]])*1.0
            self.x0 = [2, 2]
        elif function == 'banana':
            self.f = lambda x: banana_f(x)
            self.grad = lambda x: banana_grad(x)
            self.hess = lambda x: banana_hess(x)
            self.x0 = [2, 2]
        self.cr = src.cubic_reg.CubicRegularization(self.x0, f=self.f, gradient=self.grad, hessian=self.hess, conv_tol=1e-4,
                                                    L0=0.00001, aux_method="monotone_norm", verbose=1, conv_criterion='gradient')

    def run(self):
        x_opt, intermediate_points, n_iter, flag = self.cr.cubic_reg()
        return x_opt, intermediate_points, n_iter

    def plot_points(self, intermediate_points):
        xlist = np.linspace(-3.0, 3.0, 50)
        ylist = np.linspace(-3.0, 3.0, 50)
        X, Y = np.meshgrid(xlist, ylist)
        Z = np.zeros_like(X)
        for i in range(0, len(X)):
            for j in range(0, len(X)):
                Z[i, j] = self.f((X[i, j], Y[i, j]))
        points = np.asarray(intermediate_points)
        plt.clf()
        cp = plt.contour(X, Y, Z)
        plt.clabel(cp, inline=True, fontsize=10)
        plt.plot(points[:, 0], points[:, 1])
        plt.scatter(points[-1][0], points[-1][1])
        plt.title('Contour plot of function and path of cubic regularization algorithm')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.show()


if __name__ == '__main__':
    # Choose a function to run it on, and a method to use (original cubic reg or adaptive cubic reg)
    # Function choices: 'bimodal', 'simple', 'quadratic'
    # Method choices: 'adaptive', 'original'
    bm = Function(function='simple', method='original')
    # Run the algorithm on the function
    x_opt, intermediate_points, n_iter = bm.run()
    print('Argmin of function:', x_opt)
    # Plot the path of the algorithm
    bm.plot_points(intermediate_points)
