import matplotlib.pyplot as plt
import numpy as np


from scipy.optimize import rosen as banana_f
from scipy.optimize import rosen_der as banana_grad
from scipy.optimize import rosen_hess as banana_hess

x = np.random.rand(2,1)
f = lambda x: x[0,0]**2*x[1,0]**2 + x[0,0]**2 + x[1,0]**2
grad = lambda x: np.asarray([2*x[0,0]*x[1,0]**2 + 2*x[0,0], 2*x[0,0]**2*x[1,0] + 2*x[1,0]])
hess = lambda x: np.asarray([[2*x[1,0]**2 + 2, 4*x[0,0]*x[1,0]], [4*x[0,0]*x[1,0], 2*x[0,0]**2 + 2]])
print(f(x))
print(grad(x))
print(hess(x))

class CubicRegularization(Algorithm):
    def __init__(self, x0, f=None, gradient=None, hessian=None, L=None, L0=None, kappa_easy=0.0001, maxiter=10000,
                 submaxiter=10000, conv_tol=1e-5, conv_criterion='gradient', epsilon=2*np.sqrt(np.finfo(float).eps), aux_method="trust_region", verbose=0):
        Algorithm.__init__(self, x0, f=f, gradient=gradient, hessian=hessian, L=L, L0=L0, kappa_easy=kappa_easy,
                           maxiter=maxiter, submaxiter=submaxiter, conv_tol=conv_tol, conv_criterion=conv_criterion,
                           epsilon=epsilon, aux_method=aux_method, verbose=verbose)

        self.f_cubic = lambda x, y, mu: self.f(x) + np.matmul(self.grad_x.T,(y-x)) + 0.5*np.matmul((np.matmul(self.hess_x,(y-x))).T,(y-x)) + mu/6*(np.linalg.norm(y-x)**3)

    def _cubic_approx(self, f_x, s, mk):
        """
        Compute the value of the cubic approximation to f at the proposed next point
        :param f_x: Value of f(x) at current point x
        :param s: Proposed step to take
        :return: Value of the cubic approximation to f at the proposed next point
        """
        return f_x + np.matmul(self.grad_x.T,s) + 0.5*np.matmul((np.matmul(self.hess_x,s)).T,s) + mk/6*(np.linalg.norm(s)**3)

    def cubic_reg(self):
        """
        Run the cubic regularization algorithm
        :return: x_new: Final point
        :return: intermediate_points: All points visited by the cubic regularization algorithm on the way to x_new
        :return: iter: Number of iterations of cubic regularization
        """
        iter = flag = 0
        converged = False
        x_new = self.x0
        mk = self.L0
        intermediate_points = [x_new]
        while iter < self.maxiter and converged is False:
            x_old = x_new.copy()
            x_new, mk, flag = self._find_x_new(x_old, mk)
            self.grad_x = self.gradient(x_new)
            self.hess_x = self.hessian(x_new)
            self.f_x = self.f(x_new)
            self.lambda_nplus, lambda_min = self._compute_lambda_nplus()
            converged = self._check_convergence(lambda_min, mk)
            if flag != 0:
                print(RuntimeWarning('Convergence criteria not met, likely due to round-off error or ill-conditioned '
                                     'Hessian.'))
                return x_new, intermediate_points, iter, flag
            intermediate_points.append(x_new)
            iter += 1
        return x_new, intermediate_points, iter, flag

    def _find_x_new(self, x_old, mk):
        """
        Determine what M_k should be and compute the next point for the cubic regularization algorithm
        :param x_old: Previous point
        :param mk: Previous value of M_k (will start with this if L isn't specified)
        :return: x_new: New point
        :return: mk: New value of M_k
        """
        aux_problem = _AuxiliaryProblem(x_old, self.grad_x, self.hess_x, mk, self.lambda_nplus, self.kappa_easy,
                                        self.submaxiter, self.aux_method, self.verbose)
        s, flag = aux_problem.solve()
        x_new = x_old + s
        cubic_approx = self._cubic_approx(self.f(x_new), s, mk)
        upper_approximation = (cubic_approx >= self.f(x_new))
        iter = 0
        while not upper_approximation and iter < self.submaxiter:
            # If mk is too small s.t. the cubic approximation is not upper, multiply by sqrt(2).
            mk *= 2
            aux_problem = _AuxiliaryProblem(x_old, self.grad_x, self.hess_x, mk, self.lambda_nplus, self.kappa_easy,
                                            self.submaxiter, self.aux_method, self.verbose)
            s, flag = aux_problem.solve()
            x_new = x_old + s
            cubic_approx = self._cubic_approx(self.f(x_new), s, mk)
            upper_approximation = (cubic_approx >= self.f(x_new))
            iter += 1
            if iter == self.submaxiter:
                raise RuntimeError('Could not find cubic upper approximation')
        mk = max(0.5 * mk, self.L0)
        return x_new, mk, flag