import numpy as np
import cubic_reg
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error as mse
from scipy.optimize import minimize
from autograd import grad
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

np.random.seed(0)

# Hyper-parameter of the polynomial
n = 3
sample_lim = 10
a = np.random.uniform(-sample_lim,sample_lim,size=(n,n))
A = (a + a.T)/2
A[n-1, n-1] = 0
c = np.random.uniform(-sample_lim,sample_lim)
A = np.array([[-0.62169316,0.57449907,5.8095014],[0.57449907,7.49970533,2.57225283],[5.8095014,2.57225283,0.]])
c = -5.509167256191654

nb_minima = 1
minima = np.zeros(nb_minima)

for i in range(nb_minima):
    res = minimize(f_x0, -10, method='Nelder-Mead', tol=1e-6)
    x0 = np.random.uniform(-10,10,(n,))
    cr = cubic_reg.CubicRegularization(x0, f=f, gradient=grad(f), hessian=hess_f, conv_tol=1e-10, L0=1e-4, aux_method="monotone_norm", verbose=0, conv_criterion='gradient', maxiter=10000)
    x_opt, intermediate_points, n_iter, flag, intermediate_hess_cond = cr.cubic_reg()
    f_x_opt = f(x_opt)
    print("One dimensional problem sol.:", res.x, "f(x):", f_x0(res.x), "\niterations:", n_iter, ", argmin:", x_opt, "\nvalue:", f_x_opt, ", i:", i, ", termsof the objective:", f_terms(x_opt))
    plot_f(res.x, x_opt, f_x_opt)
    minima[i] = f(x_opt)

print("Local minima:", minima)
minima = np.around(minima, decimals=1)
print("\nNumber of local minima found:", len(np.unique(minima)))
print("Best local minimum:", np.min(minima))

"""
# Initialize multiple dimensions for experiments
nb_experiments = 10
N = np.arange(3, 11, 2)
nb_N = N.shape[0]

# For collecting experiment data and plotting
fig_name = "polynomial_tust_region"
time_tr = np.zeros((nb_experiments,nb_N))
time_mn = np.zeros((nb_experiments,nb_N))
estim_error_tr = np.zeros((nb_experiments,nb_N))
estim_error_mn = np.zeros((nb_experiments,nb_N))
iters_tr = np.zeros((nb_experiments,nb_N))
iters_mn = np.zeros((nb_experiments,nb_N))

for i in range(nb_experiments):
    for j in range(nb_N):
        n = N[j]
        print("Experiment: ",i, ", n: ",n)
        # Initial point for cubic regularization
        x0 = np.zeros(n)
        #x0 = np.random.randint(-10,10,size=(n,))
        # Hyper-parameter of the polynomial
        a = np.random.randint(-10,10,size=(n,n))
        A = (a + a.T)/2
        A[n-1, n-1] = 0
        print(A)
        c = 20

        start_time = time.time()
        cr = utils.CubicRegularization(x0, f=f, conv_tol=1e-4, L0=0.00001, aux_method="trust_region", verbose=1, conv_criterion='gradient')
        x_opt, intermediate_points, n_iter, flag = cr.cubic_reg()
        print("\nTrust region\n", "Iterations:", n_iter, ", time:", time.time() - start_time, ", f_opt:", f(x_opt))
        print("Argmin of f: ", x_opt) 
        x_pows = np.power(np.ones(n)*x_opt[0], np.arange(1,n+1))
        print("Power series: ", x_pows)
        time_tr[i,j] = time.time() - start_time
        estim_error_tr[i,j] = f(x_opt)
        iters_tr[i,j] = n_iter

        start_time = time.time()
        cr = utils.CubicRegularization(x0, f=f, conv_tol=1e-4, L0=0.00001, aux_method="monotone_norm", verbose=0, conv_criterion='gradient')
        x_opt, intermediate_points, n_iter, flag = cr.cubic_reg()
        print("\nMonotone norm\n", "Iterations:", n_iter, ", time:", time.time() - start_time, ", f_opt:", f(x_opt))
        print("Argmin of f: ", x_opt) 
        x_pows = np.power(np.ones(n)*x_opt[0], np.arange(1,n+1))
        print("Power series: ", x_pows)
        time_mn[i,j] = time.time() - start_time
        estim_error_mn[i,j] = f(x_opt)
        iters_mn[i,j] = n_iter


time_tr = np.average(time_tr, axis=0)
time_mn = np.average(time_mn, axis=0)
estim_error_tr = np.average(estim_error_tr, axis=0)
estim_error_mn = np.average(estim_error_mn, axis=0)
iters_tr = np.average(iters_tr, axis=0)
iters_mn = np.average(iters_mn, axis=0)

fig0, ax0 = plt.subplots()
ax0.plot(N, time_tr, label="Trust region")
#ax0.plot(N, time_mn, label="Monotone norm")
ax0.set_xlabel('k')
ax0.set_ylabel('Time')
ax0.legend(loc='best')
ax0.set_title("Time taken for different dimension k")
plt.savefig("figures/time_"+fig_name+".png", format="png")

fig1, ax1 = plt.subplots()
ax1.plot(N, estim_error_tr, label="Trust region")
#ax1.plot(N, estim_error_mn, label="Monotone norm")
ax1.set_xlabel('k')
ax1.set_ylabel('Estimation error')
ax1.legend(loc='best')
ax1.set_title("Power series estimation error for different dimension k")
plt.savefig("figures/error_"+fig_name+".png", format="png")

fig2, ax2 = plt.subplots()
ax2.plot(N, iters_tr, label="Trust region")
#ax2.plot(N, iters_mn, label="Monotone norm")
ax2.set_xlabel('k')
ax2.set_ylabel('Iterations')
ax2.legend(loc='best')
ax2.set_title("Iterations for different dimension k")
plt.savefig("figures/iterations_"+fig_name+".png", format="png")
"""

"""
f_first = lambda th, y: f(th) + np.dot(grad(th), y-th)
f_second = lambda th, y: f(th) + np.dot(grad(th), y-th) + 0.5*np.dot(hess(th), (y-th)**2)
f_cubic = lambda th, y, mu: f(th) + np.dot(grad(th), y-th) + 0.5*np.dot(hess(th), (y-th)**2) + mu/6*(np.linalg.norm(y-th)**3)

dumm = np.ones(theta.shape)

if m==1:
    dumm = np.ones((1, 1))

    T = np.arange(-10, 10, 0.1)
    f_t = np.array([f(t*dumm) for t in T])
    f_t_first = np.array([f_first(dumm, t*dumm) for t in T])
    f_t_first = f_t_first.reshape((f_t_first.shape[0], 1))
    f_t_second = np.array([f_second(dumm, t*dumm) for t in T])
    f_t_second = f_t_second.reshape((f_t_second.shape[0], 1))
    f_t_cubic_1 = np.array([f_cubic(dumm, t*dumm, 1) for t in T])
    f_t_cubic_1 = f_t_cubic_1.reshape((f_t_cubic_1.shape[0], 1))
    f_t_cubic_2 = np.array([f_cubic(dumm, t*dumm, 10) for t in T])
    f_t_cubic_2 = f_t_cubic_2.reshape((f_t_cubic_2.shape[0], 1))
    f_t_cubic_3 = np.array([f_cubic(dumm, t*dumm, 100) for t in T])
    f_t_cubic_3 = f_t_cubic_3.reshape((f_t_cubic_3.shape[0], 1))
    f_t_cubic_4 = np.array([f_cubic(dumm, t*dumm, 1000) for t in T])
    f_t_cubic_4 = f_t_cubic_4.reshape((f_t_cubic_4.shape[0], 1))

    plt.figure()
    plt.axis([-10, 10, 0, 1])
    plt.plot(T, f_t_cubic_1, 'g', T, f_t_cubic_2, 'r', T, f_t_cubic_3, 'y', T, f_t_cubic_4, 'm', T, f_t, 'b')#, T, f_t_first, 'y', T, f_t_second, 'g')
    plt.show()
"""

""" n = len(x)
    term1 = 0
    term2 = 0
    for i in range(1,n):
        term1 += (x[i]-x[i-1]*x[0])
    for i in range(0,n):
        for j in range(0,n):
            term2 += (A[i,j]*x[i]*x[j]-c)
    for i in range(1,n):
        term1 += (x[i]-x[i-1]*x[0])
    grad1 = np.zeros(n)
    grad2 = np.zeros(n)
    for i in range(1,n):
        grad1[i] += (1-x[0])
    grad1 *= 2*term1
    for i in range(0,n):
        for j in range(0,n):
            if j!=i:
                grad2[i] += A[i,j]*x[j] + A[j,i]*x[j]
        grad2[i] += 2*A[i,i]*x[i]
    grad2 *= 2*term2
    return grad1+grad2
"""