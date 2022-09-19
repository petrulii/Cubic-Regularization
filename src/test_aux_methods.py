import numpy as np
import cubic_reg
import matplotlib.pyplot as plt
import time
#from scipy.optimize import minimize

def f(x):
    """
    A quadratic objective with quadratic constraints.
    """
    n = len(x)
    term1 = 0
    term2 = 0
    for i in range(1,n):
        term1 += 10*(x[i]-x[i-1]*x[0])**2
    for i in range(0,n):
        for j in range(0,n):
            term2 += (A[i,j]*x[i]*x[j]-c)
    return 1000*term1+term2**2

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


np.random.seed(0)

# Specify number of experiments
nb_experiments = 10
# Initialize multiple dimensions
N = np.arange(3, 9, 2)
nb_N = N.shape[0]

# For collecting experiment data and plotting
fig_name = "aux_methods"
time_tr = np.zeros((nb_experiments,nb_N))
time_mn = np.zeros((nb_experiments,nb_N))
estim_error_tr = np.zeros((nb_experiments,nb_N))
estim_error_mn = np.zeros((nb_experiments,nb_N))
iters_tr = np.zeros((nb_experiments,nb_N))
iters_mn = np.zeros((nb_experiments,nb_N))

for i in range(nb_experiments):
    for j in range(nb_N):
        n = N[j]
        # Initial point for cubic regularization
        x0 = np.random.randint(-10,10,size=(n,))
        # Hyper-parameters for the quadratic objective
        a = np.random.randint(-10,10,size=(n,n))
        A = (a + a.T)/2
        A[n-1, n-1] = 0
        c = np.random.uniform(-10,10)
        # Solution to the one-dimensional form of the qudratic objective
        # res = minimize(f_one_dimensional, x0[0], method='Nelder-Mead', tol=1e-6)

        start_time = time.time()
        cr = cubic_reg.CubicRegularization(x0, f=f, conv_tol=1e-8, L0=1.e-05, aux_method="trust_region", verbose=0, conv_criterion='gradient')
        x_opt, intermediate_points, n_iter, flag, intermediate_hess_cond = cr.cubic_reg()
        #print("\nTrust region\n", "Iterations:", n_iter, ", time:", time.time() - start_time, ", f_opt:", f(x_opt))
        #print("Argmin of f: ", x_opt)
        time_tr[i,j] = time.time() - start_time
        estim_error_tr[i,j] = f(x_opt)
        iters_tr[i,j] = n_iter

        start_time = time.time()
        cr = cubic_reg.CubicRegularization(x0, f=f, conv_tol=1e-8, L0=1.e-05, aux_method="monotone_norm", verbose=0, conv_criterion='gradient')
        x_opt, intermediate_points, n_iter, flag, intermediate_hess_cond = cr.cubic_reg()
        #print("\nMonotone norm\n", "Iterations:", n_iter, ", time:", time.time() - start_time, ", f_opt:", f(x_opt))
        #print("Argmin of f: ", x_opt)
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
ax0.plot(N, time_mn, label="Monotone norm")
ax0.set_xlabel('k')
ax0.set_ylabel('Time')
ax0.legend(loc='best')
ax0.set_title("Time taken for different dimension k")
plt.savefig("figures/time_"+fig_name+".png", format="png")

fig1, ax1 = plt.subplots()
ax1.plot(N, estim_error_tr, label="Trust region")
ax1.plot(N, estim_error_mn, label="Monotone norm")
ax1.set_xlabel('k')
ax1.set_ylabel('Estimation error')
ax1.legend(loc='best')
ax1.set_title("Function value at argmin for different dimension k")
plt.savefig("figures/value_"+fig_name+".png", format="png")

fig2, ax2 = plt.subplots()
ax2.plot(N, iters_tr, label="Trust region")
ax2.plot(N, iters_mn, label="Monotone norm")
ax2.set_xlabel('k')
ax2.set_ylabel('Iterations')
ax2.legend(loc='best')
ax2.set_title("Iterations for different dimension k")
plt.savefig("figures/iterations_"+fig_name+".png", format="png")