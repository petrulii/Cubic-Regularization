import numpy as np
import src.cubic_reg_LL as utils
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error as mse

def sigmoid(x):
    """ Sigmoid function. """
    return 1./(1+np.exp(-x))

def generate_data(nb_samples, nb_features, sigma=1, density=1):
    """ Generate normally distributed data set. """
    # Initialising the feature vector.
    theta = np.random.randn(nb_features, 1)
    # Generate random input.
    x = np.random.randn(nb_samples,nb_features)
    # Generate observations.
    y = sigmoid(np.matmul(x,theta))
    return x, y, theta

# To be able to replicate simulations.
np.random.seed(1)

fig_name = "_LL"
f = lambda th: sum(np.log(1 + np.exp(np.matmul(x,th))) - np.matmul(x,th)*y) * 1/n
grad = lambda th: np.matmul(x.T, (sigmoid(np.matmul(x,th)) - y)) * 1/n
hess = lambda th: np.dot(x.T, x) * np.diag(sigmoid(np.matmul(x,th))) * np.diag(1 - sigmoid(np.matmul(x,th))) * 1/n
#print("True theta:", theta, "\n\nStarting point:", th0,"\n")

nb_experiments = 3
n = 1000
M = np.array([5, 10, 50, 75])#, 100, 125, 150, 175])
nb_M = M.shape[0]
time_tr = np.zeros((nb_experiments,nb_M))
time_mn = np.zeros((nb_experiments,nb_M))
estim_error_tr = np.zeros((nb_experiments,nb_M))
estim_error_mn = np.zeros((nb_experiments,nb_M))
iters_tr = np.zeros((nb_experiments,nb_M))
iters_mn = np.zeros((nb_experiments,nb_M))

for i in range(nb_experiments):
    for j in range(nb_M):
        print("i,j:",i,j)
        m = 2#M[j]
        x, y, theta = generate_data(n, m)
        th0 = np.array([[1], [0]])#np.random.rand(m,1)

        start_time = time.time()
        cr = utils.CubicRegularization(th0, f=f, gradient=grad, hessian=hess, conv_tol=1e-4, L0=0.00001, aux_method="trust_region", verbose=0, conv_criterion='decrement')
        x_opt, intermediate_points, n_iter, flag = cr.cubic_reg()
        time_tr[i,j] = time.time() - start_time
        estim_error_tr[i,j] = mse(x_opt, theta)
        iters_tr[i,j] = n_iter

        start_time = time.time()
        cr = utils.CubicRegularization(th0, f=f, gradient=grad, hessian=hess, conv_tol=1e-4, L0=0.00001, aux_method="monotone_norm", verbose=0, conv_criterion='decrement')
        x_opt, intermediate_points, n_iter, flag = cr.cubic_reg()
        time_mn[i,j] = time.time() - start_time
        estim_error_mn[i,j] = mse(x_opt, theta)
        iters_mn[i,j] = n_iter

time_tr = np.average(time_tr, axis=0)
time_mn = np.average(time_mn, axis=0)
estim_error_tr = np.average(estim_error_tr, axis=0)
estim_error_mn = np.average(estim_error_mn, axis=0)
iters_tr = np.average(iters_tr, axis=0)
iters_mn = np.average(iters_mn, axis=0)

fig0, ax0 = plt.subplots()
ax0.plot(M, time_tr, label="Trust region")
ax0.plot(M, time_mn, label="Monotone norm")
ax0.set_xlabel('Number of features')
ax0.set_ylabel('Time')
ax0.legend(loc='best')
ax0.set_title("Time taken for different number of features m")
plt.savefig("figures/time"+fig_name+".png", format="png")

fig1, ax1 = plt.subplots()
ax1.plot(M, estim_error_tr, label="Trust region")
ax1.plot(M, estim_error_mn, label="Monotone norm")
ax1.set_xlabel('Number of features')
ax1.set_ylabel('Estimation error')
ax1.legend(loc='best')
ax1.set_title("Accuraccy for different number of features m")
plt.savefig("figures/accuracy"+fig_name+".png", format="png")

fig2, ax2 = plt.subplots()
ax2.plot(M, iters_tr, label="Trust region")
ax2.plot(M, iters_mn, label="Monotone norm")
ax2.set_xlabel('Number of features')
ax2.set_ylabel('Iterations')
ax2.legend(loc='best')
ax2.set_title("Iterations for different number of features m")
plt.savefig("figures/iterations"+fig_name+".png", format="png")

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