import numpy as np
import src.cubic_reg as cubic_reg
import matplotlib.pyplot as plt
from src.quadratic_obj import quadratic_obj
import time

def test_aux_methods(nb_experiments=10, high_dim=9):
    """
    Compare Trust region and Monotone norm methods for solving the auxiliary problem.
    :param nb_experiments: number of times both methods will be executed
    :param high_dim: highest dimension of the problem (will run all odd dimensions starting at 3)
    """
    # Specify number of experiments
    nb_experiments = nb_experiments
    # Initialize multiple dimensions for experiments
    N = np.arange(3, high_dim, 2)
    nb_N = N.shape[0]
    # For collecting experiment data and plotting
    fig_name = "aux_methods"
    time_tr = np.zeros((nb_N, nb_experiments))
    time_mn = np.zeros((nb_N, nb_experiments))
    glob_min_tr = np.zeros((nb_N, nb_experiments))
    glob_min_mn = np.zeros((nb_N, nb_experiments))
    iters_tr = np.zeros((nb_N, nb_experiments))
    iters_mn = np.zeros((nb_N, nb_experiments))

    # Solve for multiple increasing dimensions
    for i in range(nb_N):
        # Dimension of the problem
        n = N[i]
        # Parameters for the quadratic objective
        a = np.random.randint(-1,1,size=(n,n))
        A = (a + a.T)/2
        A[n-1, n-1] = 0
        c = np.random.uniform(-10,10)
        # Generate the objective function
        f = quadratic_obj(n, A, c, lambd=1)
        # Number of experiments per dimension
        for j in range(nb_experiments):
            start_time = time.time()
            # Initial point for cubic regularization
            x0 = np.random.randint(-10,10,size=(n,))
            cr = cubic_reg.CubicRegularization(x0, f=f, conv_tol=1e-10, L0=1.e-05, aux_method="trust_region", verbose=0, conv_criterion='gradient')
            x_opt, intermediate_points, n_iter, flag, intermediate_hess_cond = cr.cubic_reg()
            if np.isclose(f(x_opt),0):
                iters_tr[i,j] = n_iter
                time_tr[i,j] = time.time() - start_time
                glob_min_tr[i,j] = 1
            else:
                iters_tr[i,j] = -1
                time_tr[i,j] = -1
                glob_min_tr[i,j] = 0
            #print("Trust region\n", "Iterations:", n_iter, ", time:", time.time() - start_time, ", f_opt:", f(x_opt))
            #print("Argmin of f: ", x_opt, ".\n")

            start_time = time.time()
            cr = cubic_reg.CubicRegularization(x0, f=f, conv_tol=1e-10, L0=1.e-05, aux_method="monotone_norm", verbose=0, conv_criterion='gradient')
            x_opt, intermediate_points, n_iter, flag, intermediate_hess_cond = cr.cubic_reg()
            if np.isclose(f(x_opt),0):
                iters_mn[i,j] = n_iter
                time_mn[i,j] = time.time() - start_time
            else:
                iters_mn[i,j] = -1
                time_mn[i,j] = -1
                glob_min_mn[i,j] = 0
            glob_min_mn[i,j] = 1 if np.isclose(f(x_opt),0) else 0
            #print("Monotone norm\n", "Iterations:", n_iter, ", time:", time.time() - start_time, ", f_opt:", f(x_opt))
            #print("Argmin of f: ", x_opt, ".\n")
        time_tr[i][time_tr[i] == -1] = np.max(time_tr[i])
        time_mn[i][time_mn[i] == -1] = np.max(time_mn[i])
        iters_tr[i][iters_tr[i] == -1] = np.max(iters_tr[i])
        iters_mn[i][iters_mn[i] == -1] = np.max(iters_mn[i])

    time_tr = np.average(time_tr, axis=1)
    time_mn = np.average(time_mn, axis=1)
    glob_min_tr = np.average(glob_min_tr, axis=1)
    glob_min_mn = np.average(glob_min_mn, axis=1)
    iters_tr = np.average(iters_tr, axis=1)
    iters_mn = np.average(iters_mn, axis=1)

    plt.figure()
    plt.xticks(N)
    plt.scatter(N, time_tr, label="Trust region")
    plt.scatter(N, time_mn, label="Monotone norm", marker='*')
    plt.xlabel('dimension $k$')
    plt.ylabel('time (s)')
    plt.legend(loc='best')
    plt.title("Time taken with increasing $k$")
    plt.savefig("figures/time_"+fig_name+".png", format="png")
    plt.show()

    plt.figure()
    plt.xticks(N)
    plt.scatter(N, glob_min_tr, label="Trust region")
    plt.scatter(N, glob_min_mn, label="Monotone norm", marker='*')
    plt.xlabel('dimension $k$')
    plt.ylim(0,1)
    plt.ylabel('success score')
    plt.legend(loc='best')
    plt.title("How often the global minimum is found")
    plt.savefig("figures/value_"+fig_name+".png", format="png")
    plt.show()

    plt.figure()
    plt.xticks(N)
    plt.scatter(N, iters_tr, label="Trust region")
    plt.scatter(N, iters_mn, label="Monotone norm", marker='*')
    plt.xlabel('dimension $k$')
    plt.ylabel('iterations')
    plt.legend(loc='best')
    plt.title("Iterations with increasing $k$")
    plt.savefig("figures/iterations_"+fig_name+".png", format="png")
    plt.show()