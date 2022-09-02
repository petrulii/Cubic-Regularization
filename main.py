import numpy as np
import src.cubic_reg
import matplotlib.pyplot as plt

import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x)) 

def f(x,y,th):
    pro = sigmoid(np.dot(x,th))
    result = sum(-y * np.log(pro) - (1-y) * np.log(1-pro))   
    result = result/len(x) #len: number of feature rows
    return result

def gradient(x,y,th):
    xTrans = x.transpose()                                      
    sig = sigmoid(np.dot(x,th))                              
    grad = np.dot(xTrans, ( sig - y ))                          
    grad = grad / len(x) #len: number of feature rows  
    return grad

def hessian(x,y,th):
    xTrans = x.transpose()                                      
    sig = sigmoid(np.dot(x,th))                              
    result = (1.0/len(x) * np.dot(xTrans, x) * np.diag(sig) * np.diag(1 - sig) )   
    return result

n = 3
m = 4
x = np.random.seed(0)
x = np.random.rand(n,m)
y = np.random.rand(n,1)

f = lambda th: (sum(-y * np.log(sigmoid(np.dot(x,th))) - (1-y) * np.log(1-sigmoid(np.dot(x,th))))) / len(x)
grad = lambda th: (np.dot(x.T, ( sigmoid(np.dot(x,th)) - y ))) / len(x)
hess = lambda th: (1.0/len(x) * np.dot(x.T, x) * np.diag(sigmoid(np.dot(x,th))) * np.diag(1 - sigmoid(np.dot(x,th))))

th0 = np.random.rand(m,1)

cr = src.cubic_reg.CubicRegularization(th0, f=f, gradient=grad, hessian=hess, conv_tol=1e-4)
x_opt, intermediate_points, n_iter, flag = cr.cubic_reg()
print(" Solution : ", x_opt, "\n\n", "Intermediate points : ", intermediate_points, "\n\n","Nb. of iterations : ", n_iter)
