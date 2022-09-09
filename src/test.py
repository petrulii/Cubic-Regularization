import numpy as np
import matplotlib.pyplot as plt

m = 3
et = np.random.rand(m)
l = np.random.rand(m)
mu = 2

f = lambda x, et, l, mu: np.linalg.norm(et/(l+3*mu*x))-x
grad = lambda x, et, l, mu: np.sum((-3*mu*np.sqrt(et*et/((l+3*mu*x)*(l+3*mu*x))))/(l+3*mu*x))-1
f_first = lambda th, y, et, l, mu: f(th, et, l, mu) + np.dot(grad(th, et, l, mu), y-th)

th0 = np.random.rand(m)
th1 = th0+0.001

print(f(th1, et, l, mu))
print(f_first(th0, th1, et, l, mu))