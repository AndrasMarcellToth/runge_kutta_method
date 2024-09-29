import numpy as np
from matplotlib import pyplot as plt
from plotter import Figure
from scipy import integrate


y0 = np.array([0, 1])
t0 = 0
tf = 200
n = 1001
t = np.linspace(t0, tf, n)
w = 1
b = 0.1

def f(t, y, b, w):
    x, v = y
    dx = v
    dv = -b*v - w**2 * x
    return np.array([dx, dv])

lfun = lambda t, y, : f(t, y, b, w)

fig=Figure(x_label="x", y_label="v", figsize=[5, 5])

results = integrate.solve_ivp(fun=lfun, t_span=(t0, tf), y0=y0, method="RK45", t_eval=t)
t = results.t
x = results.y[0]
v = results.y[1]

fig.plot(v, x, ms=3, c='k', m='', ls='-')

