import numpy as np
from matplotlib import pyplot as plt
from plotter import Figure
from scipy import integrate


y0 = np.array([0, 1])
t0 = 0
tf = 100
n = 1001
t = np.linspace(t0, tf, n)
w0 = 1
wd = 0.5
b = 0.1
A = 1

def f(t, y, b, A, w0, wd):
    x, v = y
    dx = v
    dv = -b*v - w0**2 * x - A*np.sin(wd*t)
    return np.array([dx, dv])

lfun = lambda t, y, : f(t, y, b, A, w0, wd)

fig=Figure(x_label="t", y_label="x")

results = integrate.solve_ivp(fun=lfun, t_span=(t0, tf), y0=y0, method="RK45", t_eval=t)
t = results.t
x = results.y[0]
v = results.y[1]

fig.plot(t, x, ms=3, c='k', m='', ls='-')


