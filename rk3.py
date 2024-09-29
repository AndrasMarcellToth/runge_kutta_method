import numpy as np
from matplotlib import pyplot as plt
from plotter import Figure
from scipy import integrate


y0 = np.array([0, 1])
t0 = 0
tf = 20
n = 101
t = np.linspace(t0, tf, n)

def f(t, y):
    x, v = y
    return np.array([-v, x])

fig=Figure(x_label="x", y_label="v", figsize=[5, 5])

results = integrate.solve_ivp(fun=f, t_span=(t0, tf), y0=y0, method="RK45", t_eval=t)
t = results.t
x = results.y[0]
v = results.y[1]

fig.plot(v, x, ms=3, c='k', m='', ls='-')