import numpy as np
from matplotlib import pyplot as plt
from plotter import Figure
from scipy import integrate


y0 = np.array([0])
t0 = 0
tf = 20
n = 101
t = np.linspace(t0, tf, n)

a = 1
b = 1

def f(t, y):
    return -a*y**3 + b*np.sin(t)

fig=Figure(x_min=0, x_max=tf, x_label="t", y_label="y")
for ab in ([1, 1], [0.1, 0.1], [3, 1]):
    a = ab[0]
    b = ab[1]
    results = integrate.solve_ivp(fun=f, t_span=(t0, tf), y0=y0, method="RK45", t_eval=t)
    t = results.t
    y = results.y[0]
    fig.plot(t, y, ms=3, c=None, label=f"a = {a}, b = {b}")
    
fig.legend()