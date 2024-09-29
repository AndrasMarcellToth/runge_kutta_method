import numpy as np
from matplotlib import pyplot as plt
from plotter import Figure
from scipy import integrate


I0 = np.array([0])
t0 = 0
tf = 20
n = 101
t = np.linspace(t0, tf, n)

V = 10
L = 100
R = 50

def f(t, I):
    return V/L - (R/L)*I

def exact(t):
    return V/R * (1 - np.exp(-(R*t)/L))


fig=Figure(x_min=0, x_max=tf, x_label="t", y_label="I")

results = integrate.solve_ivp(fun=f, t_span=(t0, tf), y0=I0, method="RK45", t_eval=t)
t = results.t
I = results.y[0]
I_exact = exact(t)
I_error = I_exact - I
fig.plot(t, I, m='', ls='-', c='k', label="RK45 Approximation")
fig.plot(t, I_exact, m='', ls='--', c='r', label="Exact Solution")
fig.legend()