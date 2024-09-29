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

b = 0.1
A = 1

def f(t, y, b, A, w0, wd):
    x, v = y
    dx = v
    dv = -b*v - w0**2 * x - A*np.sin(wd*t)
    return np.array([dx, dv])



wd = np.linspace(0*w0, 2*w0, 100)
b = [0.05, 0.1, 0.2, 0.5, 1]
fig=Figure(x_label="Driving Frequency", y_label="Amplitude", x_min=wd[0], x_max=wd[(len(wd)-1)], y_min=0, y_max=18)
for bahh in b:
    ampl = []
    for ahh in wd:
        lfun = lambda t, y, : f(t, y, bahh, A, w0, ahh)
        
        results = integrate.solve_ivp(fun=lfun, t_span=(t0, tf), y0=y0, method="RK45", t_eval=t)
        t = results.t
        x = results.y[0]
        v = results.y[1]
        ampl.append((max(x)-min(x))/2)
    fig.plot(wd, ampl, ms=3, c=None, m='', ls='-', label=f"b = {bahh}")

fig.legend()




