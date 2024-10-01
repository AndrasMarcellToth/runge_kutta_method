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


ampl = []
wd = np.linspace(0, 2*w0, 100)

for i in wd:
    lfun = lambda t, y, : f(t, y, b, A, w0, i)
    
    results = integrate.solve_ivp(fun=lfun, t_span=(t0, tf), y0=y0, method="RK45", t_eval=t)
    t = results.t
    x = results.y[0]
    v = results.y[1]
    x_trunc = x[int(len(x)*0.8):]
    ampl.append((max(x_trunc)-min(x_trunc))/2)

fig=Figure(x_label=r"$\omega_0$ (rad/s)", y_label="Amplitude (m)")
fig.plot(wd, ampl, lw=0.7, c='k', m='', ls='-')


