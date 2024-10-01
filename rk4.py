import numpy as np
from matplotlib import pyplot as plt
from plotter import Figure
from scipy import integrate


y0 = np.array([0, 1])
t0 = 0
tf = 5
n = 1001
t = np.linspace(t0, tf, n)
w = 1
b = 5

def f(t, y):
    x, v = y
    dx = v
    dv = -b*v - w**2 * x
    return np.array([dx, dv])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10, 5], dpi=600)

# Solving the ODE


results = integrate.solve_ivp(fun=f, t_span=(t0, tf), y0=y0, method="RK45", t_eval=t)
t = results.t
x = results.y[0]
v = results.y[1]
stop = np.sqrt(x[len(x)-1]**2 + v[len(x)-1]**2)

while stop > 0.0001:
    tf += 1
    t = np.linspace(t0, tf, n)
    results = integrate.solve_ivp(fun=f, t_span=(t0, tf), y0=y0, method="RK45", t_eval=t)
    t = results.t
    x = results.y[0]
    v = results.y[1]
    stop = np.sqrt(x[len(x)-1]**2 + v[len(x)-1]**2)

print(tf)

# Plotting on ax1
ax1.plot(t, x, label='x(t)')
ax1.plot(t, v, label='v(t)')
ax1.set_xlabel("Time (t)")
ax1.set_ylabel("Value")
ax1.tick_params(axis='both', which='both', direction='in')  # Tick marks inside
ax1.minorticks_on()  # Enable minor ticks
ax1.legend()

# Plotting on ax2
ax2.plot(v, x, c='k')
ax2.set_xlabel("Velocity (v)")
ax2.set_ylabel("Position (x)")
ax2.tick_params(axis='both', which='both', direction='in')  # Tick marks inside
ax2.minorticks_on()  # Enable minor ticks


# Adjust layout for better spacing
plt.tight_layout()
plt.savefig("r4_3.svg")

