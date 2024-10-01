import numpy as np
from matplotlib import pyplot as plt
from plotter import Figure
from scipy import integrate


y0 = np.array([0, 1])
t0 = 0
tf = 30
n = 101
t = np.linspace(t0, tf, n)

def f(t, y):
    x, v = y
    return np.array([-v, x])


######################################################################################
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10, 5], dpi=600)


results = integrate.solve_ivp(fun=f, t_span=(t0, tf), y0=y0, method="RK45", t_eval=t)
t = results.t
x = results.y[0]
v = results.y[1]


ax1.plot(t, x, label='Position (m)')
ax1.plot(t, v, label='Velocity (m/s)')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Value")
ax1.tick_params(axis='both', which='both', direction='in') 
ax1.minorticks_on() 
ax1.legend()
ax1.set_ylim([-1.2, 1.5])
ax1.set_xlim([t0, tf])

ax2.plot(v, x, c='k')
ax2.set_xlabel("Velocity (m/s)")
ax2.set_ylabel("Position (m)")
ax2.tick_params(axis='both', which='both', direction='in') 
ax2.minorticks_on()  


plt.tight_layout()
plt.savefig("r3.svg")
#####################################################################################


####################################################################################

















