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


###################################################################################################
# fig=Figure(x_min=0, x_max=tf, y_min=0, y_max=0.22, x_label="Time (s)", y_label="Current (A)")

# t_exact = np.linspace(t0, tf, 100)
# I_exact = exact(t_exact)
# fig.plot(t_exact, I_exact, m='', ls='--', c='k', lw=0.7, label="Exact Solution", zorder=5)

# for i in [4, 6, 11, 21]:
#     t = np.linspace(t0, tf, i)
#     results = integrate.solve_ivp(fun=f, t_span=(t0, tf), y0=I0, method="RK45", t_eval=t)
#     t = results.t
#     I = results.y[0]
#     fig.plot(t, I, m='', ls='-', lw=0.7, c=None, label=f"{i-1} steps")

# fig.legend()
# fig.save("r2.svg")
###################################################################################################



##################################################################################################
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[10, 5], dpi=600)


# for i in [20, 50, 500]:
#     R = i
#     results = integrate.solve_ivp(fun=f, t_span=(t0, tf), y0=I0, method="RK45", t_eval=t)
#     t = results.t
#     I = results.y[0]
#     ax1.plot(t, I, label=f'R = {i}'+r'$\Omega$')
    
# R = 50

# for j in [20, 100, 500]:    
#     L = j
#     results = integrate.solve_ivp(fun=f, t_span=(t0, tf), y0=I0, method="RK45", t_eval=t)
#     t = results.t
#     I = results.y[0]
#     ax2.plot(t, I, label=f'L = {j} H')

# ax1.set_xlabel("Time (s)")
# ax1.set_ylabel("Curent (A)")
# ax1.tick_params(axis='both', which='both', direction='in') 
# ax1.minorticks_on() 
# ax1.legend()
# ax1.set_ylim([0, 0.6])
# ax1.set_xlim([0, 20])



# ax2.set_xlabel("Time (s)")
# ax2.set_ylabel("Current (A)")
# ax2.tick_params(axis='both', which='both', direction='in') 
# ax2.minorticks_on()  
# ax2.legend()
# ax2.set_ylim([0, 0.26])
# ax2.set_xlim([0, 20])
# plt.tight_layout()

# plt.savefig("r2_2.svg")
###############################################################################################
