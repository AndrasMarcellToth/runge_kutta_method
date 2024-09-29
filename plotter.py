import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import stdtrit
from scipy.stats import tstd
from tabulate import tabulate


class Figure:
    def __init__(self,
                 figsize=[5.5, 3.999],
                 dpi=600,  # Resolution of image.
                 plot_left=0.125,  # Position of plot in figure.
                 plot_bottom=0.125,  # (0 is all the way to the left/bottom, 1 is all the way to the right/top)
                 plot_right=0.9,
                 plot_top=0.9,
                 x_label='',  # Labels of x and y-axis.
                 y_label='',
                 x_min=None,
                 x_max=None,
                 y_min=None,
                 y_max=None,
                 font=None,  # Font must be passed as a font dict.
                 axes_width=0.5,  # Line width of axes
                 tick_label_size=7,  # Font size of tick mark labels.
                 grid=None,  # Turns on gridlines. Takes 'x', 'y' or 'both'.
                 grid_line_width=0.5,
                 grid_color='grey',
                 sci_lim_upper=10**3,
                 sci_lim_lower=10**-3,
                 box=True,
                 **kwargs
                 ):
        self.fig = plt.figure(dpi=dpi, figsize=figsize)
        self.fig.subplots_adjust(left=plot_left, bottom=plot_bottom, right=plot_right, top=plot_top)
        self.ax = self.fig.add_subplot(1, 1, 1, )
        if box == False:
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['top'].set_visible(False)
        else:
            self.ax.spines['right'].set_linewidth(axes_width)
            self.ax.spines['top'].set_linewidth(axes_width)
        self.ax.spines['left'].set_linewidth(axes_width)
        self.ax.spines['bottom'].set_linewidth(axes_width)
        self.ax.set_axisbelow(True)  # Move axes and grid to background.

        self.ax.minorticks_on()  # Add minor ticks and set tick parameters
        self.ax.tick_params(axis='both', which='major', direction='in', length=5, width=axes_width,
                            labelsize=tick_label_size)
        self.ax.tick_params(axis='both', which='minor', direction='in', length=3, width=axes_width)
        self.ax.ticklabel_format(axis='both', style='sci', useMathText=True)
        if x_min is not None and x_max is not None:
            self.ax.set_xlim([x_min, x_max])
        if y_min is not None and y_max is not None:
            self.ax.set_ylim([y_min, y_max])
        if font is None:  # Set font for axes labels.
            font = {'family': 'serif', 'size': 10, 'color': 'black'}
        self.ax.set_xlabel(x_label, fontdict=font)
        self.ax.set_ylabel(y_label, fontdict=font)

        if grid == 'both':  # Turn on grid lines.
            self.ax.grid(axis='x', lw=grid_line_width, c=grid_color)
            self.ax.grid(axis='y', lw=grid_line_width, c=grid_color)
        elif grid == 'x':
            self.ax.grid(axis='x', lw=grid_line_width, c=grid_color)
        elif grid == 'y':
            self.ax.grid(axis='y', lw=grid_line_width, c=grid_color)

    def plot(self,
                  x=None,
                  y=None,
                  xerr=None,
                  yerr=None,
                  error_line_width=1,
                  error_cap_size=1,
                  m='o',
                  ms=5,
                  ls='',
                  lw=2,
                  c='k',
                  **kwargs
                  ):
        self.ax.errorbar(x, y, xerr=xerr, yerr=yerr, ecolor=c, elinewidth=error_line_width,
                         capsize=error_cap_size, capthick=error_line_width, marker=m, ms=ms,
                         c=c, ls=ls, lw=lw, **kwargs)
    
    def line(self,
                  x=None,
                  y=None,
                  marker='',
                  marker_size=5,
                  line_style='-',
                  line_width=2,
                  color='k',
                  **kwargs
                  ):
        self.ax.plot(x, y, marker=marker, ms=marker_size, c=color, ls=line_style, lw=line_width, **kwargs)
    
    def fill(self,
              x=None,
              upper=None,
              lower=None,
              color='grey',
              **kwargs
              ):
        self.ax.fill_between(x, upper, lower, color=color, **kwargs)
    
    def contour(self,
                x=None,
                y=None,
                z=None,
                levels=None,
                zorder=0,
                label=False,
                **kwargs
                ):
        contour = self.ax.contour(x, y, z, levels=levels, zorder=zorder, **kwargs)
        if label == True:
            self.ax.clabel(contour, inline=True, fontsize=10, zorder=zorder, **kwargs)

        
    def contourf(self,
                x=None,
                y=None,
                z=None,
                levels=None,
                zorder=0,
                label=False,
                **kwargs
                ):
        contourf = self.ax.contourf(x, y, z, levels=levels, zorder=zorder, **kwargs)
        if label == True:
            self.ax.clabel(contourf, inline=True, fontsize=10, zorder=zorder, **kwargs)
            
    def quiver(self,
               x=None,
               y=None,
               u=None,
               v=None,
               scale=None,
               width=0.005,
               color='k',
               **kwargs
               ):
        self.ax.quiver(x, y, u, v, scale=scale, width=width, color=color, **kwargs)
    
    def text(self, 
             x=None, 
             y=None, 
             txt=None, 
             **kwargs
             ):
        self.ax.text(x, y, txt, **kwargs)
        
    def zmap(self, 
             func, 
             x, 
             y, 
             fill=False, 
             **kwargs
             ):
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)
        if fill == True:
            self.contourf(X, Y, Z, **kwargs)
        else:
            self.contour(X, Y, Z, **kwargs)
        
    def vfield(self, 
               func, 
               x, 
               y, 
               scale=10, 
               **kwargs
               ):
        X, Y = np.meshgrid(x, y)
        u, v = func(X, Y)
        self.quiver(X, Y, u, v, scale=scale, **kwargs)
        
    def line_int(self, 
                 v, 
                 r, 
                 dr, 
                 a, 
                 b, 
                 step=100, 
                 plot=True,
                 scale=10,
                 **kwargs
                 ):
        interval = np.linspace(a, b, step, endpoint=True)
        result = 0
        if plot == True: 
            self.line(*r(interval))
        for i in range(len(interval)-1):
            t = interval[i]
            dt = interval[i + 1] - t
            c = r(t)
            a = v(*c)
            b = dr(t)
            result += np.dot(a, b) * dt
            if plot == True: 
                self.quiver(*c, *b, scale=scale, **kwargs)
        print(f"line integral = {result}")
        return result
    
    def lsqr(self, 
             func, 
             x, 
             y, 
             initial, 
             sigma=None, 
             line=None, 
             res_plot=False, 
             mark=None
             ):
        if line is None:
            line = {"marker": '', "line_style": '-'}
        if mark is None:
            mark = {"color": "grey"}
        
        names = list(initial.keys())
        init = list(initial.values())
        
        param, covar = curve_fit(func, x, y, init, sigma)
        
        y_fit = func(x, *param)
        x_fit_plot = np.linspace(x[0], x[-1:], 1000, endpoint=True)
        y_fit_plot = func(x_fit_plot, *param)
        
        stdev = np.sqrt(np.diag(covar))
        dof = len(x) - len(names)
        residual = y - y_fit
        residual_sqr = np.sum(np.power(y - y_fit, 2))
        reduced_chi_sqr = residual_sqr / dof
        r_sqr = 1 - residual_sqr / (np.sum(np.power((y - np.mean(y)), 2)))
        rmse = tstd(residual)
        t95 = stdtrit(dof, 0.975)
        
        self.plot(x_fit_plot, y_fit_plot, **line)
        if res_plot == True:
            self.plot(x, residual, **mark)
        
        table1 = [['Param', 'Value', 'Standard dev', '95% Confidance', 'Percent E']]
        table2 = [['Param', 'Value'], ['Reduced chi^2', reduced_chi_sqr], ['R sqare', r_sqr], ['rms E', rmse], ['DoF', dof], ['t95', t95]]
        for i in range(len(names)):
            new_line = [names[i], param[i], stdev[i], t95*stdev[i], 100*t95*stdev[i]/param[i]]
            table1.append(new_line)
        print(tabulate(table1, headers="firstrow", tablefmt="fancy_grid"))
        print(tabulate(table2, headers="firstrow", tablefmt="fancy_grid"))
        
    
    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def save(file_name='Figure.svg'):
        plt.savefig(file_name)
    
    @staticmethod
    def legend():
        plt.legend()


