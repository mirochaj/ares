import numpy as np
from ..util import labels
import matplotlib.pyplot as pl
from .MultiPlot import MultiPanel
from .Global21cm import Global21cm
from scipy.interpolate import interp1d
from ..physics.Constants import nu_0_mhz
from matplotlib.ticker import ScalarFormatter 

# Distinguish between mean history and fluctuations?
class PowerSpectrum(Global21cm):
    
    
    #def IonizationHistory(self, ax=None, fig=1, **kwargs):
    #    if ax is None:
    #        gotax = False
    #        fig = pl.figure(fig)
    #        ax = fig.add_subplot(111)
    #    else:
    #        gotax = True
    #       
        
    
    def PowerSpectrum(self, z, field_1='x', field_2='x', ax=None, fig=1, 
        force_draw=False, dimensionless=False, **kwargs):
        """
        Plot differential brightness temperature vs. redshift (nicely).

        Parameters
        ----------
        ax : matplotlib.axes.AxesSubplot instance
            Axis on which to plot signal.
        fig : int
            Figure number.
        
        Returns
        -------
        matplotlib.axes.AxesSubplot instance.
        
        """
        
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
        
        iz = np.argmin(np.abs(z - self.redshifts))
        
        ps_s = 'ps_%s%s' % (field_1, field_2)
        if dimensionless:
            ps = self.history[iz][ps_s] * self.k**3 / 2. / np.pi**2
        else:
            ps = self.history[iz][ps_s]
        
        ax.loglog(self.k, ps, **kwargs)
        
        if gotax and (ax.get_xlabel().strip()) and (not force_draw):
            return ax
            
        if ax.get_xlabel() == '':  
            ax.set_xlabel(labels['k'], fontsize='x-large')
        
        if ax.get_ylabel() == '':  
            if dimensionless:  
                ax.set_ylabel(labels['dpow'], fontsize='x-large')    
            else:
                ax.set_ylabel(labels['pow'], fontsize='x-large')    
        
        if 'label' in kwargs:
            if kwargs['label'] is not None:
                ax.legend(loc='best')
                 
        pl.draw()
        
        return ax

    def CorrelationFunction(self, z, field_1='x', field_2='x', ax=None, fig=1, 
        force_draw=False, **kwargs):
        """
        Plot differential brightness temperature vs. redshift (nicely).
    
        Parameters
        ----------
        ax : matplotlib.axes.AxesSubplot instance
            Axis on which to plot signal.
        fig : int
            Figure number.
    
        Returns
        -------
        matplotlib.axes.AxesSubplot instance.
    
        """
    
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
    
        iz = np.argmin(np.abs(z - self.redshifts))
    
        cf_s = 'cf_%s%s' % (field_1, field_2)
        cf = self.history[iz][cf_s]
        
        d = 2. * np.pi / self.k
    
        ax.loglog(d, cf, **kwargs)
    
        if gotax and (ax.get_xlabel().strip()) and (not force_draw):
            return ax
    
        if ax.get_xlabel() == '':  
            ax.set_xlabel(r'$r \ [\mathrm{cMpc}]$', fontsize='x-large')
    
        if ax.get_ylabel() == '':    
            s1 = field_1
            s2 = field_2
            s = r'$\xi_{%s%s}$' % (s1, s2)
            ax.set_ylabel(s, fontsize='x-large')    
    
        if 'label' in kwargs:
            if kwargs['label'] is not None:
                ax.legend(loc='best')
    
        pl.draw()
    
        return ax
    
