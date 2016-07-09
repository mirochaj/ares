import numpy as np
from ..util import labels
import matplotlib.pyplot as pl
from .MultiPlot import MultiPanel
from .Global21cm import Global21cm
from scipy.interpolate import interp1d
from ..physics.Constants import nu_0_mhz
from matplotlib.ticker import ScalarFormatter 


class PowerSpectrum(Global21cm):
    
    # Distinguish between mean history and fluctuations?
    
    def PowerSpectrum(self, z, ax=None, fig=1, **kwargs):
        """
        Plot differential brightness temperature vs. redshift (nicely).

        Parameters
        ----------
        ax : matplotlib.axes.AxesSubplot instance
            Axis on which to plot signal.
        fig : int
            Figure number.
        freq_ax : bool
            Add top axis denoting corresponding (observed) 21-cm frequency?
        time_ax : bool
            Add top axis denoting corresponding time since Big Bang?
        z_ax : bool
            Add top axis denoting corresponding redshift? Only applicable
            if xaxis='nu' (see below).
        scatter : bool
            Plot signal as scatter-plot?
        mask : int
            If scatter==True, this defines the sampling "rate" of the data,
            i.e., only every mask'th element is included in the plot.
        xaxis : str
            Determines whether x-axis is redshift or frequency. 
            Options: 'z' or 'nu'
        
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
        
        data = np.sum(self.snapshots[z], axis=0) * self.k**3 / 2. / np.pi**2
        
        ax.loglog(self.k, data, **kwargs)
        
        if gotax and (ax.get_xlabel().strip()) and (not force_draw):
            return ax
            
        if ax.get_xlabel() == '':  
            ax.set_xlabel(labels['k'], fontsize='x-large')
        
        if ax.get_ylabel() == '':    
            ax.set_ylabel(labels['dpow'], fontsize='x-large')    
        
        if 'label' in kwargs:
            if kwargs['label'] is not None:
                ax.legend(loc='best')
                 
        pl.draw()
        
        return ax
