"""

Global21cm.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Oct  3 14:57:31 PDT 2015

Description: 

"""

import numpy as np
from ..util import labels
import matplotlib.pyplot as pl
from .MultiPhaseMedium import MultiPhaseMedium

class Global21cm(MultiPhaseMedium):
    def GlobalSignature(self, ax=None, fig=1, freq_ax=False, 
        time_ax=False, z_ax=True, mask=5, scatter=False, xaxis='nu', 
        ymin=None, ymax=50, zmax=None, xscale='linear', **kwargs):
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
        
        if xaxis == 'nu' and freq_ax:
            freq_ax = False
        if xaxis == 'z' and z_ax:
            z_ax = False
        
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
        
        if scatter is False:        
            ax.plot(self.data[xaxis], self.data['igm_dTb'], **kwargs)
        else:
            ax.scatter(self.data[xaxis][-1::-mask], self.data['igm_dTb'][-1::-mask], 
                **kwargs)        
        
        if gotax:
            pl.draw()
            return ax
        
        zmax = self.pf["first_light_redshift"]
        zmin = self.pf["final_redshift"] if self.pf["final_redshift"] >= 10 \
            else 5
        
        # x-ticks
        if xaxis == 'z' and hasattr(self, 'pf'):
            xticks = list(np.arange(zmin, zmax, zmin))
            xticks_minor = list(np.arange(zmin, zmax, 1))
        else:
            xticks = np.arange(20, 200, 20)
            xticks_minor = np.arange(10, 190, 20)

        if ymin is None:
            ymin = max(min(min(self.data['igm_dTb']), ax.get_ylim()[0]), -500)
    
            # Set lower y-limit by increments of 50 mK
            for val in [-50, -100, -150, -200, -250, -300, -350, -400, -450, -500]:
                if val <= ymin:
                    ymin = int(val)
                    break
    
        if ymax is None:
            ymax = max(max(self.data['igm_dTb']), ax.get_ylim()[1])
        
        ax.set_yticks(np.linspace(ymin, 50, int((50 - ymin) / 50. + 1)))
                
        # Minor y-ticks - 10 mK increments
        yticks = np.linspace(ymin, 50, int((50 - ymin) / 10. + 1))
        yticks = list(yticks)      
        
        # Remove major ticks from minor tick list
        for y in np.linspace(ymin, 50, int((50 - ymin) / 50. + 1)):
            if y in yticks:
                yticks.remove(y) 
        
        if xaxis == 'z' and hasattr(self, 'pf'):
            ax.set_xlim(5, self.pf["initial_redshift"])
        else:
            ax.set_xlim(10, 210)
            
        ax.set_ylim(ymin, ymax)    
        
        if xscale == 'linear':
            ax.set_xticks(xticks, minor=False)
            ax.set_xticks(xticks_minor, minor=True)
            
        ax.set_yticks(yticks, minor=True)
        
        if ax.get_xlabel() == '':  
            if xaxis == 'z':  
                ax.set_xlabel(labels['z'], fontsize='x-large')
            else:
                ax.set_xlabel(labels['nu'])
        
        if ax.get_ylabel() == '':    
            ax.set_ylabel(labels['igm_dTb'], 
                fontsize='x-large')    
        
        if 'label' in kwargs:
            if kwargs['label'] is not None:
                ax.legend(loc='best')
                
        # Twin axes along the top
        if freq_ax:
            twinax = self.add_frequency_axis(ax)        
        elif time_ax:
            twinax = self.add_time_axis(ax)
        elif z_ax:
            twinax = self.add_redshift_axis(ax)
        else:
            twinax = None
        
        self.twinax = twinax
        
        ax.ticklabel_format(style='plain', axis='both')
        ax.set_xscale(xscale)
                    
        pl.draw()
        
        return ax

    def Derivative(self, mp=None, **kwargs):
        """
        Plot signal and its first derivative (nicely).

        Parameters
        ----------
        mp : MultiPlot.MultiPanel instance

        Returns
        -------
        MultiPlot.MultiPanel instance

        """

        new = True
        if mp is None:
            mp = MultiPanel(dims=(2, 1), panel_size=(1, 0.6))
            ymin, ymax = zip(*[(999999, -999999)] * 3)
        else:
            new = False
            ymin = [0, 0, 0]
            ymax = [0, 0, 0]
            for i in xrange(2):
                ymin[i], ymax[i]= mp.grid[i].get_ylim()        
                
        mp.grid[0].plot(self.data_asc['z'], self.data_asc['igm_dTb'], **kwargs)        
        mp.grid[1].plot(self.z_p, self.dTbdnu, **kwargs)
        
        zf = int(np.min(self.data['z']))
        zfl = int(np.max(self.data['z']))
            
        # Set up axes
        xticks = np.linspace(zf, zfl, 1 + (zfl - zf))
        xticks = list(xticks)
        for x in np.arange(int(zf), int(zfl)):
            if x % 5 == 0:
                xticks.remove(x)
        
        yticks = np.linspace(-250, 50, 31)
        yticks = list(yticks)
        for y in np.linspace(-250, 50, 7):
            yticks.remove(y)
                        
        mp.grid[0].set_xlabel(labels['z'])
        mp.grid[0].set_ylabel(labels['igm_dTb'])
        mp.grid[1].set_ylabel(r'$d (\delta T_{\mathrm{b}}) / d\nu \ (\mathrm{mK/MHz})$')
        
        for i in xrange(2):                
            mp.grid[i].set_xlim(5, 35)   
            
        mp.grid[0].set_ylim(min(1.05 * self.data['igm_dTb'].min(), ymin[0]), max(1.2 * self.data['igm_dTb'].max(), ymax[0]))
        mp.grid[1].set_ylim(min(1.05 * self.dTbdnu.min(), ymin[1]), max(1.2 * self.dTbdnu[:-1].max(), ymax[1]))
                     
        mp.fix_ticks()
        
        return mp
