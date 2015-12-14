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
from .MultiPlot import MultiPanel
from ..physics.Constants import nu_0_mhz
from .TurningPoints import TurningPoints
from ..util.Math import central_difference
from .MultiPhaseMedium import MultiPhaseMedium

class Global21cm(MultiPhaseMedium):
    @property
    def dTbdz(self):
        if not hasattr(self, '_dTbdz'):
            self._z_p, self._dTbdz = \
                central_difference(self.data_asc['z'], self.data_asc['igm_dTb'])
        return self._dTbdz
    
    @property
    def dTbdnu(self):
        if not hasattr(self, '_dTbdnu'):
            self._nu_p, self._dTbdnu = \
                central_difference(self.data['nu'], self.data['dTb'])
        return self._dTbdnu
        
    @property
    def dTb2dz2(self):
        if not hasattr(self, '_dTb2dz2'):
            self._z_pp, self._dTb2dz2 = \
                central_difference(self.z_p, self.dTbdz)
        return self._dTbdz
    
    @property
    def dTb2dnu2(self):
        if not hasattr(self, '_dTbdnu'):
            _dTbdnu = self.dTbdnu
            _nu = self._nu_p
            self._nu_pp, self._dTbdnu = central_difference(_nu, _dTbdnu)
        return self._dTbdnu        
    
    @property
    def z_p(self):
        if not hasattr(self, '_z_p'):
            tmp = self.dTbdz
        return self._z_p
    
    @property
    def z_pp(self):
        if not hasattr(self, '_z_p'):
            tmp = self.dTb2dz2
        return self._z_pp    
    
    @property
    def nu_p(self):
        if not hasattr(self, '_nu_p'):
            tmp = self.dTbdnu
        return self._nu_p    
        
    @property
    def zB(self):
        if not hasattr(self, '_zB'):
            if 'B' in self.turning_points:
                self._zB = self.turning_points['B'][0]    
            else:
                self._zB = None
        return self._zB
    
    @property
    def zC(self):
        if not hasattr(self, '_zC'):
            if 'C' in self.turning_points:
                self._zC = self.turning_points['C'][0]
            else:
                self._zC = None
        return self._zC
    
    @property
    def zD(self):
        if not hasattr(self, '_zD'):
            if 'D' in self.turning_points:
                self._zD = self.turning_points['D'][0]
            else:
                self._zD = None
        return self._zD

    @property
    def nuB(self):
        if not hasattr(self, '_zB'):
            if 'B' in self.turning_points:
                self._nuB = nu_0_mhz / (1. + self.zB)
            else:
                self._nuB = none
        return self._nuB
    
    @property
    def nuC(self):
        if not hasattr(self, '_zC'):
            if 'C' in self.turning_points:
                self._nuC = nu_0_mhz / (1. + self.zC)
            else:
                self._nuC = None
        return self._nuC
    
    @property
    def nuD(self):
        if not hasattr(self, '_zD'):
            if 'D' in self.turning_points:
                self._nuD = nu_0_mhz / (1. + self.zD)
            else:
                self._nuD = None
        return self._nuD
    
    @property
    def znull(self):
        if not hasattr(self, '_znull'):
            self._znull = self.locate_null()
        return self._znull

    @property
    def TB(self):
        if not hasattr(self, '_TB'):
            if 'B' in self.turning_points:
                self._TB = self.turning_points['B'][1]
            else:
                self._TB = None
        return self._TB

    @property
    def TC(self):
        if not hasattr(self, '_TC'):
            if 'B' in self.turning_points:
                self._TC = self.turning_points['C'][1]    
            else:
                self._TC = None
        return self._TC

    @property
    def TD(self):
        if not hasattr(self, '_TD'):
            if 'D' in self.turning_points:
                self._TD = self.turning_points['D'][1]
            else:
                self._TD = None
        return self._TD
    
    @property
    def track(self):
        if not hasattr(self, '_track'):     
            if hasattr(self, 'pf'):
                self._track = TurningPoints(**self.pf)
            else:
                self._track = TurningPoints()
        return self._track
    
    @property
    def turning_points(self):
        if not hasattr(self, '_turning_points'):  

            # Otherwise, find them. Not the most efficient, but it gets the job done
            # Redshifts in descending order
            for i in range(len(self.data['z'])):
                if i < 10:
                    continue
            
                stop = self.track.is_stopping_point(self.data['z'][0:i], 
                    self.data['igm_dTb'][0:i])
            
            self._turning_points = self.track.turning_points
                        
        return self._turning_points
    
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
        
        if not gotax:
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
        
        if xscale == 'linear' and (not gotax):
            ax.set_xticks(xticks, minor=False)
            ax.set_xticks(xticks_minor, minor=True)
        
        if not gotax:    
            ax.set_yticks(yticks, minor=True)
        
        if gotax:
            return ax
            
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

    def Derivative(self, mp=None, fig=1, **kwargs):
        """
        Plot signal and its first derivative (nicely).

        Parameters
        ----------
        mp : MultiPlot.MultiPanel instance

        Returns
        -------
        MultiPlot.MultiPanel instance

        """

        if mp is None:
            gotax = False
            mp = MultiPanel(dims=(2, 1), panel_size=(1, 0.6), fig=fig)
        else:
            gotax = True

        ax = self.GlobalSignature(ax=mp.grid[0], z_ax=False, **kwargs)
        
        mp.grid[1].plot(self.nu_p, self.dTbdnu, **kwargs)
        
        if not gotax:
            mp.grid[1].set_ylabel(r'$\delta T_{\mathrm{b}}^{\prime} \ (\mathrm{mK} \ \mathrm{MHz}^{-1})$')
        
        mp.grid[1].set_xticks(mp.grid[0].get_xticks())    
        mp.grid[1].set_xticklabels([])    
        mp.grid[1].set_xlim(mp.grid[0].get_xlim())
        pl.draw()
        
        return mp
