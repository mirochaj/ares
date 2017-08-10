import numpy as np
from ..util import labels
import matplotlib.pyplot as pl
from .MultiPlot import MultiPanel
from .Global21cm import Global21cm
from scipy.interpolate import interp1d
from ..physics.Constants import nu_0_mhz
from matplotlib.ticker import ScalarFormatter 
from ..analysis.BlobFactory import BlobFactory
from .MultiPhaseMedium import MultiPhaseMedium

# Distinguish between mean history and fluctuations?
class PowerSpectrum(MultiPhaseMedium,BlobFactory):
    def __init__(self, data=None, suffix='fluctuations', **kwargs):
        """
        Initialize analysis object.
        
        Parameters
        ----------
        data : dict, str
            Either a dictionary containing the entire history or the prefix
            of the files containing the history/parameters.

        """
                
        MultiPhaseMedium.__init__(self, data=data, suffix=suffix, **kwargs)
        
    @property
    def redshifts(self):
        if not hasattr(self, '_redshifts'):
            self._redshifts = self.history['z']
        return self._redshifts
        
    @property
    def gs(self):
        if not hasattr(self, '_gs'):
            if hasattr(self, 'prefix'):
                self._gs = Global21cm(data=self.prefix)
            elif 'dTb' in self.history:
                hist = {'z': self.history['z'], 'dTb': self.history['dTb']}
                self._gs = Global21cm(data=hist)
            else:
                raise AttributeError('Cannot initialize Global21cm instance!')
                
        return self._gs
        
    def PowerSpectrum(self, z, field='21', ax=None, fig=1,
        force_draw=False, dimensionless=True, take_sqrt=False, **kwargs):
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
        
        k = self.history['k']
        
        ps_s = 'ps_%s' % field
        if dimensionless and 'ps_21_dl' in self.history:
            ps = self.history['ps_21_dl'][iz]
            if take_sqrt:
                ps = np.sqrt(ps)
                
        elif dimensionless:
            norm = self.history['dTb0'][iz]**2
            ps = norm * self.history[ps_s][iz] * k**3 / 2. / np.pi**2
            
            if take_sqrt:
                ps = np.sqrt(ps)
        else:
            ps = self.history[ps_s][iz]
        
        ax.loglog(k, ps, **kwargs)
        
        if gotax and (ax.get_xlabel().strip()) and (not force_draw):
            return ax
            
        if ax.get_xlabel() == '':  
            ax.set_xlabel(labels['k'], fontsize='x-large')
        
        if ax.get_ylabel() == '':  
            if dimensionless and 'ps_21_dl' in self.history:
                ps = self.history['ps_21_dl']
            elif dimensionless:
                if take_sqrt:
                    ax.set_ylabel(r'$\Delta_{21}(k)$', fontsize='x-large')    
                else: 
                    ax.set_ylabel(labels['dpow'], fontsize='x-large')    
            else:
                ax.set_ylabel(labels['pow'], fontsize='x-large')    
                         
        ax.set_xlim(1e-2, 10)
        ax.set_ylim(1e-3, 1e4)
                 
        pl.draw()
        
        return ax

    def CorrelationFunction(self, z, field_1='x', field_2='x', ax=None, fig=1, 
        force_draw=False, **kwargs):
        """
        Plot correlation function of input fields.
    
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
        cf = self.history[cf_s][iz]
        
        k = self.history['k']
        
        dr = 1. / k
        
        ax.loglog(dr, cf, **kwargs)
    
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
    
    def BubbleSizeDistribution(self, z, ax=None, fig=1, force_draw=False, 
        by_volume=False, **kwargs):
        """
        Plot bubble size distribution.
    
        Parameters
        ----------
        ax : matplotlib.axes.AxesSubplot instance
            Axis on which to plot signal.
        fig : int
            Figure number.
        by_volume : bool
            If True, uses bubble volume rather than radius.

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

        R = self.history['R_b'][iz]
        Mb = self.history['M_b'][iz]
        bsd = self.history['bsd'][iz]
    
        rho0 = self.cosm.mean_density0
        
        R = ((Mb / rho0) * 0.75 / np.pi)**(1./3.)
        dvdr = 4. * np.pi * R**2        
        dmdr = rho0 * dvdr
        dmdlnr = dmdr * R
        dndlnR = bsd * dmdlnr

        V = 4. * np.pi * R**3 / 3.

        Q = self.history['Qi'][iz]
                
        if by_volume:
            dndV = bsd * dmdr / dvdr
            ax.loglog(V, V * dndV, **kwargs)
            ax.set_xlabel(r'$V \ [\mathrm{Mpc}^{3}]$')
            ax.set_ylabel(r'$V \ dn/dV$')
            ax.set_xlim(1, 1e6)
            ax.set_ylim(1e-8, 1e-2)
        else:
            ax.semilogx(R, V * dndlnR / Q, **kwargs)
            ax.set_xlabel(r'$R \ [\mathrm{Mpc}]$')
            ax.set_ylabel(r'$Q^{-1} V \ dn/dlnR$')
            ax.set_yscale('linear')
            ax.set_ylim(0, 1)

        pl.draw()

        return ax

    def BubbleFillingFactor(self, ax=None, fig=1, force_draw=False, 
        **kwargs):  
        """
        Plot the fraction of the volume composed of ionized bubbles.
        """
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
        
        Qall = []
        for i, z in enumerate(self.redshifts):
            Qall.append(self.history['Qi'][i])

        ax.plot(self.redshifts, Qall, **kwargs)
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$Q_{\mathrm{HII}}$')
        ax.set_ylim(0, 1)
        
        pl.draw()
        
        return ax

    def RedshiftEvolution(self, field='21', k=0.2, ax=None, fig=1, 
        dimensionless=True, show_gs=False, mp_kwargs={}, **kwargs):
        """
        Plot the fraction of the volume composed of ionized bubbles.
        """
        
        if ax is None:
            if show_gs:
                if mp_kwargs == {}:
                    mp_kwargs = {'padding': (0, 0.1)}
                    
                mp = MultiPanel(dims=(2, 1), **mp_kwargs)
            else:    
                gotax = False
                fig = pl.figure(fig)
                ax = fig.add_subplot(111)
        else:
            gotax = True
            
            if show_gs:
                print ax
                assert isinstance(ax, MultiPanel)

            mp = ax
                    
        p = []
        for i, z in enumerate(self.redshifts):
            if dimensionless and 'ps_21_dl' in self.history:
                pow_z = self.history['ps_21_dl'][i]
            else:
                pow_z = self.history['ps_%s' % field][i]
            
            p.append(np.interp(k, self.history['k'], pow_z))
            
        p = np.array(p)
        
        if dimensionless and 'ps_21_dl' in self.history:
            ps = p
        elif dimensionless:
            norm = self.history['dTb0']**2
            ps = norm * p * k**3 / 2. / np.pi**2
        else:
            ps = p
        
        if show_gs or isinstance(ax, MultiPanel):
            ax1 = mp.grid[0]
        else:
            ax1 = ax
        
        ax1.plot(self.redshifts, ps, **kwargs)
        ax1.set_xlim(min(self.redshifts), max(self.redshifts))
        ax1.set_yscale('log')
        ax1.set_xlim(6, 30)
        ax1.set_ylim(1e-2, 1e4)
        ax1.set_xlabel(r'$z$')
        
        if dimensionless:
            ax1.set_ylabel(labels['dpow'])
        else:
            ax1.set_ylabel(labels['pow'])
        
        if show_gs:
            self.gs.GlobalSignature(ax=mp.grid[1], xaxis='z', **kwargs)
            mp.grid[1].set_xlim(6, 30)
            mp.grid[1].set_xticklabels([])
            mp.grid[1].set_xlabel('')
        
        pl.draw()
        
        if show_gs or isinstance(ax, MultiPanel):
            return mp
        else:
            return ax1
            
        
        
