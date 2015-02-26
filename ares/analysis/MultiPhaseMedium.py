"""

MultiPhaseMedium.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Oct  4 12:59:46 2012

Description: 

"""

import numpy as np
from ..util import labels
import re, scipy, os, pickle
import matplotlib.pyplot as pl
from .MultiPlot import MultiPanel
from scipy.misc import derivative
from scipy.optimize import fsolve
from ..physics.Constants import *
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from .TurningPoints import TurningPoints
from ..physics import Cosmology, Hydrogen
from ..util.Math import central_difference
from ..util.SetDefaultParameterValues import *
from ..simulations import Global21cm as simG21
from ..simulations import MultiPhaseMedium as simMPM
from .DerivedQuantities import DerivedQuantities as DQ

try:
    import h5py
except ImportError:
    pass
    
turning_points = ['D', 'C', 'B', 'A']

class MultiPhaseMedium:
    def __init__(self, sim=None, prefix=None, **kwargs):
        """
        Initialize analysis object.
        
        Parameters
        ----------
        sim : instance
            Instance of glorb.Simulate.Simulation class
        history : dict, str
            Either a dictionary containing the 21-cm history or the name
            of the (HDF5) file containing the history.
        pf : dict, str
            Either a dictionary containing the parameters, or the name
            of the (binary/pickled) file containing the parameters.
                
        """
        
        if isinstance(sim, simG21) or isinstance(sim, simMPM):
            self.sim = sim
            self.pf = sim.pf
            self.data = sim.history.copy()
            
            try:
                self.cosm = sim.grid.cosm
            except AttributeError:
                self.cosm = Cosmology(omega_m_0=self.pf["omega_m_0"], 
                omega_l_0=self.pf["omega_l_0"], 
                omega_b_0=self.pf["omega_b_0"], 
                hubble_0=self.pf["hubble_0"], 
                helium_by_number=self.pf['helium_by_number'], 
                cmb_temp_0=self.pf["cmb_temp_0"], 
                approx_highz=self.pf["approx_highz"])
            
            try:
                self.hydr = sim.grid.hydr
            except AttributeError:
                self.hydr = Hydrogen(cosm=self.cosm)
        elif prefix is not None:
            pass
        #else:
        #    self.sim = None
        #    if type(history) is dict:
        #        self.data = history
        #    elif re.search('hdf5', history):
        #        f = h5py.File(history, 'r')
        #        self.data = {}
        #        for key in f.keys():
        #            self.data[key] = f[key].value[-1::-1]
        #        f.close()
        #    else:
        #        if re.search('pkl', history):
        #            f = open(history, 'rb')
        #            self.data = pickle.load(f)
        #            f.close()
        #        else:    
        #            f = open(history, 'r')
        #            cols = f.readline().split()[1:]
        #            data = np.loadtxt(f)
        #            
        #            self.data = {}
        #            for i, col in enumerate(cols):
        #                self.data[col] = data[:,i]
        #            f.close()
        #                      
        #    if pf is None:
        #        self.pf = SetAllDefaults()
        #    elif type(pf) is dict:
        #        self.pf = pf
        #    elif os.path.exists(pf):
        #        f = open(pf, 'rb')
        #        try:
        #            self.pf = pickle.load(f)
        #        except:
        #            self.pf = SetAllDefaults()
        #        f.close()
        #    
        #    self.cosm = Cosmology()
        #    self.hydr = Hydrogen()
        #
        self.kwargs = kwargs    
        
        # Derived quantities
        self._dq = DQ(self.data, self.pf)
        self.data = self._dq.data.copy()
        
        # For convenience - quantities in ascending order (in redshift)
        if self.data:
            
            data_reorder = {}
            for key in self.data.keys():
                data_reorder[key] = np.array(self.data[key])[-1::-1]   
                
            # Re-order
            if np.all(np.diff(self.data['z']) > 0):
                self.data_asc = self.data.copy()
                self.data = data_reorder
            else:
                self.data_asc = data_reorder
                
        self.interp = {}
        
        if hasattr(self, 'pf'):
            self._track = TurningPoints(**self.pf)
        else:
            self._track = TurningPoints()     
                
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
                central_difference(self.data_asc['nu'], self.data_asc['igm_dTb'])               
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
            self._nu_pp, self._dTbdnu = \
                central_difference(self.data_asc['nu'], self.data_asc['igm_dTb'])               
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
            tmp = self.dTbdz
        return self._nu_p    
        
    @property
    def zB(self):
        if not hasattr(self, '_zB'):
            self._zB = self.turning_points[0][3]    
        return self._zB
    
    @property
    def zC(self):
        if not hasattr(self, '_zC'):
            self._zC = self.turning_points[0][2]    
        return self._zC
    
    @property
    def zD(self):
        if not hasattr(self, '_zD'):
            self._zD = self.turning_points[0][1]    
        return self._zD
    
    @property
    def znull(self):
        if not hasattr(self, '_znull'):
            self._znull = self.locate_null()
        return self._znull
    
    @property
    def TB(self):
        if not hasattr(self, '_TB'):
            self._TB = self.turning_points[1][3]    
        return self._TB
    
    @property
    def TC(self):
        if not hasattr(self, '_TC'):
            self._TC = self.turning_points[1][2]    
        return self._TC
    
    @property
    def TD(self):
        if not hasattr(self, '_TD'):
            self._TD = self.turning_points[1][1]    
        return self._TD
    
    @property
    def Trei(self):
        if not hasattr(self, '_Trei'):
            self._Trei = self.turning_points[1][0]    
        return self._Trei
            
    def _initialize_interpolation(self, field):
        """ Set up scipy.interpolate.interp1d instances for all fields. """            
                
        self.interp[field] = scipy.interpolate.interp1d(self.data_asc['z'], 
            self.data_asc[field], kind='cubic')
                       
    def field(self, z, field='igm_dTb'):  
        """ Interpolate value of any field in self.data to redshift z. """ 
                
        if field not in self.interp:
            self._initialize_interpolation(field)
         
        return self.interp[field](z)
                
    def z_to_mhz(self, z):
        return nu_0_mhz / (1. + z)
                
    def zrei(self):
        """
        Find midpoint of reionization.
        """         
        
        zz = self.data['z'][self.data['z'] < 40]
        xz = self.data['cgm_h_2'][self.data['z'] < 40]
        
        return np.interp(0.5, xz, zz)
        
    def tau_CMB(self):
        """
        Compute CMB optical depth history.
        """
        
        QHII = self.data_asc['cgm_h_2'] 
        xHII = self.data_asc['igm_h_2'] 
        nH = self.cosm.nH(self.data_asc['z'])
        dldz = self.cosm.dldz(self.data_asc['z'])
        
        integrand = (QHII + (1. - QHII) * xHII) * nH
        
        if 'igm_he_1' in self.data_asc:
            QHeII = self.data_asc['cgm_h_2'] 
            xHeII = self.data_asc['igm_he_2'] 
            xHeIII = self.data_asc['igm_he_3'] 
            nHe = self.cosm.nHe(self.data_asc['z'])
            integrand += (QHeII + (1. - QHeII) * xHeII + 2. * xHeIII) \
                * nHe 
                
        integrand *= sigma_T * dldz
        
        tau = cumtrapz(integrand, self.data_asc['z'], initial=0)
            
        tau[self.data_asc['z'] > 100] = 0.0 
            
        self.data_asc['tau_CMB'] = tau
        self.data['tau_CMB'] = tau[-1::-1]
        
        # Make no arrays that go to z=0
        zlo, tlo = self.tau_post_EoR()
                
        tau_tot = tlo[-1] + tau
        
        tau_tot[self.data_asc['z'] > 100] = 0.0 
        
        self.data_asc['z_CMB'] = np.concatenate((zlo, self.data_asc['z']))
        self.data['z_CMB'] = self.data_asc['z_CMB'][-1::-1]
                
        self.data_asc['tau_CMB_tot'] = np.concatenate((tlo, tau_tot))
        self.data['tau_CMB_tot'] = tau_tot[-1::-1]
                
    @property
    def tau_e(self):
        if not hasattr(self, '_tau_e'):
            if 'tau_CMB_tot' not in self.data:
                self.tau_CMB()
                
            z50 = np.argmin(np.abs(self.data['z_CMB'] - 50))
            self._tau_e = self.data['tau_CMB_tot'][z50]
        
        return self._tau_e       
                
    @property            
    def turning_points(self):
        """
        Locate turning points.
        """
        
        # Use turning_points from ares.simulations.Global21cm if we've got 'em
        if isinstance(self.sim, simG21):
            if hasattr(self.sim, 'turning_points'):
                return self.sim.turning_points
        
        if hasattr(self, '_turning_points'):
            return self._turning_points
            
        # Otherwise, find them. Not the most efficient, but it gets the job done
        # Redshifts in descending order
        for i in range(len(self.data['z'])):
            if i < 10:
                continue
            
            stop = self._track.is_stopping_point(self.data['z'][0:i], 
                self.data['igm_dTb'][0:i])
                                
        self._turning_points = self._track.turning_points
        
        return self._turning_points
        
    def add_redshift_axis(self, ax):
        """
        Take plot with frequency on x-axis and add top axis with corresponding 
        redshift.
    
        Parameters
        ----------
        ax : matplotlib.axes.AxesSubplot instance
            Axis on which to add a redshift scale.
        """    
        
        fig = ax.xaxis.get_figure()
        
        for ax in fig.axes:
            if ax.get_xlabel() == '$z$':
                return
    
        z = np.arange(10, 110, 10)[-1::-1]
        z_minor= np.arange(15, 80, 5)[-1::-1]
        nu = nu_0_mhz / (1. + z)
        nu_minor = nu_0_mhz / (1. + z_minor)
    
        z_labels = map(str, z)
        
        # Add 25, 15 and 12, 8 to redshift labels
        z_labels.insert(-1, '15')
        z_labels.insert(-1, '12')
        z_labels.extend(['9', '8', '7', '6'])                
        #z_labels.insert(-5, '25')
        
        z = np.array(map(int, z_labels))
        
        nu = nu_0_mhz / (1. + z)
        nu_minor = nu_0_mhz / (1. + z_minor)
        
        ax_z = ax.twiny()
        ax_z.set_xlabel(labels['z'])        
        ax_z.set_xticks(nu)
        ax_z.set_xticks(nu_minor, minor=True)
            
        # A bit hack-y
        for i, label in enumerate(z_labels):
            if label in ['50', '60', '70']:
                z_labels[i] = ''
        
            if float(label) > 80:
                z_labels[i] = ''
    
        ax_z.set_xticklabels(z_labels)
        ax_z.set_xlim(ax.get_xlim())
        
        pl.draw()
    
        return ax_z    
        
    def add_frequency_axis(self, ax):
        """
        Take plot with redshift on x-axis and add top axis with corresponding 
        (observed) 21-cm frequency.
        
        Parameters
        ----------
        ax : matplotlib.axes.AxesSubplot instance
        """    
        
        nu = np.arange(20, 220, 20)[-1::-1]
        nu_minor = np.arange(30, 230, 20)[-1::-1]
        znu = nu_0_mhz / np.array(nu) - 1.
        znu_minor = nu_0_mhz / np.array(nu_minor) - 1.
        
        ax_freq = ax.twiny()
        ax_freq.set_xlabel(labels['nu'])        
        ax_freq.set_xticks(znu)
        ax_freq.set_xticks(znu_minor, minor=True)
        ax_freq.set_xlim(ax.get_xlim())
        
        freq_labels = map(str, nu)
        
        # A bit hack-y
        for i, label in enumerate(freq_labels):
            if label in ['140', '180']:
                freq_labels[i] = ''
                
            if float(label) >= 200:
                freq_labels[i] = ''

        ax_freq.set_xticklabels(freq_labels)
        
        pl.draw()
        
        return ax_freq
        
    def add_time_axis(self, ax):
        """
        Take plot with redshift on x-axis and add top axis with corresponding 
        time since Big Bang.
        
        This is crude at the moment -- should self-consistently solve for
        age of the universe.
        
        Parameters
        ----------
        ax : matplotlib.axes.AxesSubplot instance
        """
        
        age = 13.7 * 1e3 # Myr
        
        t = np.arange(100, 1e3, 100) # in Myr
        t_minor = np.arange(0, 1e3, 50)[1::2]
        
        zt = map(lambda tt: self.cosm.TimeToRedshiftConverter(0., tt * s_per_myr, 
            np.inf), t)
        zt_minor = map(lambda tt: self.cosm.TimeToRedshiftConverter(0., tt * s_per_myr, 
            np.inf), t_minor)
        
        ax_time = ax.twiny()
        ax_time.set_xlabel(labels['t_myr'])        
        ax_time.set_xticks(zt)
        ax_time.set_xticks(zt_minor, minor=True)
        ax_time.set_xlim(ax.get_xlim())
        
        # A bit hack-y
        time_labels = map(str, map(int, t))
        for i, label in enumerate(time_labels):
            tnow = float(label)
            if (tnow in [400, 600, 700]) or (tnow > 800):
                time_labels[i] = ''    
            
        ax_time.set_xticklabels(time_labels)
        
        pl.draw()
        
        return ax_time
        
    def reverse_x_axis(self, ax, twinax=None):
        """
        By default, we plot quantities vs. redshift, ascending from left to 
        right. This method will flip the x-axis, so that redshift is 
        decreasing from left to right.
        """
        
        ax.invert_xaxis()
        
        if twinax is None:
            print "If you have a twinx axis, be sure to re-run add_time_axis or add_frequency_axis!"
            print "OR: pass that axis object as a keyword argument to this function!"
        else:
            twinax.invert_xaxis()
        
        pl.draw()
        
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
                
    def TemperatureHistory(self, ax=None, fig=1, show_Tcmb=False,
        show_Ts=False, show_Tk=True, scatter=False, mask=5, 
        show_legend=False, **kwargs):
        """
        Plot kinetic, CMB, and spin temperatures vs. redshift. 
        
        Parameters
        ----------
        
        """
        
        hasax = True
        if ax is None:
            hasax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        
        if 'label' not in kwargs:
            labels = [r'$T_{\gamma}$', r'$T_K$', r'$T_S$']
        else:
            labels = [kwargs['label']]*3
            del kwargs['label']
                
        if not scatter:
            if show_Tcmb:
                ax.semilogy(self.data['z'], self.cosm.TCMB(self.data['z']), 
                    label=labels[0], **kwargs)
            
            if show_Tk:
                ax.semilogy(self.data['z'], self.data['igm_Tk'], 
                    label=labels[1], **kwargs)
            
            if show_Ts:
                ax.semilogy(self.data['z'], self.data['Ts'], 
                    label=labels[2], **kwargs)
        else:
            if show_Tcmb:
                ax.scatter(self.data['z'][-1::-mask], self.cosm.TCMB(self.data['z'][-1::-mask]), 
                    label=labels[0], **kwargs)
            
            ax.scatter(self.data['z'][-1::-mask], self.data['igm_Tk'][-1::-mask], 
                label=labels[1], **kwargs)
            
            if show_Ts:
                ax.semilogy(self.data['z'][-1::-mask], self.data['Ts'][-1::-mask], 
                    label=labels[2], **kwargs)            
                
        if not hasax:
            if hasattr(self, 'pf'):
                ax.set_xlim(int(self.pf["final_redshift"]), 
                    self.pf["initial_redshift"])
            ax.set_xlabel(r'$z$')
            
        if not ax.get_ylabel():    
            ax.set_ylabel(r'Temperature $(\mathrm{K})$') 
        
        if show_legend:
            ax.legend(loc='best')
                
        pl.draw()
        
        return ax
        
    def IonizationHistory(self, ax=None, zone=None, element='h', 
        fig=1, scatter=False, 
        mask=5, show_xi=True, show_xe=True, show_xibar=True, 
        show_legend=False, **kwargs):
        """
        Plot ionized fraction evolution. 
        
        Parameters
        ----------
        zone : str
            Either "cgm", "igm", or None. If None, will plot mean ionized
            fraction evolution
        
        """
        
        hasax = True
        if ax is None:
            hasax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        
        if element == 'h':
            xe = self.data['igm_%s_2' % element]
            xi = self.data['cgm_%s_2' % element]
            xavg = xi + (1. - xi) * xe

            to_plot = [xavg, xi, xe]
            show = [show_xibar, show_xi, show_xe]

        else:
            to_plot = [self.data['igm_he_%i' % sp] for sp in [2,3]]
            show = [True] * 2
               
        if 'label' not in kwargs:
            if element == 'h':
                labels = [r'$\bar{x}_i$', r'$x_i$', r'$x_e$']
            else:
                labels = [r'$x_{\mathrm{HeII}}$',
                          r'$x_{\mathrm{HeIII}}$']
        else:
            labels = [kwargs['label']]*len(to_plot)
            del kwargs['label']
            
        if 'ls' not in kwargs:
            ls = ['-', '--', ':']
        else:
            ls = [kwargs['ls']]*len(to_plot)
            del kwargs['ls']   
               
        if not scatter:
            
            for i, element in enumerate(to_plot):
                if not show[i]:
                    continue
                    
                ax.semilogy(self.data['z'], element, ls=ls[i], label=labels[i],
                    **kwargs)
        else:   
            
            for i, element in enumerate(to_plot):
                if not show[i]:
                    continue
                    
                ax.semilogy(self.data['z'][-1::-mask], element[-1::-mask], 
                    label=labels[i], **kwargs)

        if hasattr(self, 'pf'):
            ax.set_xlim(int(self.pf["final_redshift"]), 
                self.pf["initial_redshift"])
        ax.set_ylim(ax.get_ylim()[0], 1.5)

        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'Ionized Fraction')
        ax.set_yscale('log')

        if show_legend:
            ncol = 2 if element == 'he' else 1
            ax.legend(loc='best', ncol=ncol)
        
        pl.draw()
        
        return ax
        
    def IonizationRateHistory(self, fig=1, ax=None, species='h_1', show_legend=False,
        **kwargs):
        """
        Plot ionization rates as a function of redshift.
        """    
        
        hasax = True
        if ax is None:
            hasax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        
        n1 = map(self.cosm.nH, self.data['z']) if species == 'h_1' \
        else map(self.cosm.nHe, self.data['z'])
        
        n1 = np.array(n1) * np.array(self.data['igm_%s' % species])
        
        ratePrimary = np.array(self.data['igm_Gamma_%s' % species]) #* n1
        
        ax.semilogy(self.data['z'], ratePrimary, ls='-', **kwargs)
        
        rateSecondary = np.zeros_like(self.data['z'])
        
        for donor in ['h_1', 'he_1', 'he_2']: 
            
            field = 'igm_gamma_%s_%s' % (species, donor)
            
            if field not in self.data:
                continue
               
            n2 = map(self.cosm.nH, self.data['z']) if donor == 'h_1' \
            else map(self.cosm.nHe, self.data['z'])
        
            n2 = np.array(n2) * np.array(self.data['igm_%s' % donor])
        
            rateSecondary += \
                np.array(self.data['igm_gamma_%s_%s' % (species, donor)]) * n2 / n1
        
        ax.semilogy(self.data['z'], rateSecondary, ls='--')
        
        ax.set_xlim(self.pf['final_redshift'], self.pf['first_light_redshift'])
        
        pl.draw()
        
        return ax
        
        
    def OverplotDAREWindow(self, ax, color='r', alpha=0.5):        
        ax.fill_between(np.arange(11, 36), -150, 50, 
            facecolor = color, alpha = alpha)    
            
        pl.draw()        
        
    def OpticalDepthHistory(self, ax=None, fig=1, 
        scatter=False, show_xi=True, show_xe=True, show_xibar=True, 
        obs_mu=0.081, obs_sigma=0.012, show_obs=False, annotate_obs=False,
        **kwargs): 
        """
        Plot (cumulative) optical depth to CMB evolution. 
        
        Parameters
        ----------
        obs_mu : float
            Observationally constrained CMB optical depth. If supplied, will
            draw shaded region denoting this value +/- the error (obs_sigma,
            see below).
        obs_sigma : float
            1-sigma error on CMB optical depth.
        show_obs : bool
            Show shaded region for observational constraint?
        
        """
        
        hasax = True
        if ax is None:
            hasax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
                
        if 'tau_CMB' not in self.data:
            self.tau_CMB()

        ax.plot(self.data_asc['z_CMB'], self.data_asc['tau_CMB_tot'], **kwargs)

        ax.set_xlim(0, 20)

        ax.set_xlabel(labels['z'])
        ax.set_ylabel(r'$\tau_e$')

        #ax.plot([0, self.data['z'].min()], [tau.max() + tau_post]*2, **kwargs)
        #
        #if show_obs:
        #    if obs_mu is not None:
        #        ax.fill_between(ax.get_xlim(), [obs_mu-obs_sigma]*2, 
        #            [obs_mu+obs_sigma]*2, color='red', alpha=0.5)
        #    
        #    if annotate_obs:    
        #        #ax.annotate(r'$1-\sigma$ constraint', 
        #        #    [self.data['z'].min(), obs_mu], ha='left',
        #        #    va='center')  
        #        ax.annotate(r'plus post-EoR $\tau_e$', 
        #            [5, tau.max() + tau_post + 0.002], ha='left',
        #            va='bottom')
        #        ax.set_title('daeshed lines are upper limits')

        ax.ticklabel_format(style='plain', axis='both')

        pl.draw()

        return ax

    def RadiationBackground(self, z, ax=None, fig=1, band='lw', **kwargs):
        """
        Plot radiation background at supplied redshift and band.
        """
        
        if band != 'lw':
            raise NotImplemented('Only know LW background so far...')
            
        if not hasattr(self.sim, '_Jrb'):
            raise ValueError('This simulation didnt evolve JLw...')
            
        hasax = True
        if ax is None:
            hasax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)    

        for i, element in enumerate(self.sim._Jrb):

            if element is None:
                continue

            # Read data for this radiation background
            zlw, Elw, Jlw = element 

            # Determine redshift
            ilo = np.argmin(np.abs(zlw - z))
            if zlw[ilo] > z:
                ilo -= 1
            ihi = ilo + 1

            zlo, zhi = zlw[ilo], zlw[ihi]

            for j, band in enumerate(Elw):

                # Interpolate: flux is (Nbands x Nz x NE)
                Jlo = Jlw[j][zlo]
                Jhi = Jlw[j][zhi]

                J = np.zeros_like(Jlw[j][zlo])
                for k, nrg in enumerate(band):
                    J[k] = np.interp(z, [zlo, zhi], [Jlo[k], Jhi[k]])

                ax.plot(band, J, **kwargs)

                # Fill in sawtooth
                if j > 0:
                    ax.semilogy([band[0]]*2, [J[0] / 1e4, J[0]], **kwargs)
        
        pl.draw()
        
        return ax

    def tau_post_EoR(self):
        """
        Compute optical depth to electron scattering of the CMB.
        Equation 2 of Shull & Venkatesan 2008.
        
        Only includes optical depth to CMB contribution from a time after 
        the final redshift in the dataset. It is assumed that the universe
        is fully ionized at all times, and that helium reionization occurs
        at z = 3.
        
        Returns
        -------
        Post-EoR (as far as our data is concerned) optical depth.
        """    

        zmin = self.data['z'].min()
        ztmp = np.linspace(0, zmin, 1000)

        QHII = 1.0
        nH = self.cosm.nH(ztmp)
        dldz = self.cosm.dldz(ztmp)

        integrand = QHII * nH

        if 'igm_he_1' in self.data_asc:
            QHeII = 1.0
            xHeIII = 1.0
            nHe = self.cosm.nHe(ztmp)
            integrand += (QHeII + 2. * xHeIII) * nHe 

        integrand *= sigma_T * dldz

        tau = cumtrapz(integrand, ztmp, initial=0)
        
        return ztmp, tau
    
    def blob_analysis(self, field, z):    
        """
        Compute value of given field at given redshift.
        
        Parameters
        ----------
        fields : str
            Fields to extract
        z : list
            Redshift at which to compute value of input fields.
            'B', 'C', 'D', as well as numerical values are acceptable.
        
        Returns
        -------
        Value of field at input redshift, z.
        
        """

        # Convert turning points to actual redshifts
        if type(z) is str:
            z = self.turning_points[z][0]
            
        # Redshift x blobs
        if field in self.data_asc:
            interp = interp1d(self.data_asc['z'],
                self.data_asc[field])
        else:
            raise NotImplemented('help!')
            
        return float(interp(z))
        

        