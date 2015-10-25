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
    
    
class DummyDQ(object):
    """
    A wrapper around DerivedQuantities.
    """
    def __init__(self, pf={}):
        self._data = {}
        self._dq = DQ(ModelSet=None, pf=pf)
        
    def add_data(self, data):
        self._dq._add_data(data)
        for key in data:
            self._data[key] = data[key]
            
    def keys(self):
        return self._data.keys()    
            
    def __iter__(self):
        for key in self._data.keys():
            yield key        
            
    def __contains__(self, name):
        return name in self._data.keys()
            
    def __getitem__(self, name):
        if name in self._data:
            return self._data[name]
            
        self._data[name] = self._dq[name]
        return self._data[name]    
            
    def __setitem__(self, name, value):
        self._data[name] = value
    
turning_points = ['D', 'C', 'B', 'A']

class MultiPhaseMedium:
    def __init__(self, data=None, **kwargs):
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
                
        if isinstance(data, simG21) or isinstance(data, simMPM):
            self.sim = data
            self.pf = self.sim.pf
            history = self.sim.history

            try:
                self.cosm = self.sim.grid.cosm
            except AttributeError:
                self.cosm = Cosmology(omega_m_0=self.pf["omega_m_0"], 
                omega_l_0=self.pf["omega_l_0"], 
                omega_b_0=self.pf["omega_b_0"], 
                hubble_0=self.pf["hubble_0"], 
                helium_by_number=self.pf['helium_by_number'], 
                cmb_temp_0=self.pf["cmb_temp_0"], 
                approx_highz=self.pf["approx_highz"])
            
            try:
                self.hydr = self.sim.grid.hydr
            except AttributeError:
                self.hydr = Hydrogen(cosm=self.cosm)
        
        elif type(data) == dict:
            history = data.copy()
            
        # Read output of a simulation from disk
        elif type(data) is str:
            
            try:
                f = open('%s.history.pkl' % data, 'rb')
                history = pickle.load(f)
                f.close()
                
                f = open('%s.parameters.pkl' % data, 'rb')
                self.pf = pickle.load(f)
                f.close()
                
                self.cosm = Cosmology(**self.pf)
                self.hydr = Hydrogen(**self.pf)
                
            except IOError: 
                if re.search('pkl', data):
                    f = open(data, 'rb')
                    history = pickle.load(f)
                    f.close()
                else:
                    f = open(data, 'r')
                    cols = f.readline().split()[1:]
                    data = np.loadtxt(f)
                    
                    history = {}
                    for i, col in enumerate(cols):
                        history[col] = data[:,i]
                    f.close()
                    
        # If missing parameter file
        if not hasattr(self, 'pf'):  
            #print "ares.analysis.MPM: No parameter file found...setting all to default values."      
            self.pf = SetAllDefaults()
            self.cosm = Cosmology()
            self.hydr = Hydrogen()
            
        self.data = DummyDQ(pf=self.pf)

        # Add history to data
        self.data.add_data(history)

        if not hasattr(self, 'data'):
            raise ValueError('Must supply simulation instance, dict, or file prefix!')

        self.kwargs = kwargs    

        # Add frequencies
        if 'z' in self.data:
            self.data['nu'] = nu_0_mhz / (1. + self.data['z'])

        # For backward compatibility
        if 'dTb' in self.data:
            if 'igm_dTb' not in self.data:
                self.data['igm_dTb'] = self.data['dTb']
        elif 'igm_dTb' in self.data:
            if 'dTb' not in self.data:
                self.data['dTb'] = self.data['igm_dTb']

        # For convenience - quantities in ascending order (in redshift)
        data_reorder = {}
        for key in self.data.keys():
            data_reorder[key] = np.array(self.data[key])[-1::-1]
        
        # Re-order
        if np.all(np.diff(self.data['z']) > 0):
            self.data_asc = DummyDQ(pf=self.pf)
            self.data_asc.add_data(self.data)
            self.data = DummyDQ(pf=self.pf)
            self.data.add_data(data_reorder)
        else:
            self.data_asc = DummyDQ(pf=self.pf)
            self.data_asc.add_data(data_reorder)

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
        
    def excluded_regions(self, ax):
        """
        Overplot
        """
        
        ax.get_xlim()
        
        # Adiabatic cooling limit
        sim = simG21(tanh_model=True, tanh_T0=0.0, 
            tanh_x0=0.0, tanh_J0=1e5, tanh_Jz0=120, initial_redshift=120)
        
        ax.fill_between(sim.history['nu'], sim.history['igm_dTb'], 
            -500 * np.ones_like(sim.history['nu']), hatch='x', color='none', 
            edgecolor='gray')

        # Saturated
        sim = simG21(tanh_model=True, tanh_x0=0.0,    
            tanh_J0=1e5, tanh_Jz0=120, tanh_Tz0=120, tanh_T0=1e4, 
            initial_redshift=120)

        ax.fill_between(sim.history['nu'], sim.history['igm_dTb'], 
            80 * np.ones_like(sim.history['nu']), hatch='x', color='none', 
            edgecolor='gray')
        
        ax.set_xlim()
        
        pl.draw()
        
    def tau_CMB(self, include_He=True, z_HeII_EoR=3.):
        """
        Compute CMB optical depth history.
        """
        
        QHII = self.data_asc['cgm_h_2'] 
        if 'igm_h_2' in self.data_asc:
            xHII = self.data_asc['igm_h_2'] 
        else:
            xHII = 0.0
            
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
        elif include_He:
            integrand *= (1. + self.cosm.y)
                    
        integrand *= sigma_T * dldz
                
        tau = cumtrapz(integrand, self.data_asc['z'], initial=0)
            
        tau[self.data_asc['z'] > 100] = 0.0 
            
        self.data_asc['tau_CMB'] = tau
        self.data['tau_CMB'] = tau[-1::-1]
        
        if self.data_asc['z'][0] < z_HeII_EoR:
            raise ValueError('Simulation ran past assumed HeII EoR! See z_HeII_EoR parameter.')
        
        # Make no arrays that go to z=0
        zlo, tlo = self.tau_post_EoR(include_He=include_He, 
            z_HeII_EoR=z_HeII_EoR)
                
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
        
        #for ax in fig.axes:
        #    if ax.get_xlabel() == '$z$':
        #        return
        
        z = np.arange(10, 110, 10)[-1::-1]
        z_minor= np.arange(15, 80, 5)[-1::-1]
        nu = nu_0_mhz / (1. + z)
        nu_minor = nu_0_mhz / (1. + z_minor)
    
        z_labels = map(str, z)
        
        # Add 25, 15 and 12, 8 to redshift labels
        z_labels.insert(-1, '15')
        z_labels.insert(-1, '12')
        z_labels.extend(['8', '7', '6', '5'])
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
            if label in ['40','50', '60', '70']:
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
                ax.semilogy(self.data['z'], self.data['igm_Ts'], 
                    label=labels[2], **kwargs)
        else:
            if show_Tcmb:
                ax.scatter(self.data['z'][-1::-mask], self.cosm.TCMB(self.data['z'][-1::-mask]), 
                    label=labels[0], **kwargs)
            
            ax.scatter(self.data['z'][-1::-mask], self.data['igm_Tk'][-1::-mask], 
                label=labels[1], **kwargs)
            
            if show_Ts:
                ax.semilogy(self.data['z'][-1::-mask], self.data['igm_Ts'][-1::-mask], 
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
        mask=5, show_xi=True, show_xe=True, show_xibar=False, 
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
            if zone is None:
                if self.pf['include_igm']:
                    xe = self.data['igm_%s_2' % element]
                else:
                    xe = np.zeros_like(self.data['z'])
                if self.pf['include_cgm']:
                    xi = self.data['cgm_%s_2' % element]
                else:
                    xi = np.zeros_like(self.data['z'])
                    
                xavg = xi + (1. - xi) * xe
                
                to_plot = [xavg, xi, xe]
                show = [show_xibar, show_xi, show_xe]
            else:
                to_plot = [self.data['%s_%s_2' % (zone, element)]]
                show = [True] * 2           

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
        obs_mu=0.066, obs_sigma=0.012, show_obs=False, annotate_obs=False,
        include_He=True, z_HeII_EoR=3., **kwargs): 
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
            self.tau_CMB(include_He=include_He, z_HeII_EoR=z_HeII_EoR)

        ax.plot(self.data_asc['z_CMB'], self.data_asc['tau_CMB_tot'], **kwargs)

        ax.set_xlim(0, 20)

        ax.set_xlabel(labels['z'])
        ax.set_ylabel(r'$\tau_e$')

        #ax.plot([0, self.data['z'].min()], [tau.max() + tau_post]*2, **kwargs)
        
        if show_obs:
            if obs_mu is not None:
                ax.fill_between(ax.get_xlim(), [obs_mu-obs_sigma]*2, 
                    [obs_mu+obs_sigma]*2, color='red', alpha=0.5)
            
            if annotate_obs:    
                #ax.annotate(r'$1-\sigma$ constraint', 
                #    [self.data['z'].min(), obs_mu], ha='left',
                #    va='center')  
                ax.annotate(r'plus post-EoR $\tau_e$', 
                    [5, tau.max() + tau_post + 0.002], ha='left',
                    va='bottom')
                ax.set_title('daeshed lines are upper limits')

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

    def tau_post_EoR(self, include_He=True, z_HeII_EoR=3.):
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
        elif include_He:
            integrand *= (1. + self.cosm.y)

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
        

        