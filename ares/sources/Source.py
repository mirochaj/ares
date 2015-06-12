"""

Source.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Jul 22 16:28:08 2012

Description: Initialize a radiation source.

"""

import re, os
import numpy as np
from ..physics import Hydrogen
from ..physics import Cosmology
from scipy.integrate import quad
from ..util import ParameterFile
from ..util.Misc import sort, evolve
from ..physics.Constants import erg_per_ev, E_LL
from ..static.IntegralTables import IntegralTable
from ..static.InterpolationTables import LookupTable
from ..util.SetDefaultParameterValues import SourceParameters, \
    CosmologyParameters
from ..physics.CrossSections import PhotoIonizationCrossSection as sigma_E

try:
    import h5py
except ImportError:
    pass

np.seterr(all='ignore')   # exp overflow occurs when integrating BB
                          # will return 0 as it should for x large

cosmo_pars = CosmologyParameters()

class Source(object):
    def __init__(self, grid=None, logN=None, init_tabs=True):
        """ 
        Initialize a radiation source object. 
        
        ..note:: This is inherited by all other ares.sources classes.
    
        Parameters
        ----------
        grid: rt1d.static.Grid.Grid instance
        logN: column densities over which to tabulate integral quantities
        
        """    
        
        # Update cosmological parameters
        for par in cosmo_pars:
            if par in self.pf:
                continue
        
            self.pf[par] = cosmo_pars[par]
                
        # Modify parameter file if spectrum_file provided
        #self._load_spectrum()        
            
        # Correct emission limits if none were provided
        self.Emin = self.pf['source_Emin']
        self.Emax = self.pf['source_Emax']
        self.logEmin = np.log10(self.Emin)
        self.logEmax = np.log10(self.Emax)
                
        if self.pf['source_EminNorm'] == None:
            self.pf['source_EminNorm'] = self.pf['source_Emin']
        if self.pf['source_EmaxNorm'] == None:
            self.pf['source_EmaxNorm'] = self.pf['source_Emax']
            
        self.EminNorm = self.pf['source_EminNorm']
        self.EmaxNorm = self.pf['source_EmaxNorm']    
               
        # Number of frequencies
        #if self.discrete:
        #    self.E = np.array(self.pf['source_E'])
        #    self.LE = np.array(self.pf['source_LE'])
        #    self.Nfreq = len(self.E)
        #    
        #if self.src._name == 'DiffuseSource':
        #    self.ionization_rate = self.src.ionization_rate
        #    self.secondary_ionization_rate = self.src.secondary_ionization_rate
        #    self.heating_rate = self.src.heating_rate
        #        
        #self.Lbol = self.Lbol0 = self.BolometricLuminosity(0.0)

        # Create lookup tables for integral quantities
        if init_tabs and grid is not None:
            self._create_integral_table(logN=logN)
    
    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            if self.grid is None:
                self._cosm = Cosmology(
                    omega_m_0=self.pf['omega_m_0'], 
                    omega_l_0=self.pf['omega_l_0'], 
                    omega_b_0=self.pf['omega_b_0'],  
                    hubble_0=self.pf['hubble_0'],  
                    helium_by_number=self.pf['helium_by_number'], 
                    cmb_temp_0=self.pf['cmb_temp_0'], 
                    approx_highz=self.pf['approx_highz'], 
                    sigma_8=self.pf['sigma_8'], 
                    primordial_index=self.pf['primordial_index'])
            else:
                self._cosm = self.grid.cosm
        
        return self._cosm
    
    @property
    def multi_freq(self):
        if not hasattr(self, '_multi_freq'):
            self._multi_freq = self.discrete and not self.pf['source_multigroup']
            
        return self._multi_freq    
    
    @property        
    def multi_group(self):        
        if not hasattr(self, '_multi_group'):
            self._multi_group = self.discrete and self.pf['source_multigroup']
        
        return self._multi_group
            
    @property
    def ionizing(self):
        # See if source emits ionizing photons
        # Should also be function of absorbers
        if not hasattr(self, '_ionizing'):
            self._ionizing = self.pf['source_Emax'] > E_LL
        
        return self._ionizing
    
    @property
    def grid(self):
        if not hasattr(self, '_grid'):
            self._grid = None
        
        return self._grid
            
    @grid.setter
    def grid(self, value):
        self._grid = value
    
    @property 
    def discrete(self):
        if not hasattr(self, '_discrete'):
            self._discrete = (self.pf['source_E'] != None) #\
                  #or self.pf['optically_thin']
        
        return self._discrete
        
    @property
    def continuous(self):
        if not hasattr(self, '_continuous'):
            self._continuous = not self.discrete
            
        return self._continuous

    @property
    def hydr(self):
        if not hasattr(self, '_hydr'):
            self._hydr = None
                
        return self._hydr

    @hydr.setter
    def hydr(self, value):
        self._hydr = value

    @property
    def frec(self):
        """
        Compute average recycling fraction (i.e., spectrum-weighted frec).
        """    
        
        if self.hydr is None:
            return None
        
        n = np.arange(2, self.hydr.nmax)
        En = np.array(map(self.hydr.ELyn, n))
        In = np.array(map(self.Spectrum, En)) / En
        fr = np.array(map(self.hydr.frec, n))
        
        return np.sum(fr * In) / np.sum(In)

    @property
    def intrinsic_hardening(self):
        if not hasattr(self, '_intrinsic_hardening'): 
            if 'source_hardening' in self.pf:           
                self._intrinsic_hardening = \
                    self.pf['source_hardening'] == 'intrinsic'
            else:
                self._intrinsic_hardening = False
    
        return self._intrinsic_hardening    
        
    def _hardening_factor(self, E):
        return np.exp(-10.**self.logN \
            * (sigma_E(E, 0) + self.cosm.y * sigma_E(E, 1)))
    
    @property
    def logN(self):
        if not hasattr(self, '_logN'):
            if 'source_logN' in self.pf:
                self._logN = self.pf['source_logN']
            else:
                self._logN = -np.inf
                
        return self._logN
        
    @property
    def _normL(self):
        if not hasattr(self, '_normL_'):
            if self.intrinsic_hardening:
                self._normL_ = 1. / quad(self._Intensity,
                    self.pf['source_EminNorm'], self.pf['source_EmaxNorm'])[0]
            else:    
                integrand = lambda EE: self._Intensity(EE) / self._hardening_factor(EE)
                self._normL_ = 1. / quad(integrand,
                    self.pf['source_EminNorm'], self.pf['source_EmaxNorm'])[0]
                
        return self._normL_          
              
    #def _load_spectrum(self):
    #    """ Modify a few parameters if spectrum_file provided. """
    #    
    #    fn = self.pf['spectrum_file']
    #    
    #    if fn is None:
    #        return
    #        
    #    # Read spectrum - expect hdf5 with (at least) E, LE, and t datasets.    
    #    if re.search('.hdf5', fn):    
    #        f = h5py.File(fn)
    #        try:
    #            self.pf['tables_times'] = f['t'].value
    #        except:
    #            self.pf['tables_times'] = None
    #            self.pf['spectrum_evolving'] = False
    #                
    #        self.pf['spectrum_E'] = f['E'].value
    #        self.pf['spectrum_LE'] = f['LE'].value
    #        f.close()
    #        
    #        if len(self.pf['spectrum_LE'].shape) > 1 \
    #            and not self.pf['spectrum_evolving']:
    #            self.pf['spectrum_LE'] = self.pf['spectrum_LE'][0]
    #    else: 
    #        spec = readtab(fn)
    #        if len(spec) == 2:
    #            self.pf['spectrum_E'], self.pf['spectrum_LE'] = spec
    #        else:
    #            self.pf['spectrum_E'], self.pf['spectrum_LE'], \
    #                self.pf['spectrum_t'] = spec
                    
    @property
    def tables(self):
        if not hasattr(self, '_tables'):
            self._create_integral_table()
                    
        return self._tables            
    def _create_integral_table(self, logN=None):
        """
        Take tables and create interpolation functions.
        """
        
        if self.discrete:
            return
            
        if self._name == 'diffuse':
            return
        
        if self.pf['source_table'] is None:
            # Overide defaults if supplied - this is dangerous
            if logN is not None:
                self.pf.update({'tables_dlogN': [np.diff(tmp) for tmp in logN]})
                self.pf.update({'tables_logNmin': [np.min(tmp) for tmp in logN]})
                self.pf.update({'tables_logNmax': [np.max(tmp) for tmp in logN]})

            # Tabulate away!            
            self.tab = IntegralTable(self.pf, self, self.grid, logN)
            self.tabs = self.tab.TabulateRateIntegrals()
        else:
            self.tab = IntegralTable(self.pf, self, self.grid, logN)
            self.tabs = self.tab.load(self.pf['source_table'])
        
        self._setup_interp()
        
    def _setup_interp(self):            
        self._tables = {}
        for tab in self.tabs:
            self._tables[tab] = \
                LookupTable(self.pf, tab, self.tab.logN, self.tabs[tab], 
                    self.tab.logx, self.tab.t)                 
    
    @property
    def sigma(self):
        """
        Compute bound-free absorption cross-section for all frequencies.
        """    
        if not self.discrete:
            return None
        if not hasattr(self, '_sigma_all'):
            self._sigma_all = np.array(map(sigma_E, self.E))
        
        return self._sigma_all
        
    @property
    def Qdot(self):
        """
        Returns number of photons emitted (s^-1) at all frequencies.
        """    
        if not hasattr(self, '_Qdot_all'):
            self._Qdot_all = self.Lbol * self.LE / self.E / erg_per_ev
        
        return self._Qdot_all
        
    @property
    def hnu_bar(self):
        """
        Average ionizing (per absorber) photon energy in eV.
        """
        if not hasattr(self, '_hnu_bar_all'):
            self._hnu_bar_all = np.zeros_like(self.grid.zeros_absorbers)
            self._qdot_bar_all = np.zeros_like(self.grid.zeros_absorbers)
            for i, absorber in enumerate(self.grid.absorbers):
                self._hnu_bar_all[i], self._qdot_bar_all[i] = \
                    self._FrequencyAveragedBin(absorber=absorber)
            
        return self._hnu_bar_all
    
    def AveragePhotonEnergy(self, Emin, Emax):
        """
        Return average photon energy in supplied band.
        """
        
        integrand = lambda EE: self.Spectrum(EE) * EE
        
        return quad(integrand, Emin, Emax)[0]
        
    @property
    def qdot_bar(self):
        """
        Average ionizing photon luminosity (per absorber) in s^-1.
        """
        if not hasattr(self, '_qdot_bar_all'):
            hnu_bar = self.hnu_bar
            
        return self._qdot_bar_all   
    
    @property
    def sigma_bar(self):
        """
        Frequency averaged cross section (single bandpass).
        """
        if not hasattr(self, '_sigma_bar_all'):
            self._sigma_bar_all = np.zeros_like(self.grid.zeros_absorbers)
            for i, absorber in enumerate(self.grid.absorbers):
                integrand = lambda x: self.Spectrum(x) \
                    * self.grid.bf_cross_sections[absorber](x) / x
                    
                self._sigma_bar_all[i] = self.Lbol \
                    * quad(integrand, self.grid.ioniz_thresholds[absorber], 
                      self.Emax)[0] / self.qdot_bar[i] / erg_per_ev
            
        return self._sigma_bar_all
    
    @property
    def sigma_tilde(self):
        if not hasattr(self, '_sigma_tilde_all'):
            self._sigma_tilde_all = np.zeros_like(self.grid.zeros_absorbers)
            for i, absorber in enumerate(self.grid.absorbers):
                integrand = lambda x: self.Spectrum(x) \
                    * self.grid.bf_cross_sections[absorber](x)
                self._sigma_tilde_all[i] = quad(integrand, 
                    self.grid.ioniz_thresholds[absorber], self.Emax)[0] \
                    / self.fLbol_ionizing[i]
        
        return self._sigma_tilde_all
        
    @property
    def fLbol_ionizing(self):
        """
        Fraction of bolometric luminosity emitted above all ionization
        thresholds.
        """
        if not hasattr(self, '_fLbol_ioniz_all'):
            self._fLbol_ioniz_all = np.zeros_like(self.grid.zeros_absorbers)
            for i, absorber in enumerate(self.grid.absorbers):
                self._fLbol_ioniz_all[i] = quad(self.Spectrum, 
                    self.grid.ioniz_thresholds[absorber], self.Emax)[0]
                    
        return self._fLbol_ioniz_all
        
    @property
    def Gamma_bar(self):
        """
        Return ionization rate (as a function of radius) assuming optical 
        depth to cells and of cells is small.
        """
        if not hasattr(self, '_Gamma_bar_all'):
            self._Gamma_bar_all = \
                np.zeros([self.grid.dims, self.grid.N_absorbers])
            for i, absorber in enumerate(self.grid.absorbers):
                self._Gamma_bar_all[..., i] = self.Lbol * self.sigma_bar[i] \
                    * self.fLbol_ionizing[i] / 4. / np.pi / self.grid.r_mid**2 \
                    / self.hnu_bar[i] / erg_per_ev
                    
        return self._Gamma_bar_all
    
    @property
    def gamma_bar(self):
        """
        Return ionization rate (as a function of radius) assuming optical 
        depth to cells and of cells is small.
        """
        if not hasattr(self, '_gamma_bar_all'):
            self._gamma_bar_all = \
                np.zeros([self.grid.dims, self.grid.N_absorbers, 
                    self.grid.N_absorbers])
                    
            if not self.pf['secondary_ionization']:
                return self._gamma_bar_all
                    
            for i, absorber in enumerate(self.grid.absorbers):
                for j, otherabsorber in enumerate(self.grid.absorbers):
                    self._gamma_bar_all[..., i, j] = self.Gamma_bar[j] \
                        * (self.hnu_bar[j] * self.sigma_tilde[j] \
                        /  self.hnu_bar[i] / self.sigma_bar[j] \
                        - self.grid.ioniz_thresholds[otherabsorber] \
                        / self.grid.ioniz_thresholds[absorber])
                    
        return self._gamma_bar_all
    
    @property
    def Heat_bar(self):
        """
        Return ionization rate (as a function of radius) assuming optical 
        depth to cells and of cells is small.
        """
        if not hasattr(self, '_Heat_bar_all'):
            self._Heat_bar_all = \
                np.zeros([self.grid.dims, self.grid.N_absorbers])
            for i, absorber in enumerate(self.grid.absorbers):
                self._Heat_bar_all[..., i] = self.Gamma_bar[..., i] \
                    * erg_per_ev * (self.hnu_bar[i] * self.sigma_tilde[i] \
                    / self.sigma_bar[i] - self.grid.ioniz_thresholds[absorber])
                    
        return self._Heat_bar_all
                                
    def IonizingPhotonLuminosity(self, t=0, bin=None):
        """
        Return Qdot (photons / s) for this source at energy E.
        """
        
        if self.pf['source_type'] in [0, 1, 2]:
            return self.Qdot[bin]
        else:
            # Currently only BHs have a time-varying bolometric luminosity
            return self.BolometricLuminosity(t) * self.LE[bin] / self.E[bin] / erg_per_ev          
              
    #def _Intensity(self, E, i, Type, t=0, absorb=True):
    #    """
    #    Return quantity *proportional* to fraction of bolometric luminosity emitted
    #    at photon energy E.  Normalization handled separately.
    #    """
    #    
    #    Lnu = self.src._Intensity(E, i, Type, t=t)
    #    
    #    # Apply absorbing column
    #    if self.SpectrumPars['logN'][i] > 0 and absorb:
    #        return Lnu * np.exp(-10.**self.SpectrumPars['logN'][i] \
    #            * (sigma_E(E, 0) + y * sigma_E(E, 1)))   
    #    else:
    #        return Lnu     
    #            
    def Spectrum(self, E, t=0.0):
        r"""
        Return fraction of bolometric luminosity emitted at energy E.
        
        Elsewhere denoted as :math:`I_{\nu}`, normalized such that
        :math:`\int I_{\nu} d\nu = 1`
        
        Parameters
        ----------
        E: float
            Emission energy in eV
        t: float
            Time in seconds since source turned on.   
        i: int
            Index of component to include. If None, includes contribution
            from all components.
                    
        Returns
        -------
        Fraction of bolometric luminosity emitted at E in units of 
        eV\ :sup:`-1`\.

        """   
                
        return self._normL * self._Intensity(E, t=t)
        
    def BolometricLuminosity(self, t=0.0, M=None):
        """
        Returns the bolometric luminosity of a source in units of erg/s.  
        For accreting black holes, the bolometric luminosity will increase 
        with time, hence the optional 't' and 'M' arguments.
        """        
        
        if self._name == 'bh':
            return self.Luminosity(t, M)
        else:
            return self.Luminosity(t)
                
    def _FrequencyAveragedBin(self, absorber='h_1', Emin=None, Emax=None,
        energy_weighted=False):
        """
        Bolometric luminosity / number of ionizing photons in spectrum in bandpass
        spanning interval (Emin, Emax). Returns mean photon energy and number of 
        ionizing photons in band.
        """     
        
        if Emin is None:
            Emin = max(self.grid.ioniz_thresholds[absorber], 
                np.array(self.Emin)[self.ionizing])
        if Emax is None:
            Emax = self.Emax
            
        if energy_weighted:
            f = lambda x: x
        else:
            f = lambda x: 1.0    
            
        L = self.Lbol * quad(lambda x: self.Spectrum(x) * f(x), Emin, Emax)[0] 
        Q = self.Lbol * quad(lambda x: self.Spectrum(x) * f(x) / x, Emin, 
            Emax)[0] / erg_per_ev
                        
        return L / Q / erg_per_ev, Q            

    def dump(self, fn, E, clobber=False):
        """
        Write SED out to file.
        
        Parameters
        ----------
        fn : str
            Filename, suffix determines type. If 'hdf5' or 'h5' will write 
            to HDF5 file, otherwise, to ASCII.
        E : np.ndarray
            Array of photon energies at which to sample SED. Units = eV.
        
        """

        if os.path.exists(fn) and (clobber == False):
            raise OSError('%s exists!')

        if re.search('.hdf5', fn) or re.search('.h5', fn):
            out = 'hdf5'
        else:
            out = 'ascii'
            
        LE = map(self.Spectrum, E)    
            
        if out == 'hdf5':
            f = h5py.File(fn, 'w')    
            f.create_dataset('E', data=E)
            f.create_dataset('LE', data=LE)
            f.close()
        else:
            f = open(fn, 'w')
            print >> f, "# E     LE"
            for i, nrg in enumerate(E):
                print >> f, "%.8e %.8e" % (nrg, LE[i])
            f.close()    
    
        print "Wrote %s." % fn    
    
    def sed_name(self, i=0):
        """
        Return name of output file based on SED properties.
        """
            
        name = '%s_logM_%.2g_Gamma_%.3g_fsc_%.3g_logE_%.2g-%.2g' % \
            (self.SpectrumPars['type'][i], np.log10(self.src.M0), 
             self.src.spec_pars['alpha'][i], 
             self.src.spec_pars['fsc'][i], self.logEmin, self.logEmax)

        return name
        
        
                        
        
        