"""

RadiationSource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Jul 22 16:28:08 2012

Description: Initialize a radiation source.

"""

import re, os
import numpy as np
from scipy.integrate import quad
from ..physics.Constants import *
from .SimpleSource import SimpleSource
from .StellarSource import StellarSource
from .DiffuseSource import DiffuseSource
from .BlackHoleSource import BlackHoleSource
from .ParameterizedSource import ParameterizedSource
from ..static.IntegralTables import IntegralTable
from ..static.InterpolationTables import LookupTable
from ..util import parse_kwargs, sort, evolve#, readtab, Gauss1D, boxcar
from ..physics.CrossSections import PhotoIonizationCrossSection as sigma_E

try:
    import h5py
except ImportError:
    pass

np.seterr(all='ignore')   # exp overflow occurs when integrating BB
                            # will return 0 as it should for x large

class RadiationSource(object):
    """ Class for creation and manipulation of radiation sources. """
    def __init__(self, grid=None, logN=None, init_tabs=True, **kwargs):
        """ 
        Initialize a radiation source object. 
    
        Parameters
        ----------
        grid: rt1d.static.Grid.Grid instance
        logN: column densities over which to tabulate integral quantities
        init_tabs: bool
            Tabulate integral quantities? Can wait until later.
    
        """    
        self.pf = parse_kwargs(**kwargs)
        self.grid = grid
                
        # Modify parameter file if spectrum_file provided
        #self._load_spectrum()        
            
        # Create Source/SpectrumPars attributes
        self.SourcePars = self.src_pars = sort(self.pf, prefix='source', make_list=False)        
        self.SpectrumPars = self.spec_pars = sort(self.pf, prefix='spectrum')
        
        # Correct emission limits if none were provided
        self.Emin = min(self.SpectrumPars['Emin'])
        self.Emax = max(self.SpectrumPars['Emax'])
        self.logEmin = np.log10(self.Emin)
        self.logEmax = np.log10(self.Emax)
                
        for i, comp in enumerate(self.SpectrumPars['type']):
            if self.SpectrumPars['EminNorm'][i] == None:
                self.SpectrumPars['EminNorm'][i] = self.SpectrumPars['Emin'][i]
            if self.SpectrumPars['EmaxNorm'][i] == None:
                self.SpectrumPars['EmaxNorm'][i] = self.SpectrumPars['Emax'][i]    
        
        self.EminNorm = self.SpectrumPars['EminNorm']
        self.EmaxNorm = self.SpectrumPars['EmaxNorm']    
        
        # Source-specific initialization
        if self.SourcePars['type'] == 'toy':
            self.src = SimpleSource(self.pf, self.SourcePars, 
                self.SpectrumPars)
        elif self.SourcePars['type'] == 'star':
            self.src = StellarSource(self.pf, self.SourcePars, 
                self.SpectrumPars)
        elif self.SourcePars['type'] == 'bh':
            self.src = BlackHoleSource(self.pf, self.SourcePars, 
                self.SpectrumPars)
        elif self.SourcePars['type'] == 'diffuse':
            self.src = DiffuseSource(self.pf, self.SourcePars, 
                self.SpectrumPars)
        elif self.SourcePars['type'] == 'parameterized':
            self.src = ParameterizedSource(self.pf, self.SourcePars, 
                self.SpectrumPars)        
        else:
            raise NotImplementedError('Unrecognized source_type')
                              
        # Number of spectral components
        self.N = len(self.SpectrumPars['type'])                  
                          
        self.discrete = (self.SpectrumPars['E'][0] != None) \
                      or self.pf['optically_thin']
        self.continuous = not self.discrete
        
        # Number of frequencies
        if self.discrete:
            self.E = np.array(self.SpectrumPars['E'])
            self.LE = np.array(self.SpectrumPars['LE'])
            self.Nfreq = len(self.E)
            
        if self.src._name == 'DiffuseSource':
            self.ionization_rate = self.src.ionization_rate
            self.secondary_ionization_rate = self.src.secondary_ionization_rate
            self.heating_rate = self.src.heating_rate
                
        # We don't allow multi-component discrete spectra...for now
        # would we ever want/need this? lines on top of continuous spectrum perhaps...
        self.multi_group = self.discrete and self.SpectrumPars['multigroup'][0] 
        self.multi_freq = self.discrete and not self.SpectrumPars['multigroup'][0] 

        # See if source emits ionizing photons (component by component)
        self.ionizing = np.array(self.SpectrumPars['Emax']) > E_LL
        # Should also be function of absorbers
        
        self.Lbol = self.Lbol0 = self.BolometricLuminosity(0.0)

        # Create lookup tables for integral quantities
        if init_tabs and grid is not None:
            self._create_integral_table(logN=logN) 
        
    @property
    def _normL(self):
        if not hasattr(self, '_normL_'):
            self._normL_ = self.src._NormalizeSpectrum()
        return self._normL_
              
    def _load_spectrum(self):
        """ Modify a few parameters if spectrum_file provided. """
        
        fn = self.pf['spectrum_file']
        
        if fn is None:
            return
            
        # Read spectrum - expect hdf5 with (at least) E, LE, and t datasets.    
        if re.search('.hdf5', fn):    
            f = h5py.File(fn)
            try:
                self.pf['tables_times'] = f['t'].value
            except:
                self.pf['tables_times'] = None
                self.pf['spectrum_evolving'] = False
                    
            self.pf['spectrum_E'] = f['E'].value
            self.pf['spectrum_LE'] = f['LE'].value
            f.close()
            
            if len(self.pf['spectrum_LE'].shape) > 1 \
                and not self.pf['spectrum_evolving']:
                self.pf['spectrum_LE'] = self.pf['spectrum_LE'][0]
        else: 
            spec = readtab(fn)
            if len(spec) == 2:
                self.pf['spectrum_E'], self.pf['spectrum_LE'] = spec
            else:
                self.pf['spectrum_E'], self.pf['spectrum_LE'], \
                    self.pf['spectrum_t'] = spec
                    
    def _create_integral_table(self, logN=None):
        """
        Take tables and create interpolation functions.
        """
        
        if self.discrete or self.SourcePars['type'] == 'diffuse':
            return
        
        if self.SourcePars['table'] is None:
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
            self.tabs = self.tab.load(self.SourcePars['table'])
        
        self._setup_interp()
        
    def _setup_interp(self):            
        self.tables = {}
        for tab in self.tabs:
            self.tables[tab] = \
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
              
    def _Intensity(self, E, i, Type, t=0, absorb=True):
        """
        Return quantity *proportional* to fraction of bolometric luminosity emitted
        at photon energy E.  Normalization handled separately.
        """
        
        Lnu = self.src._Intensity(E, i, Type, t=t)
        
        # Apply absorbing column
        if self.SpectrumPars['logN'][i] > 0 and absorb:
            return Lnu * np.exp(-10.**self.SpectrumPars['logN'][i] \
                * (sigma_E(E, 0) + y * sigma_E(E, 1)))   
        else:
            return Lnu     
            
    def SourceOn(self, t):
        return self.src.SourceOn(t)        
                
    def Spectrum(self, E, t=0.0, i=None):
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
               
        emission = 0.0
        
        # Loop over spectral components
        for j in xrange(self.N):
            if i is not None:
                if j != i:
                    continue
                    
            if not (self.SpectrumPars['Emin'][j] <= E <= \
                    self.SpectrumPars['Emax'][j]):
                continue        
            
            emission += self._normL[j] * \
                self.src._Intensity(E, i=j, t=t) / self.Lbol0
                        
        return emission
        
    def BolometricLuminosity(self, t=0.0, M=None):
        """
        Returns the bolometric luminosity of a source in units of erg/s.  
        For accreting black holes, the bolometric luminosity will increase 
        with time, hence the optional 't' and 'M' arguments.
        """        
        
        if self.src._name == 'BlackHoleSource':
            return self.src.Luminosity(t, M)
        else:
            return self.src.Luminosity(t)
                
    def _FrequencyAveragedBin(self, absorber='h_1', Emin=None, Emax=None,
        energy_weighted=False):
        """
        Bolometric luminosity / number of ionizing photons in spectrum in bandpass
        spanning interval (Emin, Emax). Returns mean photon energy and number of 
        ionizing photons in band.
        """     
        
        if Emin is None:
            Emin = max(self.grid.ioniz_thresholds[absorber], 
                np.array(self.SpectrumPars['Emin'])[self.ionizing])
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
        
        
                        
        
        