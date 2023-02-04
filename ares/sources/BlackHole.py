"""

BlackHole.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Jul  8 09:56:38 MDT 2013

Description: 

"""

import numpy as np
from .Star import _Planck
from .Source import Source
from types import FunctionType
from scipy.integrate import quad
from ..util.Math import interp1d
from ..util.ReadData import read_lit
from ..util.SetDefaultParameterValues import BlackHoleParameters
from ..physics.CrossSections import PhotoIonizationCrossSection as sigma_E
from ..physics.Constants import s_per_myr, G, g_per_msun, c, t_edd, m_p, \
    sigma_T, sigma_SB
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

sptypes = ['pl', 'mcd', 'simpl']

class BlackHole(Source):
    def __init__(self, **kwargs):
        """ 
        Initialize a black hole object. 
    
        Parameters
        ----------
        pf: dict
            Full parameter file.
        src_pars: dict
            Contains source-specific parameters.
        spec_pars: dict
            Contains spectrum-specific parameters.
    
        """
        
        #self.pf = BlackHoleParameters()
        #self.pf.update(kwargs)    
        Source.__init__(self, **kwargs)
                
        self._name = 'bh'
        
        self.M0 = self.pf['source_mass']
        self.epsilon = self.pf['source_eta']
        
        # Duty cycle parameters
        if self.pf['source_fduty'] is None:
            self.fduty = 1.0
        else:
            self.fduty = self.pf['source_fduty'] 
            assert type(self.fduty) in [int, float, np.float64]    
            
        self.variable = self.fduty < 1
        #if self.src_pars['fduty'] == 1:
        #    self.variable = self.tau < self.pf['stop_time']
        
        self.toff = self.tau * (self.fduty**-1. - 1.)
        
        # Disk properties
        self.last_renormalized = 0.0
        self.r_in = self._DiskInnermostRadius(self.M0)
        self.r_out = self.pf['source_rmax'] * self._GravitationalRadius(self.M0)
        self.T_in = self._DiskInnermostTemperature(self.M0)
        self.T_out = self._DiskTemperature(self.M0, self.r_out)
        self.Lbol0 = self.Luminosity(0.0)
        self.Lbol = self.Luminosity

        self.disk_history = {}

        #if 'mcd' in self.spec_pars['type']:
        #    self.fcol = self.spec_pars['fcol'][self.spec_pars['type'].index('mcd')]
        #if 'simpl' in self.spec_pars['type']:
        #    self.fcol = self.spec_pars['fcol'][self.spec_pars['type'].index('simpl')]    
        #if 'zebra' in self.pf['source_sed']:
        #    self.T = self.src_pars['temperature']#[self.spec_pars['type'].index('zebra')]

        if self.pf['source_sed'] in sptypes:
            pass
        elif isinstance(self.pf['source_sed'], basestring):
            from_lit = read_lit(self.pf['source_sed'])
            self._UserDefined = from_lit.Spectrum
        elif type(self.pf['source_sed']) in [np.ndarray, tuple, list]:
            E, LE = self.pf['source_sed']
            tmp = interp1d(E, LE, kind='cubic')
            self._UserDefined = lambda E, t: tmp.__call__(E)
        else:
            self._UserDefined = self.pf['source_sed']    
            
        # Convert spectral types to strings
        #self.N = len(self.spec_pars['type'])
        #self.type_by_num = []
        #self.type_by_name = []
        #for i, sptype in enumerate(self.spec_pars['type']):
        #    if type(sptype) != int:
        #        
        #        if sptype in sptypes:
        #            self.type_by_name.append(sptype)                
        #            self.type_by_num.append(sptypes[sptype])
        #        elif type(sptype) is FunctionType:
        #            self._UserDefined = sptype
        #        else:
        #            from_lit = read_lit(sptype)
        #            self._UserDefined = from_lit.Spectrum
        #        
        #        continue
        #    
        #    self.type_by_num.append(sptype)
        #    self.type_by_name.append(list(sptypes.keys())[list(sptypes.values()).index(sptype)])                
                
    def _SchwartzchildRadius(self, M):
        return 2. * self._GravitationalRadius(M)

    def _GravitationalRadius(self, M):
        """ Half the Schwartzchild radius. """
        return G * M * g_per_msun / c**2    
        
    def _MassAccretionRate(self, M=None): 
        return self.Luminosity(0, M=M) / self.epsilon / c**2    
        
    def _DiskInnermostRadius(self, M):      
        """
        Inner radius of disk.  Unless SourceISCO > 0, will be set to the 
        inner-most stable circular orbit for a BH of mass M.
        """
        return self.pf['source_isco'] * self._GravitationalRadius(M)
            
    def _DiskInnermostTemperature(self, M):
        """
        Temperature (in Kelvin) at inner edge of the disk.
        """
        return (3. * G * M * g_per_msun * self._MassAccretionRate(M) / \
            8. / np.pi / self._DiskInnermostRadius(M)**3 / sigma_SB)**0.25
    
    def _DiskTemperature(self, M, r):
        return ((3. * G * M * g_per_msun * self._MassAccretionRate(M) / \
            8. / np.pi / r**3 / sigma_SB) * \
            (1. - (self._DiskInnermostRadius(M) / r)**0.5))**0.25
            
    def _PowerLaw(self, E, t=0.0):    
        """
        A simple power law X-ray spectrum - this is proportional to the 
        *energy* emitted at E, not the number of photons.  
        """

        return E**self.pf['source_alpha']
        
    def _SIMPL(self, E, t=0.0):
        """
        Purpose:
        --------
        Convolve an input spectrum with a Comptonization kernel.
    
        Inputs:
        -------
        Gamma  - Power-law index, LE ~ E**(-Gamma)
        fsc    - Fraction of seed photons that get scattered
                 (assumes all input photons have same probability of being scattered
                 and that scattering is energy-independent)
        fref   - Of the photons that impact the disk after a scattering, this is the
                 fraction that reflect back off the disk to the observer instead of 
                 being absorbed and thermalized (default 1)
        uponly - False: SIMPL-2, non-rel Comptonization, up- and down-scattering
                 True:  SIMPL-1, relativistic Comptoniztion, up-scattering only
    
        Outputs: (dictionary)
        --------
        LE - Absorbed power-law luminosity array [keV s^-1]
        E  - Energy array [keV]
        dE - Differential energy array [keV]
        
        References
        ----------
        Steiner et al. (2009). Thanks Greg Salvesen for the code!
        
        """

        # Input photon distribution
        if self.pf['source_sed'] == 'zebra':
            nin = lambda E0: _Planck(E0, self.T) / E0
        else:
            nin = lambda E0: self._MultiColorDisk(E0, t) / E0
    
        fsc = self.pf['source_fsc']

        # Output photon distribution - integrate in log-space         
        #integrand = lambda E0: nin(10**E0) \
        #    * self._GreensFunctionSIMPL(10**E0, E) * 10**E0

        #nout = (1.0 - fsc) * nin(E) + fsc \
        #    * quad(integrand, np.log10(self.Emin),
        #        np.log10(self.Emax))[0] * np.log(10.)  
        
        dlogE = self.pf['source_dlogE']
        ma = np.log10(self.Emax)
        mi = np.log10(self.Emin)
        N = (ma - mi) / dlogE + 1
        Earr = 10**np.arange(mi, ma+dlogE, dlogE)
        
        if type(E) is np.ndarray:
            nout = []
            for nrg in E:
                gf = [self._GreensFunctionSIMPL(EE, nrg) for EE in Earr]
                integrand = np.array(list(map(nin, Earr))) * np.array(gf) * Earr
                
                nout.append((1.0 - fsc) * nin(nrg) + fsc \
                    * np.trapz(integrand, dx=dlogE) * np.log(10.))
                    
            nout = np.array(nout)        
        else:
            gf = [self._GreensFunctionSIMPL(EE, E) for EE in Earr]
            integrand = np.array(list(map(nin, Earr))) * np.array(gf) * Earr
            
            nout = (1.0 - fsc) * nin(E) + fsc \
                * np.trapz(integrand, dx=dlogE) * np.log(10.)
         
        # Output spectrum
        return nout * E
    
    def _GreensFunctionSIMPL(self, Ein, Eout):
        """
        Must perform integral transform to compute output photon distribution.
        """
           
        # Careful with Gamma...
        # In Steiner et al. 2009, Gamma is n(E) ~ E**(-Gamma),
        # but n(E) and L(E) are different by a factor of E (see below)
        Gamma = -self.pf['source_alpha'] + 1.0
        
        if self.pf['source_uponly']:
            if Eout >= Ein:
                return (Gamma - 1.0) * (Eout / Ein)**(-1.0 * Gamma) / Ein
            else:
                return 0.0
        else:
            if Eout >= Ein:
                return (Gamma - 1.0) * (Gamma + 2.0) / (1.0 + 2.0 * Gamma) * \
                    (Eout / Ein)**(-1.0 * Gamma) / Ein
            else:
                return (Gamma - 1.0) * (Gamma + 2.0) / (1.0 + 2.0 * Gamma) * \
                    (Eout / Ein)**(Gamma + 1.0) / Ein
    
    def _MultiColorDisk(self, E, t=0.0):
        """
        Soft component of accretion disk spectra.

        References
        ----------
        Mitsuda et al. 1984, PASJ, 36, 741.

        """         

        # If t > 0, re-compute mass, inner radius, and inner temperature
        if t > 0 and self.pf['source_evolving'] \
            and t != self.last_renormalized:
            self.M = self.Mass(t)
            self.r_in = self._DiskInnermostRadius(self.M)
            self.r_out = self.pf['source_rmax'] * self._GravitationalRadius(self.M)
            self.T_in = self._DiskInnermostTemperature(self.M)
            self.T_out = self._DiskTemperature(self.M, self.r_out)
        
        integrand = lambda T, nrg: (T / self.T_in)**(-11. / 3.) \
            * _Planck(nrg, T) / self.T_in
            
        if type(E) == np.ndarray:
            result = \
                np.array([quad(lambda T: integrand(T, nrg), 
                    self.T_out, self.T_in)[0] for nrg in E])
        else:
            result = quad(lambda T: integrand(T, E), self.T_out, self.T_in)[0]
            
        return result

    def SourceOn(self, t):
        """ See if source is on. Provide t in code units. """        
        
        if not self.variable:
            return True
            
        if t < self.tau:
            return True
            
        if self.fduty == 1:
            return False    
            
        nacc = t / (self.tau + self.toff)
        if nacc % 1 < self.fduty:
            return True
        else:
            return False
            
    def _Intensity(self, E, t=0, absorb=True):
        """
        Return quantity *proportional* to fraction of bolometric luminosity 
        emitted at photon energy E.  Normalization handled separately.
        """
                                
        if self.pf['source_sed'] == 'pl': 
            Lnu = self._PowerLaw(E, t)    
        elif self.pf['source_sed'] == 'mcd':
            Lnu = self._MultiColorDisk(E, t)
        elif self.pf['source_sed'] == 'sazonov2004':
            Lnu = self._UserDefined(E, t)            
        elif self.pf['source_sed'] == 'simpl':
            Lnu = self._SIMPL(E, t)
        elif self.pf['source_sed'] == 'zebra':
            Lnu = self._SIMPL(E, t)            
        else:
            Lnu = self._UserDefined(E, t)
            #Lnu = 0.0
            
        if self.pf['source_logN'] > 0 and absorb:
            Lnu *= self._hardening_factor(E)
        
        return Lnu          
            
    #def _NormalizeSpectrum(self, t=0.):
    #    Lbol = self.Luminosity()
    #    # Treat simPL spectrum special
    #    if self.pf['source_sed'] == 'simpl':
    #        integral, err = quad(self._MultiColorDisk,
    #            self.EminNorm, self.EmaxNorm, args=(t, False))
    #    else:
    #        integral, err = quad(self._Intensity,
    #            self.EminNorm, self.EmaxNorm, args=(t, False))
    #        
    #    norms = Lbol / integral
    #        
    #    return norms
            
    def Luminosity(self, t=0.0, M=None):
        """
        Returns the bolometric luminosity of a source in units of erg/s.  
        For accreting black holes, the bolometric luminosity will increase 
        with time, hence the optional 't' and 'M' arguments.
        """

        if not self.SourceOn(t):
            return 0.0
            
        Mnow = self.Mass(t)
        if M is not None:
            Mnow = M

        return self.epsilon * 4.0 * np.pi * G * Mnow * g_per_msun * m_p \
            * c / sigma_T

    def Mass(self, t):
        """
        Compute black hole mass after t (seconds) have elapsed.  Relies on 
        initial mass self.M, and (constant) radiaitive efficiency self.epsilon.
        """        
        
        if self.variable:
            nlifetimes = int(t / (self.tau + self.toff))
            dtacc = nlifetimes * self.tau
            M0 = self.M0 * np.exp(((1.0 - self.epsilon) / self.epsilon) * dtacc / t_edd)  
            dt = t - nlifetimes * (self.tau + self.toff)
        else:
            M0 = self.M0
            dt = t

        return M0 * np.exp(((1.0 - self.epsilon) / self.epsilon) * dt / t_edd)         

    def Age(self, M):
        """
        Compute age of black hole based on current time, current mass, and initial mass.
        """            
                    
        return np.log(M / self.M0) * (self.epsilon / (1. - self.epsilon)) * t_edd


        
