"""

BlackHoleSource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Jul  8 09:56:38 MDT 2013

Description: 

"""

import numpy as np
from scipy.integrate import quad, romberg
from ..physics.Constants import *
from .StellarSource import _Planck
from ..util.SetDefaultParameterValues import BlackHoleParameters
from ..physics.CrossSections import PhotoIonizationCrossSection as sigma_E

sptypes = {'pl':0, 'mcd':1, 'qso':2, 'simpl':3, 'zebra':4}

tiny_number = 1e-12

class BlackHoleSource(object):
    """ Class for creation and manipulation of compact object sources. """
    def __init__(self, pf, src_pars, spec_pars):
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
        
        self.pf = pf
        self.src_pars = src_pars
        self.spec_pars = spec_pars
        
        self._name = 'BlackHoleSource'
        
        self.M0 = self.src_pars['mass']
        self.epsilon = self.src_pars['eta']
        
        # Duty cycle parameters
        self.tau = self.src_pars['lifetime'] * self.pf['time_units']
        self.fduty = self.src_pars['fduty'] 
        self.variable = self.fduty < 1
        if self.src_pars['fduty'] == 1:
            self.variable = self.tau < self.pf['stop_time']
        
        self.toff = self.tau * (self.fduty**-1. - 1.)
        
        # Disk properties
        self.last_renormalized = 0.0
        self.r_in = self._DiskInnermostRadius(self.M0)
        self.r_out = self.src_pars['rmax'] * self._GravitationalRadius(self.M0)
        self.T_in = self._DiskInnermostTemperature(self.M0)
        self.T_out = self._DiskTemperature(self.M0, self.r_out)
        self.Lbol = self.Luminosity(0.0)
        
        self.disk_history = {}
        
        #if 'mcd' in self.spec_pars['type']:
        #    self.fcol = self.spec_pars['fcol'][self.spec_pars['type'].index('mcd')]
        #if 'simpl' in self.spec_pars['type']:
        #    self.fcol = self.spec_pars['fcol'][self.spec_pars['type'].index('simpl')]    
        if 'zebra' in self.spec_pars['type']:
            self.T = self.src_pars['temperature']#[self.spec_pars['type'].index('zebra')]
        
        # Parameters for the Sazonov & Ostriker AGN template
        self.Alpha = 0.24
        self.Beta = 1.60
        self.Gamma = 1.06
        self.E_1 = 83e3
        self.K = 0.0041
        self.E_0 = (self.Beta - self.Alpha) * self.E_1
        self.A = np.exp(2e3 / self.E_1) * 2e3**self.Alpha
        self.B = ((self.E_0**(self.Beta - self.Alpha)) \
            * np.exp(-(self.Beta - self.Alpha))) / \
            (1.0 + (self.K * self.E_0**(self.Beta - self.Gamma)))
            
        # Normalization constants to make the SOS04 spectrum continuous.
        self.SX_Normalization = 1.0
        self.UV_Normalization = self.SX_Normalization \
            * ((self.A * 2e3**-self.Alpha) * \
            np.exp(-2e3 / self.E_1)) \
            / ((1.2 * 2e3**-1.7) * np.exp(2000.0 / 2000.))
        self.IR_Normalization = self.UV_Normalization * ((1.2 * 10**-1.7) \
            * np.exp(10.0 / 2e3)) / (1.2 * 159 * 10**-0.6)
        self.HX_Normalization = self.SX_Normalization \
            * (self.A * self.E_0**-self.Alpha * \
            np.exp(-self.E_0 / self.E_1)) / (self.A * self.B \
            * (1.0 + self.K * self.E_0**(self.Beta - self.Gamma)) * \
            self.E_0**-self.Beta)
            
        # Convert spectral types to strings
        self.N = len(self.spec_pars['type'])
        self.type_by_num = []
        self.type_by_name = []
        for i, sptype in enumerate(self.spec_pars['type']):
            if type(sptype) != int:
                self.type_by_name.append(sptype)                
                self.type_by_num.append(sptypes[sptype])
                continue
            
            self.type_by_num.append(sptype)
            self.type_by_name.append(sptypes.keys()[sptypes.values().index(sptype)])                
                
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
        return self.src_pars['isco'] * self._GravitationalRadius(M)
            
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
            
    def _PowerLaw(self, E, i, t=0.0):    
        """
        A simple power law X-ray spectrum - this is proportional to the 
        *energy* emitted at E, not the number of photons.  
        """

        return E**self.spec_pars['alpha'][i]
        
    def _SIMPL(self, E, i, t):
        '''
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
        '''

        # Input photon distribution
        if self.type_by_name[i] == 'zebra':
            nin = lambda E0: _Planck(E0, self.T) / E0
        else:
            nin = lambda E0: self._MultiColorDisk(E0, i, t) / E0
    
        fsc = self.spec_pars['fsc'][i]

        # Output photon distribution - integrate in log-space         
        integrand = lambda E0: nin(10**E0) * self._GreensFunctionSIMPL(10**E0, E, i) * 10**E0

        nout = (1.0 - fsc) * nin(E) + fsc \
            * quad(integrand, np.log10(self.spec_pars['Emin'][i]),
                np.log10(self.spec_pars['Emax'][i]))[0] * np.log(10.)
                                
        # Output spectrum
        return nout * E
    
    def _GreensFunctionSIMPL(self, Ein, Eout, i):
        """
        Must perform integral transform to compute output photon distribution.
        """
           
        # Careful with Gamma...
        # In Steiner et al. 2009, Gamma is n(E) ~ E**(-Gamma),
        # but n(E) and L(E) are different by a factor of E (see below)
        Gamma = -self.spec_pars['alpha'][i] + 1.0
        
        if self.spec_pars['uponly'][i]:
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
    
    def _MultiColorDisk(self, E, i, Type, t=0.0):
        """
        Soft component of accretion disk spectra. 
        
        References
        ----------
        Mitsuda et al. 1984, PASJ, 36, 741.
        
        """         
        
        # If t > 0, re-compute mass, inner radius, and inner temperature
        if t > 0 and self.spec_pars['evolving'] and t != self.last_renormalized:
            self.M = self.Mass(t)
            self.r_in = self._DiskInnermostRadius(self.M)
            self.r_out = self.src_pars['rmax'] * self._GravitationalRadius(self.M)
            self.T_in = self._DiskInnermostTemperature(self.M)
            self.T_out = self._DiskTemperature(self.M, self.r_out)
                    
        integrand = lambda T: (T / self.T_in)**(-11. / 3.) \
            * _Planck(E, T) / self.T_in
            
        return quad(integrand, self.T_out, self.T_in)[0]
        
    def _QuasarTemplate(self, E, i, Type, t=0):
        """
        Broadband quasar template spectrum.
        
        References
        ----------
        Sazonov, S., Ostriker, J.P., & Sunyaev, R.A. 2004, MNRAS, 347, 144.
        """
        
        op = (E < 10)
        uv = (E >= 10) & (E < 2e3) 
        xs = (E >= 2e3) & (E < self.E_0)
        xh = (E >= self.E_0) & (E < 4e5)        
        
        if type(E) in [int, float]:
            if op:
                F = self.IR_Normalization * 1.2 * 159 * E**-0.6
            elif uv:
                F = int(uv) * self.UV_Normalization * 1.2 * E**-1.7 * np.exp(E / 2000.0)
            elif xs:
                F = self.SX_Normalization * self.A * E**-self.Alpha * np.exp(-E / self.E_1)
            elif xh:
                F = self.HX_Normalization * self.A * self.B * (1.0 + self.K * \
                    E**(self.Beta - self.Gamma)) * E**-self.Beta
            else: 
                F = 0
                
        else:
            F = np.zeros_like(E)
            F += op * self.IR_Normalization * 1.2 * 159 * E**-0.6
            F += uv * self.UV_Normalization * 1.2 * E**-1.7 * np.exp(E / 2000.0)
            F += xs * self.SX_Normalization * self.A * E**-self.Alpha * np.exp(-E / self.E_1)
            F += xh * self.HX_Normalization * self.A * self.B * (1.0 + self.K * \
                    E**(self.Beta - self.Gamma)) * E**-self.Beta
        
        return E * F
    
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
            
    def _Intensity(self, E, i=0, t=0, absorb=True):
        """
        Return quantity *proportional* to fraction of bolometric luminosity 
        emitted at photon energy E.  Normalization handled separately.
        """
                        
        if self.type_by_name[i] == 'pl': 
            Lnu = self._PowerLaw(E, i, t)    
        elif self.type_by_name[i] == 'mcd':
            Lnu = self._MultiColorDisk(E, i, t)
        elif self.type_by_name[i] == 'qso':
            Lnu = self._QuasarTemplate(E, i, t)
        elif self.type_by_name[i] == 'simpl':
            Lnu = self._SIMPL(E, i, t)
        elif self.type_by_name[i] == 'zebra':
            Lnu = self._SIMPL(E, i, t)            
        else:
            Lnu = 0.0
            
        if self.spec_pars['logN'][i] > 0 and absorb:
            Lnu *= np.exp(-10.**self.spec_pars['logN'][i] \
                * (sigma_E(E, 0) + y * sigma_E(E, 1)))  
        
        return Lnu          
            
    def _NormalizeSpectrum(self, t=0.):
        norms = np.zeros(self.N)
        Lbol = self.Luminosity()
        for i in xrange(self.N):
            # Treat simPL spectrum special
            if self.type_by_name[i] == 'simpl':
                integral, err = quad(self._MultiColorDisk,
                    self.spec_pars['EminNorm'][i], self.spec_pars['EmaxNorm'][i], 
                    args=(i, t, False))
            else:
                integral, err = quad(self._Intensity,
                    self.spec_pars['EminNorm'][i], self.spec_pars['EmaxNorm'][i], 
                    args=(i, t, False))
                
            norms[i] = self.spec_pars['fraction'][i] * Lbol / integral
            
        return norms
            
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
        

        