"""

GalaxyPopulation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat May 23 12:13:03 CDT 2015

Description: 

"""

import numpy as np
from ..util import read_lit
import matplotlib.pyplot as pl
from types import FunctionType
from ..physics import Cosmology
from ..util import ParameterFile
from scipy.integrate import quad
from .Population import Population
from ..util.PrintInfo import print_pop
from ..util.NormalizeSED import norm_sed

try:
    from scipy.special import erfc
except ImportError:
    pass
    
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
    
try:
    from hmf import MassFunction
except ImportError:
    pass
    
sptypes = ['pl', 'mcd', 'simpl']    
lftypes = ['schecter', 'dpl']

class GalaxyPopulation(Population):
    def __init__(self, **kwargs):
        """
        Initializes a GalaxyPopulation object (duh).
        
        Requires Schecter function parameters as well as the specification
        of a bandpass. If you'd like to convert to some other band (e.g., 
        you know the 2-10 keV LF but want 0.5-2 keV), you'll need to specify
        the SED.
        """
        
        Population.__init__(self, **kwargs)

        # 
        if self.pf['pop_sed'] in sptypes:
            pass
        elif type(self.pf['pop_sed']) is FunctionType:
            self._UserDefinedSpectrum = self.pf['pop_sed']
        else:
            from_lit = read_lit(self.pf['pop_sed'])
            self.pf['pop_sed'] = from_lit.Spectrum
        
        if self.pf['pop_lf'] in lftypes:
            pass
        elif type(self.pf['pop_lf']) is FunctionType:
            self._UserDefinedLF = self.pf['pop_lf']
        else:
            from_lit = read_lit(self.pf['pop_lf'])
            self._UserDefinedLF = from_lit.LuminosityFunction
          
        # Re-normalize bolometric output
        self._init_rs()
        
        self._conversion_factors = {}  
            
    def _init_rs(self):
        """
        Initialize RadiationSource instance - normalize LW, UV, and
        X-ray luminosities appropriately.
        """
    
        sed = norm_sed(self, self.grid)
    
        self.rs = sed['rs']
        self.cX = sed['cX']
        self.cLW = sed['cLW']
        self.cUV = sed['cUV']
        self.erg_per_LW = sed['erg_per_LW']
        self.erg_per_UV = sed['erg_per_UV']
        self.erg_per_X = sed['erg_per_X']
        self.Nlw = sed['Nlw']
        self.Nion = sed['Nion']
        self.Nx = sed['Nx']
        self.Elw = sed['Elw']
        self.Eion = sed['Eion']
        self.Ex = sed['Ex']
    
    @property
    def Lmin(self):
        if not hasattr(self, '_Lmin'):
            self._Lmin = self.pf['lf_Lmin']
    
        return self._Lmin
    
    @property
    def Lstar(self):
        if not hasattr(self, '_Lstar'):
            self._Lstar = self.pf['lf_Lstar']
    
        return self._Lstar
        
    @property
    def phi0(self):    
        if not hasattr(self, '_phi0'):
            self._phi0 = self.pf['lf_norm']
    
        return self._phi0
    
    @property
    def lf_type(self):    
        if not hasattr(self, '_lf_type'):
            if type(self.pf['pop_lf']) in lftypes:
                self._lf_type = 'pre-defined'
            else:
                self._lf_type = 'user'
    
        return self._lf_type

    @property
    def lf_zfunc_type(self):    
        if not hasattr(self, '_lf_zfunc_type'):
            if type(self.pf['lf_zfunc']) == FunctionType:
                self._lf_zfunc_type = 'user'
            elif self.pf['lf_zfunc'] == 'ueda':
                self._lf_zfunc_type = self._Ueda
            else:
                raise NotImplemented('%s not implemented!' % self.pf['lf_zfunc'])
    
        return self._lf_zfunc_type      
    
    def norm(self):
        return quad(lambda x: self.__call__(x), self.Lmin, 1e50)[0]
    
    def _SchecterFunction(self, L):
        """
        Schecter function for, e.g., the galaxy luminosity function.
        
        Parameters
        ----------
        L : float
        
        """
        
        return self.phi0 * (L / self.Lstar)**self.pf['lf_slope'] \
            * np.exp(-L / self.Lstar)
            
    def _DoublePowerLaw(self, L):
        """
        Double power-law function for, e.g., the quasar luminosity function.
        
        Parameters
        ----------
        L : float
        
        """
        return self.pf['lf_norm'] * ((L / self.Lstar)**self.pf['lf_gamma1'] \
            + (L / self.Lstar)**self.pf['lf_gamma1'])**-1.
                
    def integral(self, L1, L2):
        """
        Integral of the luminosity function over some interval (in luminosity).
        
        Parameters
        ----------
        L1 : int, float
        L2 : int, float
        
        """
        
        if self.lf_type == 'schecter':
            return (gamma(L1 / self.Lstar) * \
                (gammainc(self.pf['lf_slope'] + 1., L1 / self.Lstar) - 1.) \
                - gamma(L2 / self.Lstar) \
                * (gammainc(self.pf['lf_slope'] + 1., L2 / self.Lstar)) - 1.)
    
        else:
            raise NotImplemented('have not hadded support for anything but schecter yet.')
    
    def LuminosityFunction(self, L, z=None, Emin=None, Emax=None):
        """
        Compute luminosity function.

        Parameters
        ----------
        L : int, float
            Luminosity to consider
        z : int, float 
            Redshift.
        Emin : int, float
            Lower threshold of band to consider for LF [eV]
        Emax : int, float    
            Upper threshold of band to consider for LF [eV]        

        """

        L *= self._convert_band(Emin, Emax)

        if self.lf_type == 'user':
            phi = self._UserDefinedLF(L, z)
        elif self.lf_type == 'schecter':
            phi = self._SchecterFunction(L)
        elif self.lf_type == 'dpl':
            phi = self._DoublePowerLaw(L)
        else:
            raise NotImplemented('Function type %s not supported' % self.lf_type)

        return phi
    
    def _convert_band(self, Emin, Emax, Emin_from=None, Emax_from=None):
        """
        Convert from luminosity function in reference band to given bounds.
        
        Parameters
        ----------
        Emin : int, float
            Minimum energy [eV]
        Emax : int, float
            Maximum energy [eV]
            
        Returns
        -------
        Multiplicative factor that converts LF in reference band to that 
        defined by ``(Emin, Emax)``.
        
        """
        
        different_band = False

        # Lower bound
        if (Emin is not None) and (self.rs is not None):
            different_band = True
        else:
            Emin = self.pf['lf_Emin']
        
        # Upper bound
        if (Emax is not None) and (self.rs is not None):
            different_band = True
        else:
            Emax = self.pf['lf_Emax']
            
        # Modify band if need be
        if different_band:    
            
            if (Emin, Emax) in self._conversion_factors:
                return self._conversion_factors[(Emin, Emax)]
            
            d = quad(self.rs.Spectrum, self.pf['spectrum_Emin'], 
                self.pf['spectrum_Emax'])[0]
            n = quad(self.rs.Spectrum, Emin, Emax)[0]
            factor = n / d
            
            self._conversion_factors[(Emin, Emax)] = factor
            
            return factor
        
        return 1.0
    
    def LuminosityDensity(self, z, Emin=None, Emax=None, Lmin=None, Lmax=None):
        """
        Compute luminosity density of this population of objects.
                
        Parameters
        ----------
        z : int, float
            Redshift of interest
        
        Returns
        -------
        
        
        """
        
        if Lmin is None:
            Lmin = self.Lmin
        if Lmax is None:
            Lmax = 1e45
        
        integrand = \
            lambda LL: LL * self.LuminosityFunction(LL, z=z, Emin=Emin, Emax=Emax)
        
        LphiL = quad(integrand, Lmin, Lmax)[0]
        
        return LphiL
        
    def SpaceDensity(self, z, Lmin=None, Lmax=None):
        """
        Compute space density of this population of objects.
    
        ..note:: This is just integrating the LF.
    
        Parameters
        ----------
    
    
    
        """
    
        if Lmin is None:
            Lmin = self.Lmin
        if Lmax is None:
            Lmax = 1e45
    
        integrand = lambda LL: self.LuminosityFunction(LL, z=z)
    
        phiL = quad(integrand, Lmin, Lmax)[0]
    
        return phiL
        
    def CompositeSED(self, E):
        pass
    
    def SFRD(self):
        pass
        
    def XrayLuminosityDensity(self):
        pass  
        
    def Emissivity(self):
        pass
        
              
        