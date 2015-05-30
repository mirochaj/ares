"""

GalaxyPopulation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat May 23 12:13:03 CDT 2015

Description: 

"""

import numpy as np
import matplotlib.pyplot as pl
from types import FunctionType
from ..physics import Cosmology
from ..util import ParameterFile
from scipy.integrate import quad
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


class GalaxyPopulation:
    def __init__(self, **kwargs):
        """
        Initializes a GalaxyPopulation object (duh).
        
        Requires Schecter function parameters as well as the specification
        of a bandpass. If you'd like to convert to some other band (e.g., 
        you know the 2-10 keV LF but want 0.5-2 keV), you'll need to specify
        the SED.
        """

        self.pf = ParameterFile(**kwargs)
        
        self.grid = None
        self.cosm = Cosmology(
            omega_m_0=self.pf['omega_m_0'], 
            omega_l_0=self.pf['omega_l_0'], 
            omega_b_0=self.pf['omega_b_0'],  
            hubble_0=self.pf['hubble_0'],  
            helium_by_number=self.pf['helium_by_number'], 
            cmb_temp_0=self.pf['cmb_temp_0'], 
            approx_highz=self.pf['approx_highz'],
            sigma_8=self.pf['sigma_8'],
            primordial_index=self.pf['primordial_index'])

        # Re-normalize bolometric output
        self._init_rs()
        
        self._conversion_factors = {}
        
    def __call__(self, L, z=None):
        """
        Compute luminosity function.
        """
    
        return self.LuminosityFunction(LM, z=z)    

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
            if type(self.pf['lf_func']) == FunctionType:
                self._lf_type = 'user'
            else:
                self._lf_type = self.pf['lf_func']
    
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

        if self.lf_type == 'user':
            phi = self.self.pf['lf_func'](L)
        elif self.lf_type == 'schecter':
            phi = self._SchecterFunction(L)
        elif self.lf_type == 'dpl':
            phi = self._DoublePowerLaw(L)
        else:
            raise NotImplemented('Function type %s not supported' % self.lf_type)

        return phi * self._RedshiftEvolution(z) * self._convert_band(Emin, Emax)
    
    def _convert_band(self, Emin, Emax):
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
        
        integrand = lambda LL: LL * self.LuminosityFunction(LL, z=z,
            Emin=Emin, Emax=Emax)
        
        LphiL = quad(integrand, Lmin, Lmax)[0]
        
        return LphiL
        
    def SpaceDensity(self, z, Lmin=None, Lmax=None):
        """
        Compute space density of this population of objects.
    
        ..note:: This is just integrating the LF in some band.
    
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
    
    def _RedshiftEvolution(self, z):
        """
        Multiplicative redshift evolution factor for luminosity function.
        
        Parameters
        ----------
        z : int, float
            Redshift of interest.
            
            
        """
        
        if (not self.pf['lf_zfunc']) or (z is None):
            return 1.
        
        # User-defined redshift evolution
        if self.lf_zfunc_type == 'user':
            return self.pf['lf_zfunc'](z)
            
        # Piecewise redshift evolution model from Ueda et al. 2003    
        elif self.pf['lf_zdep'] == 'ueda':
            pass
            
            
            
    def CompositeSED(self, E):
        pass
    
    def SFRD(self):
        pass
        
    def XrayLuminosityDensity(self):
        pass  
        
    def Emissivity(self):
        pass
        
              
        