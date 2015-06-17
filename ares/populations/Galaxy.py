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
from .Halo import HaloPopulation
from .Population import Population
from collections import namedtuple
from ..sources.Source import Source
from ..sources import Star, BlackHole
from ..util.PrintInfo import print_pop
from ..physics.Constants import s_per_yr, g_per_msun, erg_per_ev, rhodot_cgs, \
    E_LyA
from ..util.SetDefaultParameterValues import StellarParameters, \
    BlackHoleParameters

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

lftypes = ['schecter', 'dpl']    
    
class LiteratureSource(Source):
    def __init__(self, **kwargs):
        self.pf = kwargs
        Source.__init__(self)
        
        _src = read_lit(self.pf['pop_sed'])
        
        if hasattr(_src, 'Spectrum'):
            self._Intensity = _src.Spectrum
            
def normalize_sed(pop):
    """
    Convert yield to erg / g.
    """
    
    if pop.pf['pop_yield_units'] == 'erg/s/SFR':
        energy_per_sfr = pop.pf['pop_yield'] * s_per_yr / g_per_msun
    else:
        E1 = pop.pf['pop_EminNorm']
        E2 = pop.pf['pop_EmaxNorm']        
        erg_per_phot = pop.src.AveragePhotonEnergy(E1, E2) * erg_per_ev
        energy_per_sfr = pop.pf['pop_yield'] * erg_per_phot / g_per_msun
        
        if pop.pf['pop_yield_units'] == 'photons/baryon':
            energy_per_sfr *= pop.cosm.g_per_baryon
        elif pop.pf['pop_yield_units'] == 'photons/Msun':
            pass
        else:
            u = pop.pf['pop_yield_units']
            raise ValueError('Unrecognized yield units: %s' % u)

    return energy_per_sfr

class GalaxyPopulation(HaloPopulation):
    def __init__(self, **kwargs):
        """
        Initializes a GalaxyPopulation object (duh).
        
        Requires Schecter function parameters as well as the specification
        of a bandpass. If you'd like to convert to some other band (e.g., 
        you know the 2-10 keV LF but want 0.5-2 keV), you'll need to specify
        the SED.
        """
        
        # This is basically just initializing an instance of the cosmology
        # class. Also creates the parameter file attribute ``pf``.
        HaloPopulation.__init__(self, **kwargs)
        self.pf.update(**kwargs)

        # Initialize the LF
        if self.pf['pop_lf'] is None:
            pass
        elif self.pf['pop_lf'] in lftypes:
            pass
        elif type(self.pf['pop_lf']) is FunctionType:
            self._UserDefinedLF = self.pf['pop_lf']
        else:
            from_lit = read_lit(self.pf['pop_lf'])
            self._UserDefinedLF = from_lit.LuminosityFunction

        self._conversion_factors = {}

    @property
    def solve_rte(self):
        return self.pf['pop_solve_rte']

    @property
    def sawtooth(self):
        return self.pf['pop_sawtooth']        
        
    @property
    def is_lya_src(self):
        if not hasattr(self, '_is_lya_src'):
            self._is_lya_src = \
                self.pf['pop_Emin'] <= E_LyA <= self.pf['pop_Emax']    
        
        return self._is_lya_src
    
    @property
    def _Source(self):
        if not hasattr(self, '__Source'):
            if self.pf['pop_sed'] == 'bb':
                self.__Source = Star
            elif self.pf['pop_sed'] in ['pl', 'mcd', 'simpl']:
                self.__Source = BlackHole
            else: 
                self.__Source = LiteratureSource
        
        return self.__Source
        
    @property
    def src_kwargs(self):
        """
        Dictionary of kwargs to pass on to an ares.source instance.
        
        This is basically just converting pop_* parameters to source_* 
        parameters.
        
        """
        if not hasattr(self, '_src_kwargs'):
            self._src_kwargs = {}
            if self._Source is Star:
                spars = StellarParameters()
                for par in spars:
                    
                    par_pop = par.replace('source', 'pop')
                    if par_pop in self.pf:
                        self._src_kwargs[par] = self.pf[par_pop]
                    else:
                        self._src_kwargs[par] = spars[par]
                        
            elif self._Source is BlackHole:
                bpars = BlackHoleParameters()
                for par in bpars:
                    par_pop = par.replace('source', 'pop')
                    
                    if par_pop in self.pf:
                        self._src_kwargs[par] = self.pf[par_pop]
                    else:
                        self._src_kwargs[par] = bpars[par]
            else:
                self._src_kwargs = self.pf.copy()
        
        return self._src_kwargs

    @property
    def src(self):
        if not hasattr(self, '_src'):
            self._src = self._Source(**self.src_kwargs)
                    
        return self._src            

    @property
    def yield_per_sfr(self):
        if not hasattr(self, '_yield_per_sfr'):
            self._yield_per_sfr = normalize_sed(self)
            
        return self._yield_per_sfr
            
    @property
    def is_fcoll_model(self):
        if not hasattr(self, '_is_fcoll_model'):
            self._is_fcoll_model = (self.pf['pop_rhoL'] is None) and \
                (self.pf['pop_emissivity'] is None)
        return self._is_fcoll_model
        
    @property
    def rhoL_from_sfrd(self):
        if not hasattr(self, '_rhoL_from_sfrd'):
            self._rhoL_from_sfrd = self.is_fcoll_model \
                or self.pf['pop_sfrd'] is not None
                
        return self._rhoL_from_sfrd
    
    #@property
    #def Lmin(self):
    #    if not hasattr(self, '_Lmin'):
    #        self._Lmin = self.pf['lf_Lmin']
    #
    #    return self._Lmin
    
    #@property
    #def Lstar(self):
    #    if not hasattr(self, '_Lstar'):
    #        self._Lstar = self.pf['lf_Lstar']
    #
    #    return self._Lstar
        
    #@property
    #def phi0(self):    
    #    if not hasattr(self, '_phi0'):
    #        self._phi0 = self.pf['lf_norm']
    #
    #    return self._phi0
    
    #@property
    #def lf_type(self):    
    #    if not hasattr(self, '_lf_type'):
    #        if type(self.pf['pop_lf']) in lftypes:
    #            self._lf_type = 'pre-defined'
    #        else:
    #            self._lf_type = 'user'
    #
    #    return self._lf_type

    #@property
    #def lf_zfunc_type(self):    
    #    if not hasattr(self, '_lf_zfunc_type'):
    #        if type(self.pf['lf_zfunc']) == FunctionType:
    #            self._lf_zfunc_type = 'user'
    #        elif self.pf['lf_zfunc'] == 'ueda':
    #            self._lf_zfunc_type = self._Ueda
    #        else:
    #            raise NotImplemented('%s not implemented!' % self.pf['lf_zfunc'])
    #
    #    return self._lf_zfunc_type      
    
    #def norm(self):
    #    return quad(lambda x: self.__call__(x), self.Lmin, 1e50)[0]
    
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
        
        if self.is_fcoll_model:
            raise TypeError('this is an fcoll model!')
        
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
        
        if self.is_fcoll_model:
            raise TypeError('this is an fcoll model!')

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
        if (Emin is not None) and (self.src is not None):
            different_band = True
        else:
            Emin = self.pf['pop_Emin']

        # Upper bound
        if (Emax is not None) and (self.src is not None):
            different_band = True
        else:
            Emax = self.pf['pop_Emax']
            
        # Modify band if need be
        if different_band:    
            
            if (Emin, Emax) in self._conversion_factors:
                return self._conversion_factors[(Emin, Emax)]
            
            if Emin < self.pf['pop_Emin']:
                print "WARNING: Emin < pop_Emin"
            if Emax > self.pf['pop_Emax']:
                print "WARNING: Emax > pop_Emax"    
            
            factor = quad(self.src.Spectrum, Emin, Emax)[0]
            
            self._conversion_factors[(Emin, Emax)] = factor
            
            return factor
        
        return 1.0

    @property
    def _sfrd(self):
        if not hasattr(self, '__sfrd'):
            if self.pf['pop_sfrd'] is None:
                self.__sfrd = None
            elif type(self.pf['pop_sfrd']) is FunctionType:
                self.__sfrd = self.pf['pop_sfrd']
            else:
                tmp = read_lit(self.pf['pop_sfrd'])
                self.__sfrd = lambda z: tmp.SFRD(z, **self.pf['pop_kwargs'])
        
        return self.__sfrd
    
    @property
    def _lf(self):
        if not hasattr(self, '__lf'):
            if self.pf['pop_rhoL'] is None and self.pf['pop_lf'] is None:
                self.__lf = None
            elif type(self.pf['pop_rhoL']) is FunctionType:
                self.__lf = self.pf['pop_rhoL']
            elif type(self.pf['pop_lf']) is FunctionType:
                self.__lf = self.pf['pop_lf']  
            else:
                for key in ['pop_rhoL', 'pop_lf']:
                    if self.pf[key] is None:
                        continue
                        
                    tmp = read_lit(self.pf[key])
                    self.__lf = lambda L, z: tmp.LuminosityFunction(L, z=z,
                        **self.pf['pop_kwargs'])

                    break

        return self.__lf        

    def SFRD(self, z):
        """
        Compute the comoving star formation rate density (SFRD).
    
        Given that we're in the StellarPopulation class, we are assuming
        that all emissivities are tied to the star formation history. The
        SFRD can be supplied explicitly as a function of redshift, or can 
        be computed via the "collapsed fraction" formalism. That is, compute
        the SFRD given a minimum virial temperature of star forming halos 
        (Tmin) and a star formation efficiency (fstar).
    
        If supplied as a function, the units should be Msun yr**-1 cMpc**-3.
    
        Parameters
        ----------
        z : float
            redshift
    
        Returns
        -------
        Co-moving star-formation rate density at redshift z in units of
        g s**-1 cm**-3.
    
        """
    
        if z > self.zform:
            return 0.0
    
        # SFRD approximated by some analytic function    
        if self._sfrd is not None:
            return self.__sfrd(z) / rhodot_cgs
    
        # Most often: use fcoll model
        if self.is_fcoll_model:
           
            # SFRD computed via fcoll parameterization
            sfrd = self.pf['pop_fstar'] * self.cosm.rho_b_z0 \
                * self.dfcolldz(z) / self.cosm.dtdz(z)
            
            if sfrd < 0:
                negative_SFRD(z, self.pf['pop_Tmin'], self.pf['pop_fstar'], 
                    self.dfcolldz(z) / self.cosm.dtdz(z), sfrd)
                sys.exit(1)
                
        else:
            raise NotImplemented('dunno how to model the SFRD!')
    
        return sfrd                           

    def Emissivity(self, z, E=None, Emin=None, Emax=None):
        """
        Compute the emissivity of this population as a function of redshift
        and rest-frame photon energy [eV].
        
        Parameters
        ----------
        z : int, float
        
        """
        
        if self.is_fcoll_model or self.pf['pop_sfrd'] is not None:            
            emiss = self.SFRD(z) * self.yield_per_sfr
        else:
            raise NotImplemented('help')    
            
        # Convert from reference band to arbitrary band
        emiss *= self._convert_band(Emin, Emax)
                        
        if E is not None:
            return emiss * self.src.Spectrum(E)
        else:
            return emiss    
    
    
    #def LuminosityDensity(self, z, Emin=None, Emax=None):
    #    """
    #    Return the luminosity density in the given band.
    #    
    #    ..note:: By default, returns luminosity density in the reference band 
    #        defined by ``pop_lf_EminNorm`` and ``pop_lf_EmaxNorm``
    #        
    #    Returns
    #    -------
    #        
    #    """
    #    
    #    return self.Emissivity(z, Emin=Emin, Emax=Emax)
        

        

              
        