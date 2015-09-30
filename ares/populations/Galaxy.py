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
from .Halo import HaloPopulation
from .Population import Population
from collections import namedtuple
from ..sources.Source import Source
from ..sources import Star, BlackHole
from ..util.PrintInfo import print_pop
from scipy.integrate import quad, simps
from scipy.optimize import fsolve, fmin
from scipy.special import gamma, gammainc, gammaincc
from ..util import ParameterFile, MagnitudeSystem
from ..physics.Constants import s_per_yr, g_per_msun, erg_per_ev, rhodot_cgs, \
    E_LyA, rho_cgs, s_per_myr, cm_per_mpc
from ..util.SetDefaultParameterValues import StellarParameters, \
    BlackHoleParameters

try:
    from scipy.special import erfc
except ImportError:
    pass
    
try:
    import mpmath
except ImportError:
    pass    
    
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
    
log10 = np.log(10.)

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
    
    # Remove whitespace and convert everything to lower-case
    units = pop.pf['pop_yield_units'].replace(' ', '').lower()
        
    if units == 'erg/s/sfr':
        energy_per_sfr = pop.pf['pop_yield'] * s_per_yr / g_per_msun
    elif units == 'erg/s/hz/sfr':
        energy_per_sfr = pop.pf['pop_yield']
    else:
        E1 = pop.pf['pop_EminNorm']
        E2 = pop.pf['pop_EmaxNorm']
        erg_per_phot = pop.src.AveragePhotonEnergy(E1, E2) * erg_per_ev
        energy_per_sfr = pop.pf['pop_yield'] * erg_per_phot
        
        if units == 'photons/baryon':
            energy_per_sfr /= pop.cosm.g_per_baryon
        elif units == 'photons/msun':
            energy_per_sfr /= g_per_msun
        elif units == 'photons/s/sfr':
            energy_per_sfr *= s_per_yr / g_per_msun   
        else:
            raise ValueError('Unrecognized yield units: %s' % units)

    return energy_per_sfr

class GalaxyPopulation(HaloPopulation):
    def __init__(self, **kwargs):
        """
        Initializes a GalaxyPopulation object (duh).
        """

        # This is basically just initializing an instance of the cosmology
        # class. Also creates the parameter file attribute ``pf``.
        HaloPopulation.__init__(self, **kwargs)
        self.pf.update(**kwargs)

        self._eV_per_phot = {}
        self._conversion_factors = {}

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
        return self.pf['pop_model'].lower() == 'fcoll'
        
    @property    
    def is_ham_model(self):
        return self.pf['pop_model'].lower() == 'ham'
    
    @property
    def is_hod_model(self):
        return self.pf['pop_model'].lower() == 'hod'
    
    @property
    def is_user_model(self):
        return self.pf['pop_model'].lower() == 'user'    
        
    @property
    def rhoL_from_sfrd(self):
        if not hasattr(self, '_rhoL_from_sfrd'):
            self._rhoL_from_sfrd = self.is_fcoll_model \
                or self.pf['pop_sfrd'] is not None
                
        return self._rhoL_from_sfrd
    
    @property
    def magsys(self):
        if not hasattr(self, '_magsys'):
            self._magsys = MagnitudeSystem(**self.pf)
        
        return self._magsys
        
    @property
    def constraints(self):
        if not hasattr(self, '_constraints'):
            
            self._constraints = self.pf['pop_constraints'].copy()
            
            # Parameter file will have LF in Magnitudes...argh
            redshifts = self.pf['pop_constraints']['z']
            self._constraints['L_star'] = []
            
            for i, z in enumerate(redshifts):
                M = self.pf['pop_constraints']['M_star'][i]
                L = self.magsys.mAB_to_L(mag=M, z=z)
                self._constraints['L_star'].append(L)
        
        return self._constraints 
    
    @property
    def Macc(self):
        if not hasattr(self, '_Macc'):
            if self.pf['pop_Macc'] is None:
                self._Macc = None
            elif type(self.pf['pop_Macc']) is FunctionType:
                self._Macc = self.pf['pop_Macc']
            else:
                self._Macc = read_lit(self.pf['pop_Macc']).Macc
            
        return self._Macc        
    
    @property
    def Mmin(self):
        if not hasattr(self, '_Mmin'):
            # First, compute threshold mass vs. redshift
            if self.pf['pop_Mmin'] is not None:
                self._Mmin = self.pf['pop_Mmin']
            else:
                Mvir = lambda z: self.halos.VirialMass(self.pf['pop_Tmin'], 
                    z, mu=self.pf['mu'])
                self._Mmin = np.array(map(Mvir, self.halos.z))

        return self._Mmin
        
    @property
    def eta(self):
        """
        Correction factor for Macc.
        """
        
        if not self.is_ham_model:
            raise AttributeError('eta is a HAM thing!')
                
        # Prepare to compute eta
        self._eta = np.zeros_like(self.halos.z)
        
        for i, z in enumerate(self.halos.z):
            
            # eta = rhs / lhs
            
            logMmin = np.log10(self.Mmin[i])
            
            rhs = self.cosm.rho_b_z0 * cm_per_mpc**3
            rhs *= self.dfcolldt(z)
            
            Macc = self.Macc(z, self.halos.M) * g_per_msun / s_per_yr
            
            j = np.argmin(np.abs(self.Mmin[i] - self.halos.M))
            
            lhs = np.trapz(Macc[j:] * self.halos.dndm[i,j:])
            
            self._eta[i] = rhs / lhs                
                
        return self._eta
        
    @property
    def fstar(self):    
        """
        Compute the mass- and redshift-dependent star-formation efficiency.
        """
        if not self.is_ham_model:
            return self.pf['pop_fstar']
            
        # Otherwise, we're doing an abundance match!
        
        if not hasattr(self, '_fstar'):
            kappa_UV = 1. / self.yield_per_sfr
            
            Marr = self._Marr = np.logspace(8, 13, 12)
            
            self._fstar = np.zeros([len(self.constraints['z']), len(Marr)])
                
            for i, z in enumerate(self.constraints['z']):
                
                alpha = self.constraints['alpha'][i]
                L_star = self.constraints['L_star'][i]
                phi_star = self.constraints['phi_star'][i]            
                
                self.halos.MF.update(z=z)
                
                for j, M in enumerate(Marr):
                    
                    if M < self.Mmin[i]: 
                        continue
                    
                    Macc = self.Macc(z, M)
                    
                    # Minimum luminosity as a function of minimum mass
                    LofM = lambda fstar: fstar * Macc * self.eta[i] / kappa_UV    
                    
                    # Number of halos at masses > M
                    int_nMh = np.interp(M, self.halos.M, self.halos.MF.ngtm)
                    
                    def to_min(fstar):
                        Lmin = LofM(fstar[0])

                        if Lmin < 0:
                            return np.inf
                        
                        #integr = lambda x: x**alpha * np.exp(-x)
                        #int_phiL = quad(integr, Lmin / L_star, np.inf)[0]
                        int_phiL = self._schecter_integral_inf(Lmin / L_star, 
                            alpha)
                             
                        int_phiL *= phi_star
                                
                        return abs(int_phiL - int_nMh)
                    
                    fast = fsolve(to_min, 0.1, factor=0.1, maxfev=30)[0]

                    self._fstar[i,j] = fast
                    
        return self._fstar  
                  
    def _schecter_integral_inf(self, xmin, alpha):
        """
        Integral of the luminosity function over some interval (in luminosity).
    
        Parameters
        ----------
        xmin : int, float
            Lower limit of integration, (Lmin / Lstar)
        alpha : int, float
            Faint-end slope
        """
    
        return mpmath.gammainc(alpha + 1., xmin, np.inf)
        
    #def _SchecterFunction(self, L):
    #    """
    #    Schecter function for, e.g., the galaxy luminosity function.
    #    
    #    Parameters
    #    ----------
    #    L : float
    #    
    #    """
    #    
    #    return self.phi0 * (L / self.Lstar)**self.pf['lf_slope'] \
    #        * np.exp(-L / self.Lstar)
            
    def _DoublePowerLaw(self, L):
        """
        Double power-law function for, e.g., the quasar luminosity function.
        
        Parameters
        ----------
        L : float
        
        """
        return self.pf['lf_norm'] * ((L / self.Lstar)**self.pf['lf_gamma1'] \
            + (L / self.Lstar)**self.pf['lf_gamma1'])**-1.
                

    
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

        elif self.is_ham_model:
            # Only know LF at a few redshifts...
            pass

        elif self.is_hod_model:
            
            self.halos.MF.update(z=z)
            dndm = self._dndm = self.halos.MF.dndm.copy() / self.cosm.h70**4
            fstar_of_m = self.fstar(M=self.halos.M)
            
            integrand = dndm * fstar_of_m * self.halos.M
            
            # Msun / cMpc**3
            integral = simps(dndm, x=self.halos.M)
            
            tdyn = s_per_yr * 1e6
        else:
            raise NotImplemented('need help w/ this model!')    
            
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

    def _get_energy_per_photon(self, Emin, Emax):
        # Should this go in Population
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
            
        if (Emin, Emax) in self._eV_per_phot:
            return self._eV_per_phot[(Emin, Emax)]
        
        if Emin < self.pf['pop_Emin']:
            print "WARNING: Emin < pop_Emin"
        if Emax > self.pf['pop_Emax']:
            print "WARNING: Emax > pop_Emax"    
        
        integrand = lambda E: self.src.Spectrum(E) * E
        Eavg = quad(integrand, Emin, Emax)[0]
        
        self._eV_per_phot[(Emin, Emax)] = Eavg 
        
        return Eavg 

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
        elif self.is_halo_model:
            if self.halo_model == 'hod':

                
                self.halos.MF.update(z=z)
                dndm = self._dndm = self.halos.MF.dndm.copy() / self.cosm.h70**4
                fstar_of_m = self.fstar(M=self.halos.M)
                 
                integrand = dndm * fstar_of_m * self.halos.M
                
                # Apply mass cut
                if self.pf['pop_Mmin'] is not None:
                    iM = np.argmin(np.abs(self.halos.M - self.pf['pop_Mmin']))
                else:
                    iM = 0

                # Msun / cMpc**3
                integral = simps(dndm[iM:], x=self.halos.M[iM:])
                
                tdyn = s_per_myr * self.pf['pop_tSF']

                return self.cosm.fbaryon * integral / rho_cgs / tdyn
                
            elif self.halo_model == 'clf':
                raise NotImplemented('havent implemented CLF yet')    
                    
        else:
            raise NotImplemented('dunno how to model the SFRD!')
    
        return sfrd                           

    #@property
    #def _mdep_fstar(self):
    #    if not hasattr(self, '__mdep_fstar'):
    #        self.__mdep_fstar = type(self.pf['pop_fstar']) is FunctionType
    #    return self.__mdep_fstar
    #
    #def fstar(self, M=None, z=None):
    #    if self.is_fcoll_model or (not self._mdep_fstar):
    #        return self.pf['pop_fstar']
    #
    #    return self.pf['pop_fstar'](M, z)
            
    def Emissivity(self, z, E=None, Emin=None, Emax=None):
        """
        Compute the emissivity of this population as a function of redshift
        and rest-frame photon energy [eV].
        
        Parameters
        ----------
        z : int, float
        
        Returns
        -------
        Emissivity in units of erg / s / c-cm**3 [/ eV]
        
        """
        
        if self.is_fcoll_model or self.pf['pop_sfrd'] is not None:            
            rhoL = self.SFRD(z) * self.yield_per_sfr
        else:
            raise NotImplemented('help')    
                    
        # Convert from reference band to arbitrary band
        rhoL *= self._convert_band(Emin, Emax)
        
        if Emax > 13.6 and Emin < self.pf['pop_Emin_xray']:
            rhoL *= self.pf['pop_fesc']

        if E is not None:
            return rhoL * self.src.Spectrum(E)
        else:
            return rhoL    
    
    def NumberEmissivity(self, z, E=None, Emin=None, Emax=None):
        return self.Emissivity(z, E, Emin, Emax) / (E * erg_per_ev)


        

              
        