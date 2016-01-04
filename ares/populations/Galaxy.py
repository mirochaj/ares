"""

GalaxyPopulation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat May 23 12:13:03 CDT 2015

Description: 

"""

import inspect
import numpy as np
from ..util import read_lit
import os, pickle, inspect, re
import matplotlib.pyplot as pl
from types import FunctionType
from ..physics import Cosmology
from .Halo import HaloPopulation
from .Population import Population
from collections import namedtuple
from ..sources.Source import Source
from ..sources import Star, BlackHole
from ..util.PrintInfo import print_pop
from scipy.interpolate import interp1d
from scipy.integrate import quad, simps
from ..util.AbundanceMatching import HAM
from scipy.optimize import fsolve, fmin, curve_fit
from scipy.special import gamma, gammainc, gammaincc
from ..util.BlobFactory import BlobFactory
from ..util import ParameterFile, MagnitudeSystem, ProgressBar
from ..physics.Constants import s_per_yr, g_per_msun, erg_per_ev, rhodot_cgs, \
    E_LyA, rho_cgs, s_per_myr, cm_per_mpc, h_p, c, ev_per_hz
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

ARES = os.getenv('ARES')    
log10 = np.log(10.)

def param_redshift(par):
    
    m = re.search(r"\[(\d+(\.\d*)?)\]", par)
    
    prefix = par.replace(m.group(0), '')

    return prefix, float(m.group(1))

lftypes = ['schecter', 'dpl']    

class LiteratureSource(Source):
    def __init__(self, **kwargs):
        self.pf = kwargs
        Source.__init__(self)

        _src = read_lit(self.pf['pop_sed'])

        if hasattr(_src, 'Spectrum'):
            if inspect.ismethod(_src.Spectrum) or \
                (type(_src.Spectrum) is FunctionType):
                self._Intensity = _src.Spectrum
            else:
                self._Intensity = _src.Spectrum(**self.pf['pop_kwargs'])
        else:
            if self.pf['pop_sed'] != 'leitherer1999':
                raise NotImplemented('help')
            
            self.pop = _src.StellarPopulation(**self.pf)
            
def normalize_sed(pop):
    """
    Convert yield to erg / g.
    """
    
    # Remove whitespace and convert everything to lower-case
    units = pop.pf['pop_yield_units'].replace(' ', '').lower()
                
    if units == 'erg/s/sfr':
        return pop.pf['pop_yield'] * s_per_yr / g_per_msun

    E1 = pop.pf['pop_EminNorm']
    E2 = pop.pf['pop_EmaxNorm']
    erg_per_phot = pop.src.AveragePhotonEnergy(E1, E2) * erg_per_ev
    
    energy_per_sfr = pop.pf['pop_yield']
    
    if units == 'photons/baryon':
        energy_per_sfr *= erg_per_phot / pop.cosm.g_per_baryon
    elif units == 'photons/msun':
        energy_per_sfr *= erg_per_phot / g_per_msun
    elif units == 'photons/s/sfr':
        energy_per_sfr *= erg_per_phot * s_per_yr / g_per_msun   
    else:
        raise ValueError('Unrecognized yield units: %s' % units)

    return energy_per_sfr

class GalaxyPopulation(HaloPopulation,BlobFactory):
    def __init__(self, **kwargs):
        """
        Initializes a GalaxyPopulation object (duh).
        """

        # This is basically just initializing an instance of the cosmology
        # class. Also creates the parameter file attribute ``pf``.
        HaloPopulation.__init__(self, **kwargs)
        #self.pf.update(**kwargs)
        
        self._eV_per_phot = {}
        self._conversion_factors = {}

    @property
    def is_lya_src(self):
        if not hasattr(self, '_is_lya_src'):
            self._is_lya_src = \
                (self.pf['pop_Emin'] <= E_LyA <= self.pf['pop_Emax']) \
                and self.pf['pop_lya_src']

        return self._is_lya_src
    
    @property
    def _Source(self):
        if not hasattr(self, '_Source_'):
            if self.pf['pop_sed'] == 'bb':
                self._Source_ = Star
            elif self.pf['pop_sed'] in ['pl', 'mcd', 'simpl']:
                self._Source_ = BlackHole
            else:
                self._Source_ = LiteratureSource

        return self._Source_

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
                self._src_kwargs.update(self.pf['pop_kwargs'])
        
        return self._src_kwargs

    @property
    def src(self):
        if not hasattr(self, '_src'):
            self._src = self._Source(**self.src_kwargs)
                    
        return self._src

    @property
    def yield_per_sfr(self):
        if not hasattr(self, '_yield_per_sfr'):
            if isinstance(self.src, LiteratureSource):
                s99 = self.src.pop
                self._yield_per_sfr = s99.yield_per_sfr(*self.reference_band)
            else:
                self._yield_per_sfr = normalize_sed(self)
            
        return self._yield_per_sfr
        
    @property
    def Nion(self):
        if not hasattr(self, '_Nion'):
            pass
        
        return self._Nion

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
    def is_user_fstar(self):
        return type(self.pf['pop_fstar']) == FunctionType
        
    @property
    def rhoL_from_sfrd(self):
        if not hasattr(self, '_rhoL_from_sfrd'):
            self._rhoL_from_sfrd = self.is_fcoll_model \
                or self.is_ham_model or self.pf['pop_sfrd'] is not None
                
        return self._rhoL_from_sfrd
    
    @property
    def ham(self):
        if not hasattr(self, '_ham'):
            self._ham = HAM(galaxy=self)
        return self._ham
        
    @property
    def sed_tab(self):
        if not hasattr(self, '_sed_tab'):
            if self.pf['pop_sed'] == 'leitherer1999':
                self._sed_tab = True
            else:
                self._sed_tab = False
        return self._sed_tab

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
                
            # If tabulated, do things differently
            if self.sed_tab:
                factor = self.src.pop.yield_per_sfr(Emin, Emax) \
                    / self.src.pop.yield_per_sfr(*self.reference_band)
            else:
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
        
        if self.sed_tab:
            Eavg = self.src.pop.eV_per_phot(Emin, Emax)
        else:
            integrand = lambda E: self.src.Spectrum(E) * E
            Eavg = quad(integrand, Emin, Emax)[0]
        
        self._eV_per_phot[(Emin, Emax)] = Eavg 
        
        return Eavg 

    @property
    def _sfrd(self):
        if not hasattr(self, '_sfrd_'):
            if self.pf['pop_sfrd'] is None:
                self._sfrd_ = None
            elif type(self.pf['pop_sfrd']) is FunctionType:
                self._sfrd_ = self.pf['pop_sfrd']
            elif inspect.ismethod(self.pf['pop_sfrd']):
                self._sfrd_ = self.pf['pop_sfrd']
            else:
                tmp = read_lit(self.pf['pop_sfrd'])
                self._sfrd_ = lambda z: tmp.SFRD(z, **self.pf['pop_kwargs'])
        
        return self._sfrd_
    
    @property
    def _lf(self):
        if not hasattr(self, '_lf_'):
            if self.pf['pop_rhoL'] is None and self.pf['pop_lf'] is None:
                self._lf_ = None
            elif type(self.pf['pop_rhoL']) is FunctionType:
                self._lf_ = self.pf['pop_rhoL']
            elif type(self.pf['pop_lf']) is FunctionType:
                self._lf_ = self.pf['pop_lf']  
            else:
                for key in ['pop_rhoL', 'pop_lf']:
                    if self.pf[key] is None:
                        continue
                        
                    tmp = read_lit(self.pf[key])
                    self._lf_ = lambda L, z: tmp.LuminosityFunction(L, z=z,
                        **self.pf['pop_kwargs'])

                    break

        return self._lf_
        
    def LuminosityFunction(self, L=None, M=None, z=None, Emin=None, Emax=None):
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
            assert z in self.ham.redshifts
        
            if L is None:
        
                Mst = self.pf['pop_lf_Mstar[%g]' % z]
                pst = self.pf['pop_lf_pstar[%g]' % z]
                a = self.pf['pop_lf_alpha[%g]' % z]
        
                phi_of_M = 0.4 * np.log(10) * pst \
                    * (10**(0.4 * (Mst - M)))**(1. + a) \
                    * np.exp(-10**(0.4 * (Mst - M)))
        
                return phi_of_M
        
            else:
                raise NotImplemented('help')
        
        elif self.is_hod_model:
        
            self.halos.MF.update(z=z)
            dndm = self._dndm = self.halos.MF.dndm.copy() / self.cosm.h70**4
            fstar_of_m = self.fstar(z=z, M=self.halos.M)
        
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
            if self.pf['pop_sfrd_units'].lower() == 'g/s/cm^3':
                return self._sfrd(z)
            elif self.pf['pop_sfrd_units'].lower() == 'msun/yr/mpc^3':
                return self._sfrd(z) / rhodot_cgs
            else:
                raise NotImplemented('Unrecognized SFRD units!')
    
        # Most often: use fcoll model
        if self.is_fcoll_model:
           
            # SFRD computed via fcoll parameterization
            sfrd = self.pf['pop_fstar'] * self.cosm.rho_b_z0 * self.dfcolldt(z)
            
            if sfrd < 0:
                negative_SFRD(z, self.pf['pop_Tmin'], self.pf['pop_fstar'], 
                    self.dfcolldz(z) / self.cosm.dtdz(z), sfrd)
                sys.exit(1)
        elif self.is_ham_model:
            return self.ham.SFRD(z)
                
        #elif self.is_halo_model:
        #    if self.halo_model == 'hod':
        #
        #        
        #        self.halos.MF.update(z=z)
        #        dndm = self._dndm = self.halos.MF.dndm.copy() / self.cosm.h70**3
        #        fstar_of_m = self.fstar(M=self.halos.M)
        #         
        #        integrand = dndm * fstar_of_m * self.halos.M
        #        
        #        # Apply mass cut
        #        if self.pf['pop_Mmin'] is not None:
        #            iM = np.argmin(np.abs(self.halos.M - self.pf['pop_Mmin']))
        #        else:
        #            iM = 0
        #
        #        # Msun / cMpc**3
        #        integral = simps(dndm[iM:], x=self.halos.M[iM:])
        #        
        #        tdyn = s_per_myr * self.pf['pop_tSF']
        #
        #        return self.cosm.fbaryon * integral / rho_cgs / tdyn
        #        
        #    elif self.halo_model == 'clf':
        #        raise NotImplemented('havent implemented CLF yet')    
                    
        else:
            raise NotImplemented('dunno how to model the SFRD!')
    
        return sfrd                           
            
    @property
    def reference_band(self):
        if not hasattr(self, '_reference_band'):
            self._reference_band = \
                (self.pf['pop_EminNorm'], self.pf['pop_EmaxNorm'])
        return self._reference_band
            
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

        # This assumes we're interested in the (EminNorm, EmaxNorm) band
        if self.rhoL_from_sfrd:
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

    def save(self, prefix, clobber=False):
        """
        Output population-specific data to disk.
        
        Parameters
        ----------
        prefix : str
            
        """
        
        fn = '%s.ham_coeff.pkl'
        
        if os.path.exists(fn) and (not clobber):
            raise IOError('%s exists! Set clobber=True to overwrite.' % fn)
        
        with open(fn, 'wb') as f:
            data = (self.constraints, self._fstar_coeff)
            pickle.dump(data, f)
            
        print "Wrote %s." % fn
        

              
