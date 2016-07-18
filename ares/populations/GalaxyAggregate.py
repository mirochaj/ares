"""

GalaxyAggregate.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat May 23 12:13:03 CDT 2015

Description: 

"""

import sys
import numpy as np
from ..util import read_lit
import os, pickle, inspect, re
from types import FunctionType
from ..physics import Cosmology
from .Halo import HaloPopulation
from .Population import Population
from collections import namedtuple
from ..sources.Source import Source
from ..sources import Star, BlackHole
from scipy.interpolate import interp1d
from scipy.integrate import quad, simps
from ..util.Warnings import negative_SFRD
from .SynthesisModel import SynthesisModel
from ..util.ParameterFile import get_php_pars
from scipy.optimize import fsolve, fmin, curve_fit
from scipy.special import gamma, gammainc, gammaincc
from ..phenom.HaloProperty import ParameterizedHaloProperty
from ..util import ParameterFile, MagnitudeSystem, ProgressBar
from ..physics.Constants import s_per_yr, g_per_msun, erg_per_ev, rhodot_cgs, \
    E_LyA, rho_cgs, s_per_myr, cm_per_mpc, h_p, c, ev_per_hz, E_LL
from ..util.SetDefaultParameterValues import StellarParameters, \
    BlackHoleParameters

def normalize_sed(pop):
    """
    Convert yield to erg / g.
    """
    
    # In this case, we're just using Nlw, Nion, etc.
    if not pop.pf['pop_sed_model']:
        return 1.0
    
    E1 = pop.pf['pop_EminNorm']
    E2 = pop.pf['pop_EmaxNorm']
    
    if pop.pf['pop_yield_Z_index'] is not None:
        Zfactor = (pop.pf['pop_Z'] / 0.02)**pop.pf['pop_yield_Z_index']
    else:
        Zfactor = 1.
            
    if pop.pf['pop_yield'] == 'from_sed':
        return pop.src.yield_per_sfr(E1, E2)
    else:    
        # Remove whitespace and convert everything to lower-case
        units = pop.pf['pop_yield_units'].replace(' ', '').lower()
        if units == 'erg/s/sfr':
            return Zfactor * pop.pf['pop_yield'] * s_per_yr / g_per_msun

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

    return energy_per_sfr * Zfactor

class GalaxyAggregate(HaloPopulation):
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

        if not self.pf['pop_sed_model']:
            pass

    @property
    def id_num(self):
        if not hasattr(self, '_id_num'):
            self._id_num = None
        return self._id_num
        
    @id_num.setter
    def id_num(self, value):
        self._id_num = int(value)
        
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
            elif self.pf['pop_sed'] is None:
                self._Source_ = None
            else:
                self._Source_ = SynthesisModel

        return self._Source_

    @property
    def src_kwargs(self):
        """
        Dictionary of kwargs to pass on to an ares.source instance.
        
        This is basically just converting pop_* parameters to source_* 
        parameters.
        
        """
        if not hasattr(self, '_src_kwargs'):
            
            if self._Source is None:
                self._src_kwargs = {}
                return {}
            
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
            if self.pf['pop_psm_instance'] is not None:
                self._src = self.pf['pop_psm_instance']
            elif self._Source is not None:
                self._src = self._Source(**self.src_kwargs)
            else:
                self._src = None

        return self._src

    @property
    def yield_per_sfr(self):
        if not hasattr(self, '_yield_per_sfr'):
            self._yield_per_sfr = normalize_sed(self)
            
        return self._yield_per_sfr

    @property
    def is_fcoll_model(self):
        return self.pf['pop_sfr_model'].lower() == 'fcoll'
    
    @property
    def is_user_sfrd(self):
        return (self.pf['pop_sfr_model'].lower() == 'sfrd-func')
            
    @property
    def is_link_sfrd(self):
        if re.search('link', self.pf['pop_sfr_model']):
            return True
        return False        
        
    @property
    def is_user_sfe(self):
        return type(self.pf['pop_sfr_model']) == 'sfe-func'
        
    @property
    def sed_tab(self):
        if not hasattr(self, '_sed_tab'):
            if self.pf['pop_sed'] in ['leitherer1999', 'eldridge2009']:
                self._sed_tab = True
            else:
                self._sed_tab = False
        return self._sed_tab

    #def _sfrd_func(self, z):
    #    # This is a cheat so that the SFRD spline isn't constructed
    #    # until CALLED. Used only for tunneling (see `pop_tunnel` parameter). 
    #    
    #    return self.SFRD(z)    

    @property
    def _sfrd(self):
        if not hasattr(self, '_sfrd_'):
            if self.pf['pop_sfrd'] is None:
                self._sfrd_ = None
            elif type(self.pf['pop_sfrd']) is FunctionType:
                self._sfrd_ = self.pf['pop_sfrd']
            elif inspect.ismethod(self.pf['pop_sfrd']):
                self._sfrd_ = self.pf['pop_sfrd']
            elif isinstance(self.pf['pop_sfrd'], interp1d):
                self._sfrd_ = self.pf['pop_sfrd']  
            elif self.pf['pop_sfrd'][0:3] == 'php':
                pars = get_php_pars(self.pf['pop_sfrd'], self.pf)
                self._sfrd_ = ParameterizedHaloProperty(**pars)    
            else:
                tmp = read_lit(self.pf['pop_sfrd'])
                self._sfrd_ = lambda z: tmp.SFRD(z, **self.pf['pop_kwargs'])
        
        return self._sfrd_
        
    @_sfrd.setter
    def _sfrd(self, value):
        self._sfrd_ = value
            
    def _sfrd_func(self, z):
        # This is a cheat so that the SFRD spline isn't constructed
        # until CALLED. Used only for tunneling (see `pop_tunnel` parameter). 
        return self.SFRD(z)
    
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
        
        # SFRD given by some function  
        if self.is_link_sfrd:    
            # Already in the right units
            return self._sfrd(z)  
        elif self.is_user_sfrd:
            if self.pf['pop_sfrd_units'] == 'internal':
                return self._sfrd(z)
            else:
                return self._sfrd(z) / rhodot_cgs
                
        if (not self.is_fcoll_model) and (not self.is_user_sfe):
            raise ValueError('Must be an fcoll model!')
                   
        # SFRD computed via fcoll parameterization
        sfrd = self.pf['pop_fstar'] * self.cosm.rho_b_z0 * self.dfcolldt(z)
        
        if sfrd < 0:
            negative_SFRD(z, self.pf['pop_Tmin'], self.pf['pop_fstar'], 
                self.dfcolldz(z) / self.cosm.dtdz(z), sfrd)
            sys.exit(1)

        return sfrd                           
            
    @property
    def reference_band(self):
        if not hasattr(self, '_reference_band'):
            self._reference_band = \
                (self.pf['pop_EminNorm'], self.pf['pop_EmaxNorm'])
        return self._reference_band
            
    @property
    def model(self):
        return self.pf['pop_model']
    
    @property
    def is_lya_src(self):
        if not hasattr(self, '_is_lya_src'):
            if self.pf['pop_sed_model']:
                self._is_lya_src = \
                    (self.pf['pop_Emin'] <= E_LyA <= self.pf['pop_Emax']) \
                    and self.pf['pop_lya_src']
            else:
                return self.pf['pop_lya_src']
    
        return self._is_lya_src
    
    @property
    def is_uv_src(self):
        if not hasattr(self, '_is_uv_src'):
            if self.pf['pop_sed_model']:
                self._is_uv_src = \
                    (self.pf['pop_Emax'] > E_LL) \
                    and self.pf['pop_ion_src_cgm']
            else:
                self._is_uv_src = self.pf['pop_ion_src_cgm']        
    
        return self._is_uv_src    
    
    @property
    def is_xray_src(self):
        if not hasattr(self, '_is_xray_src'):
            if self.pf['pop_sed_model']:
                self._is_xray_src = \
                    (E_LL <= self.pf['pop_Emin']) \
                    and self.pf['pop_heat_src_igm']
            else:
                self._is_xray_src = self.pf['pop_heat_src_igm']        
    
        return self._is_xray_src    
    
    def _convert_band(self, Emin, Emax):
        """
        Convert from fractional luminosity in reference band to given bounds.
    
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
    
        # If we're here, it means we need to use some SED info
    
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
                factor = self.src.yield_per_sfr(Emin, Emax) \
                    / self.src.yield_per_sfr(*self.reference_band)
            else:
                factor = quad(self.src.Spectrum, Emin, Emax)[0] \
                    / quad(self.src.Spectrum, *self.reference_band)[0]
    
            self._conversion_factors[(Emin, Emax)] = factor
    
            return factor
    
        return 1.0
    
    def _get_energy_per_photon(self, Emin, Emax):
        """
        Compute the mean energy per photon in the provided band.
    
        If sed_tab or yield provided, will need Spectrum instance.
        Otherwise, assumes flat SED?
    
        Parameters
        ----------
        Emin : int, float
            Minimum photon energy to consider in eV.
        Emax : int, float
            Maximum photon energy to consider in eV.    
    
        Returns
        -------
        Photon energy in eV.
    
        """
               
        if not self.pf['pop_sed_model']:
            Eavg = np.mean([Emin, Emax])   
            self._eV_per_phot[(Emin, Emax)] = Eavg 
            return Eavg    
                
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
            Eavg = self.src.eV_per_phot(Emin, Emax)
        else:
            integrand = lambda E: self.src.Spectrum(E) * E
            Eavg = quad(integrand, Emin, Emax)[0] \
                / quad(self.src.Spectrum, Emin, Emax)[0]
        
        self._eV_per_phot[(Emin, Emax)] = Eavg 
        
        return Eavg    
        
    def Emissivity(self, z, E=None, Emin=None, Emax=None):
        """
        Compute the emissivity of this population as a function of redshift
        and rest-frame photon energy [eV].
        
        ..note:: If `E` is not supplied, this is a luminosity density in the
            (Emin, Emax) band.
        
        Parameters
        ----------
        z : int, float
        
        Returns
        -------
        Emissivity in units of erg / s / c-cm**3 [/ eV]

        """

        # This assumes we're interested in the (EminNorm, EmaxNorm) band
        rhoL = self.SFRD(z) * self.yield_per_sfr
                
        if not self.pf['pop_sed_model']:
            if (Emin, Emax) == (10.2, 13.6):
                return rhoL * self.pf['pop_Nlw'] * self.pf['pop_fesc_LW'] \
                    * self._get_energy_per_photon(Emin, Emax) * erg_per_ev \
                    / self.cosm.g_per_baryon 
            elif (Emin, Emax) == (13.6, 24.6):
                return rhoL * self.pf['pop_Nion'] * self.pf['pop_fesc'] \
                    * self._get_energy_per_photon(Emin, Emax) * erg_per_ev \
                    / self.cosm.g_per_baryon #/ (Emax - Emin)
            else:
                return rhoL * self.pf['pop_fX'] * self.pf['pop_cX'] \
                    / (g_per_msun / s_per_yr)
                                
        # Convert from reference band to arbitrary band
        rhoL *= self._convert_band(Emin, Emax)
       
        if Emax > 13.6 and Emin < self.pf['pop_Emin_xray']:
            rhoL *= self.pf['pop_fesc']
        elif Emax <= 13.6:
            rhoL *= self.pf['pop_fesc_LW']    
                            
        if E is not None:
            return rhoL * self.src.Spectrum(E)
        else:
            return rhoL

    def NumberEmissivity(self, z, E=None, Emin=None, Emax=None):
        return self.Emissivity(z, E, Emin, Emax) / (E * erg_per_ev)

    def LuminosityDensity(self, z, Emin=None, Emax=None):
        """
        Return the luminosity density in the (Emin, Emax) band.
    
        Parameters
        ----------
        z : int, flot
            Redshift of interest.
    
        Returns
        -------
        Luminosity density in erg / s / c-cm**3.
    
        """
    
        return self.Emissivity(z, Emin=Emin, Emax=Emax)
    
    def PhotonLuminosityDensity(self, z, Emin=None, Emax=None):
        """
        Return the photon luminosity density in the (Emin, Emax) band.
    
        Parameters
        ----------
        z : int, flot
            Redshift of interest.
    
        Returns
        -------
        Photon luminosity density in photons / s / c-cm**3.
    
        """
    
        rhoL = self.LuminosityDensity(z, Emin, Emax)
        eV_per_phot = self._get_energy_per_photon(Emin, Emax)
        
        return rhoL / (eV_per_phot * erg_per_ev)
    
    
    
    
          
