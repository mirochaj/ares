"""

Population.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May 28 13:59:41 MDT 2015

Description: 

"""

import re
import numpy as np
from ..physics import Cosmology
from ..util import ParameterFile
from scipy.integrate import quad
from ..sources import Star, BlackHole, StarQS, SynthesisModel
from ..physics.Constants import g_per_msun, erg_per_ev, E_LyA, E_LL, s_per_yr, \
    ev_per_hz

_multi_pop_error_msg = "Parameters for more than one population detected! "
_multi_pop_error_msg += "Population objects are by definition for single populations."
_multi_pop_error_msg += 'This population: '

from ..util.SetDefaultParameterValues import StellarParameters, \
    BlackHoleParameters, SynthesisParameters
    
_synthesis_models = ['leitherer1999', 'eldridge2009']
_single_star_models = ['schaerer2002']
_sed_tabs = ['leitherer1999', 'eldridge2009', 'schaerer2002']

def normalize_sed(pop):
    """
    Convert yield to erg / g.
    """
        
    # In this case, we're just using Nlw, Nion, etc.
    if not pop.pf['pop_sed_model']:
        return 1.0
    
    E1 = pop.pf['pop_EminNorm']
    E2 = pop.pf['pop_EmaxNorm']

    # Deprecated? Just use ParameterizedQuantity now?
    if pop.pf['pop_rad_yield_Z_index'] is not None:
        Zfactor = (pop.pf['pop_Z'] / 0.02)**pop.pf['pop_rad_yield_Z_index']
    else:
        Zfactor = 1.
        
    if pop.pf['pop_rad_yield'] == 'from_sed':
        # In this case, the *Norm parameters are irrelevant
        E1 = pop.pf['pop_Emin']
        E2 = pop.pf['pop_Emax']
        return pop.src.rad_yield(E1, E2)
    else:    
        # Remove whitespace and convert everything to lower-case
        units = pop.pf['pop_rad_yield_units'].replace(' ', '').lower()
        if units.startswith('erg/s/sfr'):
            return Zfactor * pop.pf['pop_rad_yield'] * s_per_yr / g_per_msun

    erg_per_phot = pop.src.AveragePhotonEnergy(E1, E2) * erg_per_ev
    energy_per_sfr = pop.pf['pop_rad_yield']
        
    if units == 'photons/baryon':
        energy_per_sfr *= erg_per_phot / pop.cosm.g_per_baryon
    elif units == 'photons/msun':
        energy_per_sfr *= erg_per_phot / g_per_msun
    elif units == 'photons/s/sfr':
        energy_per_sfr *= erg_per_phot * s_per_yr / g_per_msun
    else:
        raise ValueError('Unrecognized yield units: {!s}'.format(units))

    return energy_per_sfr * Zfactor


class Population(object):
    def __init__(self, grid=None, **kwargs):

        # why is this necessary?
        if 'problem_type' in kwargs:
            del kwargs['problem_type']

        self.pf = ParameterFile(**kwargs)
        
        assert self.pf.Npops == 1, _multi_pop_error_msg + str(self.id_num)
        
        self.grid = grid

        self.zform = min(self.pf['pop_zform'], self.pf['first_light_redshift'])
        self.zdead = self.pf['pop_zdead']
        
        self._eV_per_phot = {}
        self._conversion_factors = {}
        
    def run(self):
        # Avoid breaks in fitting (make it look like ares.simulation object)
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
    def cosm(self):
        if not hasattr(self, '_cosm'):    
            if self.grid is None:
                self._cosm = Cosmology(**self.pf)
            else:
                self._cosm = grid.cosm
                
        return self._cosm
        
    @property
    def zone(self):
        if not hasattr(self, '_zone'):
            if self.affects_cgm and (not self.affects_igm):
                self._zone = 'cgm'
            elif self.affects_igm and (not self.affects_cgm):
                self._zone = 'igm'
            else:
                raise ValueError("Populations should only affect one zone!")
                
        return self._zone    
        
    @property
    def affects_cgm(self):
        if not hasattr(self, '_affects_cgm'):
            self._affects_cgm = self.is_src_ion_cgm 
        return self._affects_cgm
    
    @property
    def affects_igm(self):
        if not hasattr(self, '_affects_igm'):
            self._affects_igm = self.is_src_ion_igm or self.is_src_heat_igm
        return self._affects_igm    
    
    @property
    def is_src_lya(self):
        if not hasattr(self, '_is_src_lya'):
            if self.pf['pop_sed_model']:
                self._is_src_lya = \
                    (self.pf['pop_Emin'] <= 10.2 <= self.pf['pop_Emax']) \
                    and self.pf['pop_lya_src']
            else:
                self._is_src_lya = self.pf['pop_lya_src']
    
        return self._is_src_lya
    
    @property
    def is_src_lya_fl(self):
        if not hasattr(self, '_is_src_lya_fl'):
            self._is_src_lya_fl = False
            if not self.is_src_lya:
                pass
            else:
                if self.pf['pop_lya_fl'] and self.pf['include_lya_fl']:
                    self._is_src_lya_fl = True
    
        return self._is_src_lya_fl
    
    @property
    def is_src_ion_cgm(self):
        if not hasattr(self, '_is_src_ion_cgm'):
            if self.pf['pop_sed_model']:
                self._is_src_ion_cgm = \
                    (self.pf['pop_Emax'] > E_LL) \
                    and self.pf['pop_ion_src_cgm']
            else:
                self._is_src_ion_cgm = self.pf['pop_ion_src_cgm']        
    
        return self._is_src_ion_cgm
        
    @property
    def is_src_ion_igm(self):
        if not hasattr(self, '_is_src_ion_igm'):
            if self.pf['pop_sed_model']:
                self._is_src_ion_igm = \
                    (self.pf['pop_Emax'] > E_LL) \
                    and self.pf['pop_ion_src_igm']
            else:
                self._is_src_ion_igm = self.pf['pop_ion_src_igm']        
    
        return self._is_src_ion_igm
        
    @property
    def is_src_ion(self):
        if not hasattr(self, '_is_src_ion'):    
            self._is_src_ion = self.is_src_ion_cgm #or self.is_src_ion_igm
        return self._is_src_ion
        
    @property
    def is_src_ion_fl(self):
        if not hasattr(self, '_is_src_ion_fl'):
            self._is_src_ion_fl = False
            if not self.is_src_ion:
                pass
            else:
                if self.pf['pop_ion_fl'] and self.pf['include_ion_fl']:
                    self._is_src_ion_fl = True
    
        return self._is_src_ion_fl    
        
    @property
    def is_src_heat(self):
        return self.is_src_heat_igm
        
    @property
    def is_src_heat_igm(self):
        if not hasattr(self, '_is_src_heat_igm'):
            if self.pf['pop_sed_model']:
                self._is_src_heat_igm = \
                    (E_LL <= self.pf['pop_Emin']) \
                    and self.pf['pop_heat_src_igm']
            else:
                self._is_src_heat_igm = self.pf['pop_heat_src_igm']        
    
        return self._is_src_heat_igm
        
    @property
    def is_src_heat_fl(self):
        if not hasattr(self, '_is_src_heat_fl'):
            self._is_src_heat_fl = False
            if not self.is_src_heat:
                pass
            else:
                if self.pf['pop_temp_fl'] and self.pf['include_temp_fl']:
                    self._is_src_heat_fl = True
    
        return self._is_src_heat_fl
    
    @property
    def is_src_uv(self):
        # Delete this eventually but right now doing so will break stuff
        if not hasattr(self, '_is_src_uv'):
            if self.pf['pop_sed_model']:
                self._is_src_uv = \
                    (self.pf['pop_Emax'] > E_LL) \
                    and self.pf['pop_ion_src_cgm']
            else:
                self._is_src_uv = self.pf['pop_ion_src_cgm']        
    
        return self._is_src_uv
        
    @property
    def is_src_lya(self):
        if not hasattr(self, '_is_src_lya'):
            if self.pf['pop_sed_model']:
                self._is_src_lya = \
                    (self.pf['pop_Emin'] <= 10.2 <= self.pf['pop_Emax']) \
                    and self.pf['pop_lya_src']
            else:
                return self.pf['pop_lya_src']
    
        return self._is_src_lya
    
    @property
    def is_src_uv(self):
        if not hasattr(self, '_is_src_uv'):
            if self.pf['pop_sed_model']:
                self._is_src_uv = \
                    (self.pf['pop_Emax'] > E_LL) \
                    and self.pf['pop_ion_src_cgm']
            else:
                self._is_src_uv = self.pf['pop_ion_src_cgm']        
    
        return self._is_src_uv    
    
    @property
    def is_src_xray(self):
        if not hasattr(self, '_is_src_xray'):
            if self.pf['pop_sed_model']:
                self._is_src_xray = \
                    (E_LL <= self.pf['pop_Emin']) \
                    and self.pf['pop_heat_src_igm']
            else:
                self._is_src_xray = self.pf['pop_heat_src_igm']        
        
        return self._is_src_xray    

    @property
    def is_src_lw(self):
        if not hasattr(self, '_is_src_lw'):
            if not self.pf['radiative_transfer']:
                self._is_src_lw = False
            elif not self.pf['pop_lw_src']:
                self._is_src_lw = False
            elif self.pf['pop_sed_model']:
                self._is_src_lw = \
                    (self.pf['pop_Emin'] <= 11.2 <= self.pf['pop_Emax']) and \
                    (self.pf['pop_Emin'] <= E_LL <= self.pf['pop_Emax'])
            else:
                raise NotImplementedError('help')
    
        return self._is_src_lw    

    @property
    def is_emissivity_separable(self):
        """
        Are the frequency and redshift-dependent components independent?
        """
        return True
    
    @property
    def is_emissivity_scalable(self):
        """
        Can we just determine a luminosity density by scaling the SFRD?
    
        The answer will be "no" for any population with halo-mass-dependent
        values for photon yields (per SFR), escape fractions, or spectra.
        """

        if not hasattr(self, '_is_emissivity_scalable'):
            self._is_emissivity_scalable = True

            if self.pf.Npqs == 0:
                return self._is_emissivity_scalable

            for par in self.pf.pqs:
    
                # Exceptions. Ideally, exotic_heating_func wouldn't make it
                # to the population parameter files...
                if (par == 'pop_fstar') or (not par.startswith('pop_')):
                #if par in ['pop_fstar', 'exotic_heating_func', 'spin_temperature_floor']:
                    continue
                    
                # Could just skip parameters that start with pop_    
    
                if type(self.pf[par]) is str:
                    self._is_emissivity_scalable = False
                    break
    
                for i in xrange(self.pf.Npqs):
                    pn = '{0!s}[{1}]'.format(par,i)
                    if pn not in self.pf:
                        continue
    
                    if type(self.pf[pn]) is str:
                        self._is_emissivity_scalable = False
                        break
    
        return self._is_emissivity_scalable
    
    @property
    def _Source(self):
        if not hasattr(self, '_Source_'):
            if self.pf['pop_sed'] == 'bb':
                self._Source_ = Star
            elif self.pf['pop_sed'] in ['pl', 'mcd', 'simpl']:
                self._Source_ = BlackHole
            elif self.pf['pop_sed'] is None:
                self._Source_ = None
            elif self.pf['pop_sed'] in _synthesis_models:    
                self._Source_ = SynthesisModel
            elif self.pf['pop_sed'] in _single_star_models:
                self._Source_ = StarQS
            elif type(self.pf['pop_sed']) is FunctionType or \
                 inspect.ismethod(self.pf['pop_sed']) or \
                 isinstance(self.pf['pop_sed'], interp1d):
                 self._Source_ = BlackHole
            else:
                self._Source_ = read_lit(self.pf['pop_sed'], 
                    verbose=self.pf['verbose'])
    
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
            if self._Source in [Star, StarQS]:
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
    
            elif self._Source is SynthesisModel:
                bpars = SynthesisParameters()
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
                try:
                    self._src = self._Source(**self.src_kwargs)
                except TypeError:
                    # For litdata
                    self._src = self._Source
            else:
                self._src = None
    
        return self._src
    
    @property
    def yield_per_sfr(self):
        if not hasattr(self, '_yield_per_sfr'):
            self._yield_per_sfr = normalize_sed(self)
            
            # Correction: supplied normalization at monochromatic energy
            if self.pf['pop_Enorm'] is not None:
                self._yield_per_sfr = self._yield_per_sfr \
                    / self.src.Spectrum(self.pf['pop_Enorm']) \
                
                if self.pf['pop_rad_yield_units'].endswith('hz'):
                    self._yield_per_sfr /= ev_per_hz
            
    
        return self._yield_per_sfr
    
    @property
    def is_fcoll_model(self):
        return self.pf['pop_sfr_model'].lower() == 'fcoll'
    
    @property
    def is_user_sfrd(self):
        return (self.pf['pop_sfr_model'].lower() == 'sfrd-func')
    
    @property
    def is_link_sfrd(self):
        if re.search('link:sfrd', self.pf['pop_sfr_model']):
            return True
        return False  
    
    @property
    def is_user_sfe(self):
        return type(self.pf['pop_sfr_model']) == 'sfe-func'
    
    @property
    def sed_tab(self):
        if not hasattr(self, '_sed_tab'):
            if self.pf['pop_sed'] in _sed_tabs:
                self._sed_tab = True
            else:
                self._sed_tab = False
        return self._sed_tab
    
    @property
    def reference_band(self):
        if not hasattr(self, '_reference_band'):
            if self.sed_tab:
                self._reference_band = self.src.Emin, self.src.Emax
            else:
                self._reference_band = \
                    (self.pf['pop_EminNorm'], self.pf['pop_EmaxNorm'])
        return self._reference_band
    
    @property
    def full_band(self):
        if not hasattr(self, '_full_band'):
            self._full_band = (self.pf['pop_Emin'], self.pf['pop_Emax'])
        return self._full_band    
    
    @property
    def model(self):
        return self.pf['pop_model']
    
    def _convert_band(self, Emin, Emax):
        """
        Convert from fractional luminosity in reference band to given bounds.
    
        If limits are None, will use (pop_Emin, pop_Emax).
    
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
                print(("WARNING: Emin ({0:.2g} eV) < pop_Emin ({1:.2g} eV) " +\
                    "[pop_id={2}]").format(Emin, self.pf['pop_Emin'],\
                    self.id_num))
            if Emax > self.pf['pop_Emax']:
                print(("WARNING: Emax ({0:.2g} eV) > pop_Emax ({1:.2g} eV) " +\
                    "[pop_id={2}]").format(Emax, self.pf['pop_Emax'],\
                    self.id_num))
    
            # If tabulated, do things differently
            if self.sed_tab:
                factor = self.src.rad_yield(Emin, Emax) \
                    / self.src.rad_yield(*self.reference_band)
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
            print("WARNING: Emin < pop_Emin")
        if Emax > self.pf['pop_Emax']:
            print("WARNING: Emax > pop_Emax")
    
        if self.sed_tab:
            Eavg = self.src.eV_per_phot(Emin, Emax)
        else:
            integrand = lambda E: self.src.Spectrum(E) * E
            Eavg = quad(integrand, Emin, Emax)[0] \
                / quad(self.src.Spectrum, Emin, Emax)[0]
    
        self._eV_per_phot[(Emin, Emax)] = Eavg
    
        return Eavg    
    
    