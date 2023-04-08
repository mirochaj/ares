"""

Population.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May 28 13:59:41 MDT 2015

Description:

"""

import re
import inspect
import numpy as np
from inspect import ismethod
from types import FunctionType
from ..physics import Cosmology
from ..util import ParameterFile
from scipy.integrate import quad
from ..obs import MagnitudeSystem
from functools import cached_property
from ..util.Misc import numeric_types
from ares.data import read as read_lit
from scipy.interpolate import interp1d
from ..util.PrintInfo import print_pop
from ..util.Warnings import no_lya_warning
from ..util.ParameterFile import get_pq_pars
from ..obs.DustCorrection import DustCorrection
from ..obs.DustExtinction import DustExtinction
from scipy.interpolate import interp1d as interp1d_scipy
from ..phenom.ParameterizedQuantity import ParameterizedQuantity
from ..sources import Star, BlackHole, StarQS, Toy, DeltaFunction, \
    SynthesisModel, SynthesisModelToy, SynthesisModelHybrid
from ..physics.Constants import g_per_msun, erg_per_ev, E_LyA, E_LL, s_per_yr, \
    ev_per_hz, h_p, cm_per_pc, c, cm_per_mpc

_multi_pop_error_msg = "Parameters for more than one population detected! "
_multi_pop_error_msg += "Population objects are by definition for single populations."
_multi_pop_error_msg += 'This population: '

from ..util.SetDefaultParameterValues import StellarParameters, \
    BlackHoleParameters, SynthesisParameters

_synthesis_models = ['leitherer1999', 'eldridge2009', 'eldridge2017',
    'bpass_v1', 'bpass_v2', 'starburst99']
_single_star_models = ['schaerer2002']
_sed_tabs = ['leitherer1999', 'eldridge2009', 'eldridge2017',
    'schaerer2002', 'hybrid',
    'bpass_v1', 'bpass_v2', 'starburst99', 'sps-toy']

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
        # In this case Emin, Emax, EminNorm, EmaxNorm are irrelevant
        E1 = pop.src.Emin
        E2 = pop.src.Emax
        return pop.src.get_rad_yield(E1, E2)
    else:
        # Remove whitespace and convert everything to lower-case
        units = pop.pf['pop_rad_yield_units'].replace(' ', '').lower()
        if units == 'erg/s/sfr':
            return Zfactor * pop.pf['pop_rad_yield'] * s_per_yr / g_per_msun

    energy_per_sfr = pop.pf['pop_rad_yield']

    # RARE: monochromatic normalization
    if units == 'erg/s/sfr/hz':
        assert pop.pf['pop_Enorm'] is not None
        energy_per_sfr *= s_per_yr / g_per_msun / ev_per_hz
    else:
        erg_per_phot = pop.src.get_avg_photon_energy(E1, E2) * erg_per_ev

    if units == 'photons/baryon':
        energy_per_sfr *= erg_per_phot / pop.cosm.g_per_baryon
    elif units == 'photons/msun':
        energy_per_sfr *= erg_per_phot / g_per_msun
    elif units == 'photons/s/sfr':
        energy_per_sfr *= erg_per_phot * s_per_yr / g_per_msun
    elif units == 'erg/s/sfr/hz':
        pass
    else:
        raise ValueError('Unrecognized yield units: {!s}'.format(units))

    return energy_per_sfr * Zfactor


class Population(object):
    def __init__(self, grid=None, cosm=None, **kwargs):

        self.pf = ParameterFile(**kwargs)

        assert self.pf.Npops == 1, _multi_pop_error_msg + str(self.id_num)

        self.grid = grid
        self._cosm_ = cosm

        self.zform = min(self.pf['pop_zform'], self.pf['first_light_redshift'])
        self.zdead = self.pf['pop_zdead']

        self._eV_per_phot = {}
        self._conversion_factors = {}

        stars = self.pf['pop_star_formation']
        bhs = self.pf['pop_bh_formation']
        assert stars + bhs <= 1, \
            "Populations can only form stars OR black holes."

        if self.is_src_neb and (not self.is_src_ion):
            raise ValueError('Including nebular line emission for non-ionizing source!')

    def run(self):
        # Avoid breaks in fitting (make it look like ares.simulation object)
        pass

    def _get_function(self, par):
        """
        Returns a function representation of input parameter `par`.

        For example, the user supplies the parameter `pop_dust_yield`. This
        routien figures out if that's a number, a function, or a string
        indicating a ParameterizedQuantity, and creates a callable function
        no matter what.
        """

        if not hasattr(self, '_get_{}'.format(par.strip('pop_'))):
            t = type(self.pf[par])

            if t in numeric_types:
                func = lambda **kwargs: self.pf[par]
            elif t == FunctionType:
                func = lambda **kwargs: self.pf[par](**kwargs)
            elif isinstance(self.pf[par], str) and self.pf[par].startswith('pq'):
                pars = get_pq_pars(self.pf[par], self.pf)
                ob = ParameterizedQuantity(**pars)
                func = lambda **kwargs: ob.__call__(**kwargs)
                setattr(self, '_obj_{}'.format(par.strip('pop_')), ob)
            else:
                raise NotImplementedError(f"Unrecognized option for `{par}`.")

            if f'{par}_inv' in self.pf:
                if self.pf[f'{par}_inv']:
                    func = lambda **kwargs: 1. - func(**kwargs)

            setattr(self, '_get_{}'.format(par.strip('pop_')), func)

        return getattr(self, '_get_{}'.format(par.strip('pop_')))

    @property
    def info(self):
        if not self.parameterized:
            try:
                print_pop(self)
            except AttributeError:
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
    def dust(self):
        if not hasattr(self, '_dust'):
            self._dust = DustCorrection(**self.pf)
        return self._dust

    @property
    def dustext(self):
        if not hasattr(self, '_dustext'):
            self._dustext = DustExtinction(**self.pf)
        return self._dustext

    @property
    def magsys(self):
        if not hasattr(self, '_magsys'):
            self._magsys = MagnitudeSystem(cosm=self.cosm, **self.pf)
        return self._magsys

    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            if self.grid is not None:
                self._cosm = grid.cosm
            elif self._cosm_ is not None:
                self._cosm = self._cosm_
            else:
                self._cosm = Cosmology(pf=self.pf, **self.pf)

        return self._cosm

    @property
    def zone(self):
        if not hasattr(self, '_zone'):
            if self.affects_cgm and (not self.affects_igm):
                self._zone = 'cgm'
            elif self.affects_igm and (not self.affects_cgm):
                self._zone = 'igm'
            elif (not self.affects_cgm) and (not self.affects_igm):
                self._zone = None
            else:
                s = "Populations should only affect one zone!"
                s += "In general, UV sources should have pop_ion_src_cgm=True "
                s += "while X-ray sources should have pop_*src_igm=True."
                raise ValueError(s)

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
    def is_metallicity_constant(self):
        if not hasattr(self, '_is_metallicity_constant'):
            self._is_metallicity_constant = not self.pf['pop_enrichment']
        return self._is_metallicity_constant

    @cached_property
    def is_sfe_constant(self):
        """ Is the SFE constant in redshift (at fixed halo mass)?"""

        _is_sfe_constant = 1

        for mass in [1e7, 1e8, 1e9, 1e10, 1e11, 1e12]:
            _is_sfe_constant *= self.get_fstar(z=10, Mh=mass) \
                             == self.get_fstar(z=20, Mh=mass)

        return bool(_is_sfe_constant)

    @cached_property
    def is_central_pop(self):
        return self.pf['pop_centrals']

    @cached_property
    def is_satellite_pop(self):
        return not self.is_central_pop

    @cached_property
    def is_star_forming(self):
        return not self.is_quiescent

    @cached_property
    def is_quiescent(self):
        return (self.pf['pop_sfr_model'] == 'smhm-func') and \
            (self.pf['pop_ssfr'] is None)

    @property
    def is_aging(self):
        return self.pf['pop_aging']

    @property
    def is_hod(self):
        """
        Is this a halo occupation model, i.e., does NOT require time
        integration?
        """
        return self.is_user_smhm

    @property
    def is_sam(self):
        """
        Is this a semi-analytic model, i.e., requires time integration?
        """
        return not self.is_hod

    @property
    def is_diffuse(self):
        return self.pf['pop_ihl'] is not None

    @property
    def is_src_radio(self):
        if not hasattr(self, '_is_src_radio'):
            if self.pf['pop_sed_model']:
                E21 = 1.4e9 * (h_p / erg_per_ev)
                self._is_src_radio = \
                    (self.pf['pop_Emin'] <= E21 <= self.pf['pop_Emax']) \
                    and self.pf['pop_radio_src']
            else:
                self._is_src_radio = self.pf['pop_radio_src']

        return self._is_src_radio

    @property
    def is_src_neb(self):
        return self.pf['pop_nebular'] and \
            (self.pf['pop_nebular_lines'] or self.pf['pop_nebular_continuum'])

    @property
    def is_src_fir(self):
        return False

    @property
    def is_src_radio_fl(self):
        return False

    @property
    def is_src_lya(self):
        if not hasattr(self, '_is_src_lya'):
            if self.pf['pop_sed_model']:
                self._is_src_lya = \
                    (self.pf['pop_Emin'] <= E_LyA <= self.pf['pop_Emax']) \
                    and self.pf['pop_lya_src']

                if self.pf['pop_lya_src'] and (not self._is_src_lya):
                    if abs(self.pf['pop_Emin'] - E_LyA) < 1.:
                        no_lya_warning(self)
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
                if self.pf['pop_lya_fl'] and self.pf['ps_include_lya']:
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
                if self.pf['pop_ion_fl'] and self.pf['ps_include_ion']:
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
                if self.pf['pop_temp_fl'] and self.pf['ps_include_temp']:
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
                    (self.pf['pop_Emin'] <= 11.2 <= self.pf['pop_Emax'])
            else:
                self._is_src_lw = False

        return self._is_src_lw

    @property
    def is_src_lw_fl(self):
        return False

    @cached_property
    def is_emissivity_reprocessed(self):
        """
        Does intrinsic SED of source populations get modified by, e.g., dust or
        nebular line emission?
        """
        return (self.pf['pop_nebular'] not in [0, 1]) or \
               (self.pf['pop_dustext_template'] is not None) or \
               (self.pf['pop_dust_yield'] is not None)

    @property
    def is_emissivity_separable(self):
        """
        Are the frequency and redshift-dependent components independent?
        """
        return True

    @cached_property
    def is_emissivity_bruteforce(self):
        return not self.pf['pop_emissivity_tricks']

    @property
    def is_emissivity_scalable(self):
        """
        Can we just determine a luminosity density by scaling the SFRD?

        The answer will be "no" for any population with halo-mass-dependent
        values for photon yields (per SFR), escape fractions, or spectra.
        """

        if not hasattr(self, '_is_emissivity_scalable'):

            if self.is_aging:
                self._is_emissivity_scalable = False
                return self._is_emissivity_scalable

            if self.is_quiescent:
                if type(self.pf['pop_age']) not in numeric_types:
                    self._is_emissivity_scalable = False
                    return self._is_emissivity_scalable

            if self.pf['pop_dustext_template'] is not None:
                if type(self.pf['pop_Av']) not in numeric_types:
                    self._is_emissivity_scalable = False
                    return self._is_emissivity_scalable

            self._is_emissivity_scalable = True

            # If an X-ray source and no PQs, we're scalable.
            if (self.pf.Npqs == 0) and (self.affects_igm) and \
               (not self.affects_cgm) and (not self.is_src_lya):
                return self._is_emissivity_scalable

            # The use of affects_cgm here is to indicate whether we're likely
            # to have an fesc that must be handled carefully.

            # At this stage, we need to set is_emissivity_scalable=False IFF:
            # (1) there are mass- or time-dependent radiative properties
            # (2) if there are wavelength-dependent escape fractions.
            # (3) maybe that's it?

            if (self.affects_cgm) and (not self.affects_igm):
                if self.pf['pop_fesc'] != self.pf['pop_fesc_LW']:
                    if self.pf['verbose']:
                        print("# WARNING: revisit scalability wrt fesc.")
                    #print("Not scalable cuz fesc pop={}".format(self.id_num))
            #        self._is_emissivity_scalable = False
            #        return False

            for par in self.pf.pqs:

                # Exceptions.
                if par not in ['pop_rad_yield', 'pop_fesc', 'pop_fesc_LW']:
                    continue
                #if (par == 'pop_fstar') or (not par.startswith('pop_')):
                #    continue

                # Could just skip parameters that start with pop_

                if type(self.pf[par]) is str:
                    self._is_emissivity_scalable = False
                    break

                for i in range(self.pf.Npqs):
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
            elif self.pf['pop_sed'] == 'delta':
                self._Source_ = DeltaFunction
            elif self.pf['pop_sed'] is None:
                self._Source_ = None
            elif self.pf['pop_sed'] in _synthesis_models:
                self._Source_ = SynthesisModel
            elif self.pf['pop_sed'] in ['hybrid']:
                self._Source_ = SynthesisModelHybrid
            elif self.pf['pop_sed'] in _single_star_models:
                self._Source_ = StarQS
            elif self.pf['pop_sed'] == 'sps-toy':
                self._Source_ = SynthesisModelToy
            elif type(self.pf['pop_sed']) is FunctionType or \
                 inspect.ismethod(self.pf['pop_sed']) or \
                 isinstance(self.pf['pop_sed'], interp1d_scipy):
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

            self._src_kwargs = dict(self.pf)
            if self._Source in [Star, StarQS, Toy, DeltaFunction]:
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

            elif self._Source in [SynthesisModel, SynthesisModelToy]:
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

        # Sometimes we need to know about cosmology...

        return self._src_kwargs


    @property
    def is_synthesis_model(self):
        if not hasattr(self, '_is_synthesis_model'):
            self._is_synthesis_model = \
                self.pf['pop_sed'] in _synthesis_models
        return self._is_synthesis_model

    @property
    def src(self):
        if not hasattr(self, '_src'):
            if self.pf['pop_psm_instance'] is not None:
                # Should phase this out for more generate approach below.
                self._src = self.pf['pop_psm_instance']
            elif self.pf['pop_src_instance'] is not None:
                self._src = self.pf['pop_src_instance']
            elif self._Source is not None:
                try:
                    self._src = self._Source(cosm=self.cosm, **self.src_kwargs)
                except TypeError:
                    # For litdata
                    self._src = self._Source
            else:
                self._src = None

        return self._src

    @property
    def _src_csfr(self):
        """
        Exact clone of `src` except forces source_ssp=False.
        """
        if not hasattr(self, '_src_csfr_'):
            if self.pf['pop_psm_instance'] is not None:
                # Should phase this out for more generate approach below.
                self._src_csfr_ = self.pf['pop_psm_instance']
            elif self.pf['pop_src_instance'] is not None:
                self._src_csfr_ = self.pf['pop_src_instance']
            elif self._Source is not None:
                try:
                    kw = self.src_kwargs.copy()
                    kw['source_ssp'] = False
                    self._src_csfr_ = self._Source(cosm=self.cosm, **kw)
                except TypeError:
                    # For litdata
                    self._src_csfr_ = self._Source
            else:
                self._src_csfr_ = None

        return self._src_csfr_

    @cached_property
    def tab_radiative_yield(self):
        """
        This is the conversion factor between star formation and luminosity.

        If this is a star-forming population, i.e., self.is_star_forming=True,
        then it is [erg/s/(Msun/yr)].

        If this is a quiescent population (self.is_quiescent=True), then the
        units are [erg/s/Msun] for the corresponding age (`pop_age`).
        """
        #if not hasattr(self, '_yield_per_sfr'):

        ## erg/g
        #self._yield_per_sfr = normalize_sed(self)

        if self.src.is_sed_tabular:
            E1 = self.src.Emin
            E2 = self.src.Emax
            return self.src.get_rad_yield(E1, E2)
        else:
            return normalize_sed(self)#self.pf['source_rad_yield']

    #@yield_per_sfr.setter
    #def yield_per_sfr(self, value):
    #    self._yield_per_sfr = value

    @property
    def is_fcoll_model(self):
        return self.pf['pop_sfr_model'].lower() == 'fcoll'

    @property
    def is_user_sfrd(self):
        return (self.pf['pop_sfr_model'].lower() in \
            ['sfrd-func', 'sfrd-tab', 'sfrd-class'])

    @property
    def is_link_sfrd(self):
        if re.search('link:sfrd', self.pf['pop_sfr_model']):
            return True
        # For BHs right now...
        elif self.pf['pop_frd'] is not None:
            if re.search('link:frd', self.pf['pop_frd']):
                return True
        return False

    @property
    def is_user_sfe(self):
        return self.pf['pop_sfr_model'] == 'sfe-func'

    @property
    def is_user_smhm(self):
        return self.pf['pop_sfr_model'] == 'smhm-func'

    @property
    def is_sed_tab(self):
        return self.src.is_sed_tabular

    @property
    def reference_band(self):
        if not hasattr(self, '_reference_band'):
            if self.is_sed_tab:
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

    #@property
    #def model(self):
    #    return self.pf['pop_model']

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

        if self.is_aging:
            raise AttributeError('This shouldn\'t happen! Aging of spectrum should be handled by pop itself.')

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

            if round(Emin, 2) < round(self.pf['pop_Emin'], 2):
                print(("WARNING: Emin ({0:.2f} eV) < pop_Emin ({1:.2f} eV) " +\
                    "[pop_id={2}]").format(Emin, self.pf['pop_Emin'],\
                    self.id_num))
            if Emax > self.pf['pop_Emax']:
                print(("WARNING: Emax ({0:.2f} eV) > pop_Emax ({1:.2f} eV) " +\
                    "[pop_id={2}]").format(Emax, self.pf['pop_Emax'],\
                    self.id_num))

            # If tabulated, do things differently
            if self.is_sed_tab:
                factor = self.src.get_rad_yield(Emin, Emax) \
                    / self.src.get_rad_yield(*self.reference_band)
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
            print(("WARNING: Emin ({0:.2g} eV) < pop_Emin ({1:.2g} eV) " +\
                "[pop_id={2}]").format(Emin, self.pf['pop_Emin'],\
                self.id_num))
        if Emax > self.pf['pop_Emax']:
            print(("WARNING: Emax ({0:.2g} eV) > pop_Emax ({1:.2g} eV) " +\
                "[pop_id={2}]").format(Emax, self.pf['pop_Emax'],\
                self.id_num))

        if self.is_sed_tab:
            Eavg = self.src.eV_per_phot(Emin, Emax)
        else:
            integrand = lambda E: self.src.Spectrum(E) * E
            Eavg = quad(integrand, Emin, Emax)[0] \
                / quad(self.src.Spectrum, Emin, Emax)[0]

        self._eV_per_phot[(Emin, Emax)] = Eavg

        return Eavg

    def on(self, z):
        if type(z) in [int, float, np.float64]:
            if (z > self.zform) or (z < self.zdead):
                return 0
            else:
                on = 1
        else:
            on = np.logical_and(z <= self.zform, z >= self.zdead)

        return on

    @property
    def Mmin(self):
        if not hasattr(self, '_Mmin'):
            self._Mmin = lambda z: \
                np.interp(z, self.halos.tab_z, self._tab_Mmin)

        return self._Mmin

    @property
    def _tab_Mmin(self):
        if not hasattr(self, '_tab_Mmin_'):
            # First, compute threshold mass vs. redshift
            if self.pf['feedback_LW_guesses'] is not None:
                guess = self._guess_Mmin()
                if guess is not None:
                    self._tab_Mmin = guess
                    return self._tab_Mmin_

            if self.pf['pop_Mmin'] is not None:
                if ismethod(self.pf['pop_Mmin']) or \
                   type(self.pf['pop_Mmin']) == FunctionType:
                    self._tab_Mmin_ = np.array([self.pf['pop_Mmin'](_z) \
                        for _z in self.halos.tab_z])
                elif type(self.pf['pop_Mmin']) is np.ndarray:
                    self._tab_Mmin_ = self.pf['pop_Mmin']
                    assert self._tab_Mmin.size == self.halos.tab_z.size
                else:
                    self._tab_Mmin_ = self.pf['pop_Mmin'] \
                        * np.ones(self.halos.tab_z.size)
            else:
                Mvir = lambda z: self.halos.VirialMass(z, self.pf['pop_Tmin'],
                    mu=self.pf['mu'])
                self._tab_Mmin_ = np.array([Mvir(_z) \
                    for _z in self.halos.tab_z])

            self._tab_Mmin_ = self._apply_lim(self._tab_Mmin_, 'min')

        return self._tab_Mmin_

    def _apply_lim(self, arr, s='min', zarr=None):
        """
        Adjust Mmin or Mmax so that Mmax > Mmin and/or obeys user-defined
        floor and ceiling.
        """
        out = None

        if zarr is None:
            zarr = self.halos.tab_z

        # Might need these if Mmin is being set dynamically
        if self.pf['pop_M%s_ceil' % s] is not None:
            out = np.minimum(arr, self.pf['pop_M%s_ceil'] % s)
        if self.pf['pop_M%s_floor' % s] is not None:
            out = np.maximum(arr, self.pf['pop_M%s_floor'] % s)
        if self.pf['pop_T%s_ceil' % s] is not None:
            _f = lambda z: self.halos.VirialMass(self.pf['pop_T%s_ceil' % s],
                z, mu=self.pf['mu'])
            _MofT = np.array(map(_f, zarr))
            out = np.minimum(arr, _MofT)
        if self.pf['pop_T%s_floor' % s] is not None:
            _f = lambda z: self.halos.VirialMass(self.pf['pop_T%s_floor' % s],
                z, mu=self.pf['mu'])
            _MofT = np.array(map(_f, zarr))
            out = np.maximum(arr, _MofT)

        if out is None:
            out = arr.copy()

        # Impose a physically-motivated floor to Mmin as a last resort,
        # by default this will be the Tegmark+ limit.
        if s == 'min':
            out = np.maximum(out, self._tab_Mmin_floor)

        return out

    @property
    def _tab_Mmin_floor(self):
        if not hasattr(self, '_tab_Mmin_floor_'):
            self._tab_Mmin_floor_ = self.halos.Mmin_floor(self.halos.tab_z)
        return self._tab_Mmin_floor_

    def get_mags_abs(self, z, mags):
        """
        Convert apparent magnitudes to absolute magnitudes.
        """
        d_pc = self.cosm.LuminosityDistance(z) / cm_per_pc
        return mags - 5 * np.log10(d_pc / 10.) + 2.5 * np.log10(1. + z)

    def get_mags_app(self, z, mags):
        """
        Convert absolute magnitudes to apparent magnitudes.
        """
        d_pc = self.cosm.LuminosityDistance(z) / cm_per_pc
        return mags + 5 * np.log10(d_pc / 10.) - 2.5 * np.log10(1. + z)

    def get_tab_emissivity(self, z, E):
        """
        Tabulate emissivity over photon energy and redshift.

        .. note :: This is not quite the emissivity -- it contains a factor of
            the Hubble parameter and has units of photons, not erg, so as to
            be more readily integrate-able in ares.solvers.UniformBackground.

        For a scalable emissivity, the tabulation is done for the emissivity
        in the (EminNorm, EmaxNorm) band because conversion to other bands
        can simply be applied after-the-fact. However, if the emissivity is
        NOT scalable, then it is tabulated separately in the (10.2, 13.6),
        (13.6, 24.6), and X-ray band.

        Parameters
        ----------
        z : np.ndarray
            Array of redshifts
        E : np.ndarray
            Array of photon energies [eV]
        pop : object
            Better be some kind of Galaxy population object.

        Returns
        -------
        A 2-D array, first axis corresponding to redshift, second axis for
        photon energy. Units are photons / s / Hz / (co-moving cm)^3.

        """

        Nz, Nf = len(z), len(E)

        Inu = np.zeros(Nf)

        # Special case: delta function SED! Can't normalize a-priori without
        # knowing binning, so we do it here.
        if self.src.is_delta:
            # This is a little weird. Trapezoidal integration doesn't make
            # sense for a delta function, but it's what happens later, so
            # insert a factor of a half now so we recover all the flux we
            # should.
            Inu[-1] = 1.
        else:
            for i in range(Nf):
                Inu[i] = self.src.Spectrum(E[i])

        # Convert to photon *number* (well, something proportional to it)
        Inu_hat = Inu / E

        # Now, redshift dependent parts
        epsilon = np.zeros([Nz, Nf])

        #if Nf == 1:
        #    return epsilon

        scalable = self.is_emissivity_scalable
        separable = self.is_emissivity_separable
        reprocessed = self.is_emissivity_reprocessed

        H = np.array([self.cosm.HubbleParameter(_z_) for _z_ in z])

        ##
        # Most general case: src.Spectrum does not contain all information.
        if self.is_emissivity_bruteforce or reprocessed:
            _waves = h_p * c * 1e8 / (E * erg_per_ev)

            _window = 2 * np.abs(np.diff(_waves))
            window = [round(_window[jj],0) for jj in range(Nf-1)]
            window.append(1)

            if self.is_quiescent:
                window = np.ones_like(_waves)

            #for jj in range(Nf):
            #    _window[jj]

            for ll in range(Nz):
                iz = np.argmin(np.abs(z[ll] - self.halos.tab_z))

                ok = np.logical_and(self.halos.tab_M >= self.get_Mmin(z[ll]),
                    self.halos.tab_M < self.get_Mmax(z[ll]))

                for jj in range(Nf):

                    # Skip: could put this in get_lum?
                    skip = False
                    if self.pf['pop_sed_null_except'] is not None:
                        for element in self.pf['pop_sed_null_except']:
                            lo, hi, keep_cont = element
                            if (not keep_cont) and not (lo <= _waves[jj] < hi):
                                skip = True
                                break

                    if skip:
                        continue

                    # [erg/s/Hz]
                    lum_v_Mh = self.get_lum(z[ll], wave=_waves[jj], raw=False,
                        window=window[jj] if window[jj] % 2 == 1 else window[jj]+1)

                    # Setup integrand over population [erg/s/Hz/cMpc^3]
                    integrand = lum_v_Mh * self.halos.tab_dndlnm[iz,:] \
                        * self.tab_focc[iz,:] * ok

                    # Integrate
                    _tot = np.trapz(integrand, x=np.log(self.halos.tab_M))
                    #_cumtot = cumtrapz(integrand, x=np.log(self.halos.tab_M),
                    #    initial=0.0)

                    #_tmp = _tot - \
                    #    np.interp(np.log(self._tab_Mmin[iz]),
                    #        np.log(self.halos.tab_M), _cumtot)

                    # Convert from luminosity in erg to photons
                    epsilon[ll,jj] = _tot / H[ll] / (E[jj] * erg_per_ev)

            epsilon /= cm_per_mpc**3

        elif scalable:
            Lbol = self.get_emissivity(z)

            # Only appropriate if dust extinction constant w/ Ms, z, etc
            if self.pf['pop_dustext_template'] is not None:
                wave = h_p * c * 1e8 / (E * erg_per_ev)
                T = self.dustext.get_T_lam(wave, Av=self.pf['pop_Av'])
            else:
                T = 1.

            for ll in range(Nz):
                epsilon[ll,:] = T * Inu_hat * Lbol[ll] * ev_per_hz / H[ll] \
                    / erg_per_ev

        else:

            # There is only a distinction here for computational
            # convenience, really. The LWB gets solved in much more detail
            # than the LyC or X-ray backgrounds, so it makes sense
            # to keep the different emissivity chunks separate.
            ct = 0
            for band in [(10.2, 13.6), (13.6, 24.6), None]:

                if band is not None:

                    if self.src.Emin > band[1]:
                        continue

                    if self.src.Emax < band[0]:
                        continue

                # Remind me of this distinction?
                if band is None:
                    b = self.full_band
                    fix = 1.

                    # Means we already generated the emissivity.
                    if ct > 0:
                        continue

                else:
                    b = band

                    # If aging population, is handled within the pop object.
                    if not self.is_aging:
                        fix = 1. / self._convert_band(*band)
                    else:
                        fix = 1.

                in_band = np.logical_and(E >= b[0], E <= b[1])

                # Shouldn't be any filled elements yet
                if np.any(epsilon[:,in_band==1] > 0):
                    raise ValueError("Non-zero elements already!")

                if not np.any(in_band):
                    continue

                ###
                # No need for spectral correction in this case, at least
                # in Lyman continuum. Treat LWB more carefully.
                if self.is_aging and band == (13.6, 24.6):
                    fix = 1. / Inu_hat[in_band==1]

                elif self.is_aging and band == (10.2, 13.6):

                    if self.pf['pop_synth_lwb_method'] == 0:
                        # No approximation: loop over energy below
                        raise NotImplemented('sorry dude')
                    elif self.pf['pop_synth_lwb_method'] == 1:
                        # Assume SED of continuousy-star-forming source.
                        Inu_hat_p = self._src_csfr.Spectrum(E[in_band==1]) \
                            / E[in_band==1]
                        fix = Inu_hat_p / Inu_hat[in_band==1][0]
                    else:
                        raise NotImplemented('sorry dude')
                ###

                # By definition, rho_L integrates to unity in (b[0], b[1]) band
                # BUT, Inu_hat is normalized in (EminNorm, EmaxNorm) band,
                # hence the 'fix'.

                for ll, redshift in enumerate(z):

                    if (redshift < self.pf['final_redshift']):
                        continue
                    if (redshift < self.zdead):
                        continue
                    if (redshift > self.zform):
                        continue
                    if redshift < self.pf['kill_redshift']:
                        continue
                    if redshift > self.pf['first_light_redshift']:
                        continue

                    # Use Emissivity here rather than rho_L because only
                    # GalaxyCohort objects will have a rho_L attribute.
                    epsilon[ll,in_band==1] = fix \
                        * self.get_emissivity(redshift, Emin=b[0], Emax=b[1]) \
                        * ev_per_hz * Inu_hat[in_band==1] / H[ll] / erg_per_ev

                ct += 1

        return epsilon
