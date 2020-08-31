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
import os, inspect, re
from types import FunctionType
from .Halo import HaloPopulation
from collections import namedtuple
from ..util.Math import interp1d
from scipy.integrate import quad, simps
from ..util.Warnings import negative_SFRD
from ..util.ParameterFile import get_pq_pars, pop_id_num
from scipy.interpolate import interp1d as interp1d_scipy
from scipy.optimize import fsolve, fmin, curve_fit
from scipy.special import gamma, gammainc, gammaincc
from ..sources import Star, BlackHole, StarQS, SynthesisModel
from ..util import ParameterFile, ProgressBar
from ..phenom.ParameterizedQuantity import ParameterizedQuantity
from ..physics.Constants import s_per_yr, g_per_msun, erg_per_ev, rhodot_cgs, \
    E_LyA, rho_cgs, s_per_myr, cm_per_mpc, h_p, c, ev_per_hz, E_LL, k_B

_sed_tab_attributes = ['Nion', 'Nlw', 'rad_yield', 'L1600_per_sfr']
tiny_sfrd = 1e-15

class GalaxyAggregate(HaloPopulation):
    def __init__(self, **kwargs):
        """
        Initializes a GalaxyPopulation object (duh).
        """

        # This is basically just initializing an instance of the cosmology
        # class. Also creates the parameter file attribute ``pf``.
        HaloPopulation.__init__(self, **kwargs)
        #self.pf.update(**kwargs)

    @property
    def _sfrd(self):
        if not hasattr(self, '_sfrd_'):
            if self.pf['pop_sfrd'] is None:
                self._sfrd_ = None
            elif type(self.pf['pop_sfrd']) is FunctionType:
                self._sfrd_ = self.pf['pop_sfrd']
            elif inspect.ismethod(self.pf['pop_sfrd']):
                self._sfrd_ = self.pf['pop_sfrd']
            elif inspect.isclass(self.pf['pop_sfrd']):

                # Translate to parameter names used by external class
                pmap = self.pf['pop_user_pmap']

                # Need to be careful with pop ID numbers here.
                pars = {}
                for key in pmap:

                    val = pmap[key]

                    prefix, popid = pop_id_num(val)

                    if popid != self.id_num:
                        continue

                    pars[key] = self.pf[prefix]

                self._sfrd_ = self.pf['pop_sfrd'](**pars)
            elif type(self.pf['pop_sfrd']) is tuple:
                z, sfrd = self.pf['pop_sfrd']

                assert np.all(np.diff(z) > 0), "Redshifts must be ascending."

                if self.pf['pop_sfrd_units'] == 'internal':
                    sfrd[sfrd * rhodot_cgs <= tiny_sfrd] = tiny_sfrd / rhodot_cgs
                else:
                    sfrd[sfrd <= tiny_sfrd] = tiny_sfrd

                interp = interp1d(z, np.log(sfrd), kind=self.pf['pop_interp_sfrd'],
                    bounds_error=False, fill_value=-np.inf)

                self._sfrd_ = lambda **kw: np.exp(interp(kw['z']))
            elif isinstance(self.pf['pop_sfrd'], interp1d_scipy):
                self._sfrd_ = self.pf['pop_sfrd']
            elif self.pf['pop_sfrd'][0:2] == 'pq':
                pars = get_pq_pars(self.pf['pop_sfrd'], self.pf)
                self._sfrd_ = ParameterizedQuantity(**pars)
            else:
                tmp = read_lit(self.pf['pop_sfrd'], verbose=self.pf['verbose'])
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

        on = self.on(z)
        if not np.any(on):
            return z * on

        # SFRD given by some function
        if self.is_link_sfrd:
            # Already in the right units

            return self._sfrd(z) * on
        elif self.is_user_sfrd:
            if self.pf['pop_sfrd_units'] == 'internal':
                return self._sfrd(z=z) * on
            else:
                return self._sfrd(z=z) * on / rhodot_cgs

        if (not self.is_fcoll_model) and (not self.is_user_sfe):
            raise ValueError('Must be an fcoll model!')

        # SFRD computed via fcoll parameterization
        sfrd = self.pf['pop_fstar'] * self.cosm.rho_b_z0 * self.dfcolldt(z) * on

        if np.any(sfrd < 0):
            negative_SFRD(z, self.pf['pop_Tmin'], self.pf['pop_fstar'],
                self.dfcolldz(z) / self.cosm.dtdz(z), sfrd)
            sys.exit(1)

        return sfrd

    def _frd_func(self, z):
        return self.FRD(z)

    def FRD(self, z):
        """
        In the odd units of stars / cm^3 / s.
        """

        return self.SFRD(z) / self.pf['pop_mass'] / g_per_msun

    def Emissivity(self, z, E=None, Emin=None, Emax=None):
        """
        Compute the emissivity of this population as a function of redshift
        and rest-frame photon energy [eV].

        ..note:: If `E` is not supplied, this is a luminosity density in the
            (Emin, Emax) band. Otherwise, if E is supplied, or the SED is
            a delta function, the result is a monochromatic luminosity. If
            nothing is supplied, it's the luminosity density in the
            reference band.

        Parameters
        ----------
        z : int, float

        Returns
        -------
        Emissivity in units of erg / s / c-cm**3 [/ eV]

        """

        on = self.on(z)
        if not np.any(on):
            return z * on

        if self.pf['pop_sed_model'] and (Emin is not None) and (Emax is not None):
            if (Emin > self.pf['pop_Emax']):
                return 0.0
            if (Emax < self.pf['pop_Emin']):
                return 0.0

        # This assumes we're interested in the (EminNorm, EmaxNorm) band
        rhoL = self.SFRD(z) * self.yield_per_sfr * on

        ##
        # Models based on photons / baryon
        ##
        if not self.pf['pop_sed_model']:
            if (round(Emin, 1), round(Emax, 1)) == (10.2, 13.6):
                return rhoL * self.pf['pop_Nlw'] * self.pf['pop_fesc_LW'] \
                    * self._get_energy_per_photon(Emin, Emax) * erg_per_ev \
                    / self.cosm.g_per_baryon
            elif round(Emin, 1) == 13.6:
                return rhoL * self.pf['pop_Nion'] * self.pf['pop_fesc'] \
                    * self._get_energy_per_photon(Emin, Emax) * erg_per_ev \
                    / self.cosm.g_per_baryon #/ (Emax - Emin)
            else:
                return rhoL * self.pf['pop_fX'] * self.pf['pop_cX'] \
                    / (g_per_msun / s_per_yr)

        # Convert from reference band to arbitrary band
        rhoL *= self._convert_band(Emin, Emax)
        if (Emax is None) or (Emin is None):
            if self.pf['pop_reproc']:
                rhoL *= (1. - self.pf['pop_fesc']) * self.pf['pop_frep']
        elif Emax > E_LL and Emin < self.pf['pop_Emin_xray']:
            rhoL *= self.pf['pop_fesc']
        elif Emax <= E_LL:
            if self.pf['pop_reproc']:
                fesc = (1. - self.pf['pop_fesc']) * self.pf['pop_frep']
            elif Emin >= E_LyA:
                fesc = self.pf['pop_fesc_LW']
            else:
                fesc = 1.

            rhoL *= fesc

        if E is not None:
            return rhoL * self.src.Spectrum(E)
        else:
            return rhoL

    def NumberEmissivity(self, z, E=None, Emin=None, Emax=None):
        return self.Emissivity(z, E=E, Emin=Emin, Emax=Emax) / (E * erg_per_ev)

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

    def IonizingEfficiency(self, z):
        """
        This is not quite the standard definition of zeta. It has an extra
        factor of fbaryon since fstar is implemented throughout the rest of
        the code as an efficiency wrt baryonic inflow, not matter inflow.
        """
        zeta = self.pf['pop_Nion'] * self.pf['pop_fesc'] \
            * self.pf['pop_fstar'] #* self.cosm.fbaryon
        return zeta

    def HeatingEfficiency(self, z, fheat=0.2):
        ucorr = s_per_yr * self.cosm.g_per_b / g_per_msun
        zeta_x = fheat * self.pf['pop_rad_yield'] * ucorr \
               * (2. / 3. / k_B / self.pf['ps_saturated'] / self.cosm.TCMB(z))

        return zeta_x
