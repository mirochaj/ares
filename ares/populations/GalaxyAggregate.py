"""

GalaxyAggregate.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat May 23 12:13:03 CDT 2015

Description:

"""

import sys
import numpy as np
from .Halo import HaloPopulation
from ..util.Warnings import negative_SFRD
from ..physics.Constants import s_per_yr, g_per_msun, erg_per_ev, rhodot_cgs, \
    E_LyA, s_per_myr, cm_per_mpc, c, E_LL, k_B

class GalaxyAggregate(HaloPopulation):
    def __init__(self, pf=None, **kwargs):
        """
        Initializes a GalaxyAggregate object.

        The defining feature of GalaxyAggregate models is that galaxy properties
        are not specified as a function of halo mass -- they may only be
        functions of redshift, hence the 'aggregate' designation, as we're
        averaging over the whole population at any given redshift.

        The most important parameter is `pop_sfr_model`. It should be either
        'fcoll', or the user should have provided `pop_sfrd` directly.
        """

        # This is basically just initializing an instance of the cosmology
        # class. Also creates the parameter file attribute ``pf``.
        HaloPopulation.__init__(self, pf=pf, **kwargs)

    def get_sfrd(self, z):
        """
        Compute the comoving star formation rate density (SFRD).

        Parameters
        ----------
        z : int, float, np.ndarray
            Redshift(s) of interest.

        Returns
        -------
        Co-moving star-formation rate density at redshift z in units of
        g s**-1 cm**-3.

        """

        on = self.on(z)
        if not np.any(on):
            return z * on

        # If we already setup a function, call it.
        # This will also cover the case where it has been linked to the SFRD
        # of another source population.
        if hasattr(self, '_get_sfrd'):
            return self._get_sfrd(z=z) * on

        # Check to see if supplied directly by user.
        if self.pf['pop_sfrd'] is not None:
            func = self._get_function('pop_sfrd')
            if func is not None:
                return func(z=z)

        # Sanity check.
        if (not self.is_fcoll_model) and (not self.is_user_sfe):
            raise ValueError('Must be an fcoll model!')

        # SFRD computed via fcoll parameterization
        sfrd = self.pf['pop_fstar'] * self.cosm.rho_b_z0 * self.dfcolldt(z) * on

        # At the moment, `sfrd` has cgs units. From version 1 onward, we'll use
        # Msun/cMpc^3/yr as the default internal unit.
        sfrd *= rhodot_cgs

        return sfrd

    def get_emissivity(self, z, x=None, band=None, units='eV'):
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
        Emissivity in units of erg / s / cMpc^3 [/ eV]

        """

        on = self.on(z)
        if not np.any(on):
            return z * on

        #from_band = (Emin is not None) and (Emax is not None)

        #if self.pf['pop_sed_model'] and from_band:
        #    if (Emin > self.src.Emax):
        #        return 0.0
        #    if (Emax < self.src.Emin):
        #        return 0.0

        if (x is None) and (band is None):
            band = self.src.Emin, self.src.Emax
            assert units.lower() == 'ev'

        ##
        # Models based on photons / baryon
        ##
        if self.pf['pop_sed'] is None:
            bname = self.src.get_band_name(x=x, band=band, units=units)

            # In this case, photon yields have been provided via parameters.
            # Just need to convert SFRD to baryons/s/cMpc^3
            if bname == 'LW':
                return (self.get_sfrd(z) * self.cosm.b_per_msun / s_per_yr) \
                    * self.pf['pop_Nlw'] * self.pf['pop_fesc_LW'] \
                    * self._get_energy_per_photon(band, units=units) * erg_per_ev
                #return rhoL * self.pf['pop_Nlw'] * self.pf['pop_fesc_LW'] \
                #    * self._get_energy_per_photon(Emin, Emax) * erg_per_ev \
                #    / self.cosm.g_per_baryon
            elif bname == 'LyC':
                return (self.get_sfrd(z) * self.cosm.b_per_msun / s_per_yr) \
                    * self.pf['pop_Nion'] * self.pf['pop_fesc'] \
                    * self._get_energy_per_photon(band, units=units) * erg_per_ev
                #return rhoL * self.pf['pop_Nion'] * self.pf['pop_fesc'] \
                #    * self._get_energy_per_photon(Emin, Emax) * erg_per_ev \
                #    / self.cosm.g_per_baryon #/ (Emax - Emin)
            else:
                raise NotImplemented('help')
                #return rhoL * self.pf['pop_fX'] * self.pf['pop_cX'] \
                #    / ((self.cosm.g_per_baryon / g_per_msun) / s_per_yr)

        # This assumes we're interested in the (EminNorm, EmaxNorm) band
        if self.is_quiescent:
            rhoL = self.get_smd(z) * self.tab_radiative_yield * on
        else:
            rhoL = self.get_sfrd(z) * self.tab_radiative_yield * on

        # At this point [rhoL] = erg/s/cMpc^-3

        # Convert from reference band to arbitrary band
        rhoL *= self._convert_band(band, units=units)
        rhoL *= self.get_fesc(z, Mh=None, x=x, band=band, units=units)

        #print('hi', band, self._convert_band(band, units=units),
        #    self.get_fesc(z, Mh=None, x=x, band=band, units=units))

        #Emin, Emax = self.src.get_ev_from_x(band, units=units)

        ## Apply reprocessing
        #if (Emax is None) or (Emin is None):
        #    if self.pf['pop_reproc']:
        #        rhoL *= (1. - self.pf['pop_fesc']) * self.pf['pop_frep']
        #elif Emax > E_LL and Emin < self.pf['pop_Emin_xray']:
        #    rhoL *= self.pf['pop_fesc']
        #elif Emax <= E_LL:
        #    if self.pf['pop_reproc']:
        #        fesc = (1. - self.pf['pop_fesc']) * self.pf['pop_frep']
        #    elif Emin >= E_LyA:
        #        fesc = self.pf['pop_fesc_LW']
        #    else:
        #        fesc = 1.

        #    rhoL *= fesc

        if x is not None:
            return rhoL * self.src.get_spectrum(x, units=units)
        else:
            return rhoL

    #def get_fesc(self, z):
    #    """
    #    Get the escape fraction of ionizing photons.
    #    """
    #    func = self._get_function('pop_fesc')
    #    return func(z=z)

    def get_photon_emissivity(self, z, band=None, units='eV'):
        """
        Return the photon luminosity density in the (Emin, Emax) band.

        Parameters
        ----------
        z : int, flot
            Redshift of interest.

        Returns
        -------
        Photon luminosity density in photons / s / cMpc**3.

        """

        rhoL = self.get_emissivity(z, band=band, units=units)
        eV_per_phot = self._get_energy_per_photon(band, units=units)

        return rhoL / (eV_per_phot * erg_per_ev)

    def get_zeta_ion(self, z):
        """
        This is not quite the standard definition of zeta. It has an extra
        factor of fbaryon since fstar is implemented throughout the rest of
        the code as an efficiency wrt baryonic inflow, not matter inflow.
        """

        if not self.is_src_ion:
            zeta = 0.0
        else:
            zeta = self.pf['pop_Nion'] * self.pf['pop_fesc'] \
                * self.pf['pop_fstar']

        return zeta
