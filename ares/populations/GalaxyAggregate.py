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

    def get_emissivity(self, z, x=None, units='eV', band=None,
        units_out='erg/s'):
        """
        Compute the emissivity of this population as a function of redshift
        and rest-frame photon wavelength, energy, or frequency, depending on
        `units`.

        .. note :: If neither `x` or `band` are provided, this will assume
            the band of interest is the entire range given by the source's
            `pop_Emin` and `pop_Emax` parameters.

        Parameters
        ----------
        z : int, float
            Redshift of interest.
        x : int, float
            Wavelength, photon energy, etc. depending on `units`.
        units : str
            Units of `x`, can be 'Ang', 'eV', 'Hz' at the moment.
        units_out : str
            Controls output units. Not that there is an assumed "/cMpc^3"
            that the user need not supply.
        band : tuple, optional
            If provided, defines a band in which we report the integrated
            emissivity.

        Returns
        -------
        Emissivity in units of `units_out` per cMpc^3, i.e., the per unit
        volume need not be provided by the user in `units_out`. This is
        generally just `erg/s/cMpc^3`, but for some tests etc. we may output
        *specific* emissivities, which have units of, e.g., `erg/s/Hz/cMpc^3`.
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

        ucon = self._check_band_and_units(x, band, units, units_out)


        ##
        # Models based on photons / baryon
        ##
        if self.pf['pop_sed'] is None:
            bname = self.src.get_band_name(x=x, band=band, units=units)

            # In this case, photon yields have been provided via parameters.
            # Just need to convert SFRD to baryons/s/cMpc^3
            # and be careful with converting photon luminosity to `units_out`
            # provided by user
            if bname == 'LW':
                # Regardless of input `units`, compute in erg/s/cMpc^3
                lum_cgs = (self.get_sfrd(z) * self.cosm.b_per_msun / s_per_yr) \
                    * self.pf['pop_Nlw'] * self.pf['pop_fesc_LW'] \
                    * self._get_energy_per_photon(band, units=units) * erg_per_ev

                # May have additional unit conversion, e.g., to put back a
                # factor of Hz^-1 or eV^-1 or Ang^-1.
                return ucon * lum_cgs

            elif bname == 'LyC':
                lum_cgs = (self.get_sfrd(z) * self.cosm.b_per_msun / s_per_yr) \
                    * self.pf['pop_Nion'] * self.pf['pop_fesc'] \
                    * self._get_energy_per_photon(band, units=units) * erg_per_ev
                return ucon * lum_cgs
            else:
                raise NotImplemented('help')

        ##
        # If we made it this far, we're dealing with a more sophisticated
        # SED model, i.e., we're not using LW or LyC photon yields set by hand.
        ##

        # This assumes we're interested in the (EminNorm, EmaxNorm) band
        if self.is_quiescent:
            rhoL = self.get_smd(z) * self.tab_radiative_yield * on
        else:
            rhoL = self.get_sfrd(z) * self.tab_radiative_yield * on

        # At this point [rhoL] = erg/s/cMpc^-3

        # Convert from reference band to arbitrary band, i.e., determine
        # fraction of luminosity emitted in supplied `band` relative to total.
        # Also hit with a factor of the escape fraction if applicable.
        rhoL *= self._convert_band(band, units=units)
        rhoL *= self.get_fesc(z, Mh=None, x=x, band=band, units=units)

        ##
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
            return ucon * rhoL * self.src.get_spectrum(x, units=units)
        else:
            return ucon * rhoL

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
