"""

DustEmission.py

Author: Felix Bilodeau-Chagnon
Affiliation: McGill University
Created on: Thursday 9 July 2020 17:03:55 EDT

Description: Contains a class which calculates the dust emissions
for each galaxy in a GalaxyEnsemble object based on Imara et al. (2018).

"""

import numpy as np
from scipy.integrate import simps
from ares.physics.Constants import c, h, k_B, g_per_msun, cm_per_kpc, Lsun

# T_dust parameters
PAR_ALPHA = -60
PAR_K = 1e13

class DustEmission(object):
    def __init__(self, galaxyEnsemble, fmin = 1e14, fmax = 1e17, Nfreqs = 500,\
        zmin = 4, zmax = 10, Nz = 7):
        """
        (GalaxyEnsemble, **kwargs) -> DustPopulation

        Creates a DustPopulation instance for the given GalaxyEnsemble
        instance.

        PARAMETERS:

        galaxyEnsemble: GalaxyEnsemble instance

            the GalaxyEnsemble instance for which the dust emissions must be calculated

        fmin: number

            minimum stellar emission frequency sampled in Hertz (default = 1e14 Hz)

        fmax: number

            maximum stellar emission frequency sampled in Hertz (default = 1e17 Hz)

        Nfreqs: integer

            number of frequencies between fmin and fmax to be sampled (default = 500)

        zmin: number

            minimum redshift where the dust emissions will be calculated (default = 4)

        zmax: number

            maximum redshift where the dust emissions will be calculated (default = 10)

        Nz: integer

            number of redshift between zmin and zmax where emissions will be calculated
            (default = 7)
        """
        self._pop = galaxyEnsemble

        self._fmin = fmin
        self._fmax = fmax
        self._Nfreqs = Nfreqs

        self._zmin = zmin
        self._zmax = zmax
        self._Nz = Nz

        self._Ngalaxies = galaxyEnsemble.histories['SFR'].shape[0]

        # Swap the order if they were put in wrong
        if self._fmin > self._fmax:
            self._fmin, self._fmax = self._fmax, self._fmin

        if self._zmin > self._zmax:
            self._zmin, self._zmax = self._zmax, self._zmin

        self._frequencies = np.linspace(self._fmin, self._fmax, self._Nfreqs)
        self._z = np.linspace(self._zmin, self._zmax, self._Nz)

    # Getters for the arguments and simply-derived quantites
    @property
    def pop(self):
        return self._pop

    @property
    def pf(self):
        return self._pop.pf

    @property
    def histories(self):
        return self._pop.histories

    @property
    def fmin(self):
        return self._fmin

    @property
    def fmax(self):
        return self._fmax

    @property
    def Nfreqs(self):
        return self._Nfreqs

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def zmin(self):
        return self._zmin

    @property
    def zmax(self):
        return self._zmax

    @property
    def Nz(self):
        return self._Nz

    @property
    def z(self):
        return self._z

    @property
    def Ngalaxies(self):
        return self._Ngalaxies

    # Properties which must be calculated
    @property
    def L_nu(self):
        """
        (void) -> 3darray

        Calculates and / or returns the specific luminosity L_nu in
        ergs / s / Hz for each galaxy at all frequencies and redshifts
        provided.

        Note: This is a very expensive calculation! Make sure you really need
        to see all those redshifts. Lower redshifts tend to take longer to
        calculate.

        first axis: galaxy index
        second axis: data at given frequency index (ergs / s / Hz)
        third axis: data at given redshift index
        """
        if not hasattr(self, '_L_nu'):
            self._L_nu = np.zeros((self.Ngalaxies, self.Nfreqs, self.Nz))
            waves = c / self.frequencies * 1e8
            for i in range(len(self.z)):
                self._L_nu[:,:,i] = self.pop.synth.Spectrum(waves, \
                    zobs = self.z[i], sfh = self.histories['SFR'], tarr = self.histories['t'])

        return self._L_nu

    @property
    def R_dust(self):
        """
        (void) -> 2darray

        Calculates and / or returns the dust radius for each galaxy at each redshift
        in kpc.

        first axis: galaxy index
        second axis: redshift index
        """
        if not hasattr(self, '_R_dust'):
            Mh = np.zeros((self.Ngalaxies, self.Nz))

            for i in range(self.Nz):
                Mh[:,i] = self.pop.get_field(self.z[i], 'Mh')

            self._R_dust = self.pop.halos.VirialRadius(self.z[:], Mh[:,:]) * 0.018

        return self._R_dust

    @property
    def M_dust(self):
        """
        (void) -> 2darray

        Calculates and / or returns the dust mass in each galaxy at each redshift in solar masses.

        first axis: galaxy index
        second axis: redshift index
        """
        if not hasattr(self, '_M_dust'):
            if (self.pf.get('pop_dust_experimental') is None) or (not self.pf['pop_dust_experimental']):
                self._M_dust = np.zeros((self.Ngalaxies, self.Nz))

                for i in range(self.Nz):
                    self._M_dust[:,i] = self.pop.get_field(self.z[i], 'Md')

            elif self.pf['pop_dust_experimental']:
                self._M_dust = self.M_gas * self.DGR
                NaNs = np.isnan(self._M_dust)
                self._M_dust[NaNs] = 0

            else:
                raise ValueError("Parameter 'pop_dust_experimental' must be True, False, or non-existent (None)")

        return self._M_dust

    @property
    def Z(self):
        """
        (void) -> 2darray

        Calculates and / or returns the metallicity calculated from the Imara et al. (2018) paper.
        The convention is Z = 12 + log(O/H). This is only used in experimental mode
        and is NOT self-consistent.

        first axis: galaxy index
        second axis: redshift index
        """
        if not hasattr(self, '_Z'):
            SFR = np.zeros((self.Ngalaxies, self.Nz))
            M_star = np.zeros((self.Ngalaxies, self.Nz))

            for i in range(self.Nz):
                SFR[:,i] = self.pop.get_field(self.z[i], 'SFR')
                M_star[:,i] = self.pop.get_field(self.z[i], 'Ms')

            print("SFR =", SFR)
            fq = 1 / (M_star / (10**(10.2 + 0.5 * self.z[None,:])) + 1)
            SFR /= 1 - fq

            self._Z = -0.14*np.log10(SFR) + 0.37 * np.log10(M_star) + 4.82
            NaNs = np.isnan(self._Z)
            self._Z[NaNs] = 0

        return self._Z

    @property
    def DGR(self):
        """
        (void) -> 2darray

        Calculates and / or returns the Dust-to-Gas Ratio (DGR) for all galaxies at all redshifts.
        For now, this is only used in experimental mode and is NOT self-consistent. Based on
        Imara et al. (2018).

        first axis: galaxy index
        second axis: redshift index
        """
        if not hasattr(self, '_DGR'):
            Z_SUN = 8.7 # based on Asplund et al. (2009)
            log_of_ratio = np.log10(self.Z / Z_SUN)
            small_Z = self.Z <= 0.26 * Z_SUN
            log_of_ratio[small_Z] = 3.15 * log_of_ratio[small_Z] + 1.25
            log_DGR = log_of_ratio - 2.21
            self._DGR = 10**log_DGR
            NaNs = np.isnan(self._DGR)
            self._DGR[NaNs] = 0

        return self._DGR

    @property
    def M_gas(self):
        """
        (void) -> 2darray

        Calculates and / or returns the gas mass based on Imara et al. This is only used in
        experimental mode, and is NOT self-consistent. Based on Imara et al. (2018)

        first axis: galaxy index
        second axis: redshift index
        """
        if not hasattr(self, '_M_gas'):
            M_star = np.zeros((self.Ngalaxies, self.Nz))
            for i in range(self.Nz):
                M_star[:,i] = self.pop.get_field(self.z[i], 'Ms')

            self._M_gas = 3.87e9 * (1 + self.z[None,:])**1.35 * (M_star / 1e10)**0.49

        return self._M_gas


    @property
    def kappa_nu(self):
        """
        (void) -> 3darray

        Returns and / or calculates the dust opacity given the frequencies. The data
        returned is in cm^2 / g.

        first axis: galaxy index
        second axis: frequency index
        third axis: redshift index
        """
        if not hasattr(self, '_kappa_nu'):
            self._kappa_nu = np.zeros((self.Ngalaxies, self.Nfreqs, self.Nz))
            self._kappa_nu += (0.1 * (self.frequencies / 1e12)**2)[None, :, None]

        return self._kappa_nu

    @property
    def tau_nu(self):
        """
        (void) -> 3darray

        Returns and / or calculates the optical depth of the dust. This
        quantity is dimensionless.

        first axis: galaxy index
        second axis: frequency index
        third axis: redshift index
        """
        if not hasattr(self, '_tau_nu'):
            self._tau_nu = 3 * (self.M_dust[:, None, :] * g_per_msun) * self.kappa_nu \
                / (4 * np.pi * (self.R_dust[:, None, :] * cm_per_kpc)**2)

        return self._tau_nu

    @property
    def T_cmb(self):
        """
        (void) -> 2darray

        Returns and / or calculates the Cosmic Microwave Background temperature
        for each galaxy at each redshift.

        first axis: galaxy index
        second axis: redshift index
        """
        if not hasattr(self, '_T_cmb'):
            self._T_cmb = np.zeros((self.Ngalaxies, self.Nz))
            self._T_cmb += self.pop.cosm.TCMB(self.z[None, :])

        return self._T_cmb

    def __T_dust(self, z, L_nu, tau_nu, R_dust, T_cmb):
        """
        (1darray, 3darray, 3darray, 2darray, 2darray) -> 2darray

        Calculates and returns the dust temperature for each galaxy at each redshift.
        If 'pop_dust_experimental' is False, this is based on Imara et al (2018).
        If 'pop_dust_experimental' is True, this is a log-linear parametrization.

        first axis: galaxy index
        second axis: redshift index
        """
        # --------------------------------------------------------------------------------------------------
        Ngalaxies = L_nu.shape[0]
        Nz = len(z)

        # Calculate total power absorbed per dust mass
        if (self.pf.get('pop_dust_experimental') is None) or (not self.pf['pop_dust_experimental']):

            if (self.pf.get('pop_dust_distrib') is None) or (self.pf['pop_dust_distrib'] == 'homogeneous'):
                f_geom = (1 - np.exp(-tau_nu)) / tau_nu

            elif self.pf['pop_dust_distrib'] == 'pt src':
                f_geom = np.exp(-tau_nu)

            else:
                raise ValueError("Parameter pop_dust_distrib must be 'homogeneous' or 'pt src'.")

            kappa_nu = np.zeros((Ngalaxies, self.Nfreqs, Nz))
            kappa_nu += (0.1 * (self.frequencies / 1e12)**2)[None, :, None]

            tmp_stellar = L_nu * f_geom \
                * self.pop.histories['fcov'] * kappa_nu \
                / (R_dust[:,None,:] * cm_per_kpc)**2


            cmb_freqs = np.linspace(1, 1e14, 1000)

            cmb_kappa_nu = np.zeros((Ngalaxies, 1000, Nz))
            cmb_kappa_nu += (0.1 * (cmb_freqs / 1e12)**2)[None, :, None]

            tmp_cmb = 8 * np.pi * h / c**2 * cmb_kappa_nu * (cmb_freqs[None, :, None])**3 \
                / (np.exp(h * cmb_freqs[None,:,None] / k_B / T_cmb[:,None,:]) - 1)

            tmp_power = simps(tmp_stellar, self.frequencies, axis = 1)
            tmp_power += simps(tmp_cmb, cmb_freqs, axis = 1)

            if self.pf.get('pop_dust_experimental'):
                print("power =", tmp_power)

            # This prefactor is based on analytically performing the integral,
            # so getting the dust temperature for a different model
            # would work differently

            tmp_prefactor = 64e-25 / 63 * np.pi**7 * k_B**6 / c**2 / h**5
            T_dust = (tmp_power / tmp_prefactor)**(1/6)
            NaNs = np.isnan(T_dust)
            T_dust[NaNs] = 0.

        elif self.pf['pop_dust_experimental']:
            # Log-linear parametrization
            # For now, this is redshift independent,
            # probably need to add redshift dependence to it at some point
            M_star = np.zeros((Ngalaxies, Nz))
            for i in range(Nz):
                M_star[:,i] = self.pop.get_field(z[i], 'Ms')

            # Parameters

            PAR_ALPHA = -60
            PAR_K = 1e13

            # Calculation

            T_dust = PAR_ALPHA * np.log10(M_star /  PAR_K)
            T_dust[M_star == 0] = 0

        else:
            raise ValueError("Parameter 'pop_dust_experimental' must be True, False, or non-existent (None)")

        return T_dust

        # --------------------------------------------------------------------------------------------------

    @property
    def T_dust(self):
        """
        (void) -> 2darray

        Calculates and / or returns the dust temperature for each galaxy at
        each redshift.

        first axis: galaxy index
        second axis: redshift index

        If the temperature returns nan, then this means that there was no dust
        in the galaxy in the first place (maybe galaxy was not born yet, or the halo
        has no galaxy at all)
        """
        if not hasattr(self, '_T_dust'):

            self._T_dust = self.__T_dust(self.z, self.L_nu, self.tau_nu, self.R_dust, self.T_cmb)

        return self._T_dust

    def dust_sed(self, fmin, fmax, Nfreqs):
        """
        (num, num, int) -> 1darray, 3darray

        Returns and / or calculates the dust SED in ergs / s / Hz for each galaxy
        at each redshift.

        RETURNS

        frequencies, SED: 1darray, 3darray

        first axis: galaxy index
        second axis: frequency index
        third axis: redshift index

        PARAMETERS

        fmin: number
            minimum frequency of the band

        fmax: number
            maximum frequency of the band

        Nfreqs: integer
            number of frequencies between fmin and fmax to be calculated
        """
        freqs = np.linspace(fmin, fmax, Nfreqs)
        kappa_nu = np.zeros((self.Ngalaxies, Nfreqs, self.Nz))
        kappa_nu += 0.1 * (freqs[None,:,None] / 1e12)**2
        SED = 8 * np.pi * h / c**2 * freqs[None,:,None]**3 * kappa_nu \
            / (np.exp(h * freqs[None,:,None] / k_B / self.T_dust[:,None,:]) - 1) \
            * self.M_dust[:,None,:] * g_per_msun
        return freqs, SED

    def Luminosity(self, z, wave = 3e5, band=None, idnum=None, window=1,
        load=True, energy_units=True, total_IR = False):
        """
        (num, num) -> 1darray

        Calculates the luminosity function for dust emissions in galaxies.
        The results are returned in [ergs / s / Hz]. Returns nu * L_nu

        PARAMETERS

        z: number
            redshift where the luminosity function will be calculated

        wave: number
            wavelength (in Angstroms) where the luminosity will be
            calculated

        band: tuple
            tuple of wavelengths (in Angstroms) where the luminosity
            will be calculated and then averaged over the band
            !! NOT IMPLEMENTED YET !!

        total_IR: boolean
            if False: returns the luminosity at the given wavelength in [ergs / s / Hz]
            if True: ignore wave, and returns the total luminosity in [ergs / s]
            integrated between 8 and 1000 microns

        RETURNS

        luminosities: 1darray
            luminosity in [ergs / s / Hz] of each galaxy for the given redshift and wavelength
            OR
            total luminosity in [ergs / s] integrated between 8 and 1000 microns
        """
        # TODO add functionality for the keywords: band, window, load, energy_units

        # print("CALLING DUST LUMINOSITY FUNCTION")

        # is cached, we take it from the cache
        if load and (z in self.z) and hasattr(self, '_L_nu'):
            index = np.where(self.z == z)[0][0]
            # Here we have the luminosities in ergs / s / Hz
            if not total_IR:
                luminosities = (self.dust_sed(c / (wave * 1e-8), 0, 1))[1][:, 0, index]
            else:
                fmax = c / (8 * 1e-4)
                fmin = c / (1000 * 1e-4)
                freqs, luminosities = self.dust_sed(fmin, fmax, 1000)
                luminosities = simps(luminosities[:,:,index], freqs, axis = 1)

        # is not cached, we calculate everything for the given z and wave
        else:
            # This code calculates the dust temperature
            M_dust = self.pop.get_field(z, 'Md')

            waves = c / self.frequencies * 1e8

            L_nu = self.pop.synth.Spectrum(waves, \
                        zobs = z, sfh = self.histories['SFR'], tarr = self.histories['t'])

            R_dust = self.pop.halos.VirialRadius(z, self.pop.get_field(z, 'Mh')) * 0.018

            kappa_nu = 0.1 * (self.frequencies / 1e12)**2

            tau_nu = 3 * (M_dust[:, None] * g_per_msun) * kappa_nu[None, :]\
                    / (4 * np.pi * (R_dust[:, None] * cm_per_kpc)**2)

            T_cmb = self.pop.cosm.TCMB(z)

            T_dust = self.__T_dust(np.array([z]), L_nu[:,:,None], tau_nu[:,:,None], R_dust[:,None], np.array([T_cmb])[:,None])
            T_dust = T_dust[:,0]

            # Now we can finally calculate the luminosities

            if not total_IR:
                nu = c / wave * 1e8
                kappa_nu = 0.1 * (nu / 1e12)**2
                luminosities = 8 * np.pi * h / c**2 * nu**3 * kappa_nu \
                    / (np.exp(h * nu / k_B / T_dust) - 1) * (M_dust * g_per_msun)
            else:
                fmax = c / (8e-4)
                fmin = c / (1000e-4)
                nu = np.linspace(fmin, fmax, 1000)
                kappa_nu = 0.1 * (nu / 1e12)**2
                luminosities = 8 * np.pi * h / c**2 * nu[None,:]**3 * kappa_nu[None,:] \
                    / (np.exp(h * nu[None,:] / k_B / T_dust[:,None]) - 1) * (M_dust[:,None] * g_per_msun)
                luminosities = simps(luminosities, nu, axis = 1)


        if idnum is not None:
            return luminosities[idnum]

        return luminosities
