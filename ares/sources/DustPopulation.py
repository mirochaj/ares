"""

DustPopulation.py

Author: Felix Bilodeau-Chagnon
Affiliation: McGill University
Created on: Thursday 9 July 2020 17:03:55 EDT

Description: Contains a class which calculates the dust emissions
for each galaxy in a GalaxyEnsemble object.

"""

import numpy as np
from scipy.integrate import simps
from ares.physics.Constants import c, h, k_B, g_per_msun, cm_per_kpc

class DustPopulation:
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

        Fetches and / or returns the dust mass in each galaxy at each redshift in solar masses.

        first axis: galaxy index
        second axis: redshift index
        """
        if not hasattr(self, '_M_dust'):
            self._M_dust = np.zeros((self.Ngalaxies, self.Nz))

            for i in range(self.Nz):
                self._M_dust[:,i] = self.pop.get_field(self.z[i], 'Md')

        return self._M_dust

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
            # Calculate total power absorbed per dust mass
            tmp_stellar = self.L_nu * (1 - np.exp(-self.tau_nu)) \
                / self.tau_nu * self.pop.histories['fcov'] * self.kappa_nu \
                / (self.R_dust[:,None,:] * cm_per_kpc)**2
            
            tmp_cmb = 8 * np.pi * h / c**2 * self.kappa_nu * (self.frequencies[None, :, None])**3 \
                / (np.exp(h * self.frequencies[None,:,None] / k_B / self.T_cmb[:,None,:]) - 1)

            tmp_power = simps(tmp_stellar + tmp_cmb, self.frequencies, axis = 1)
            # This prefactor is based on analytically performing the integral
            tmp_prefactor = 64e-25 / 63 * np.pi**7 * k_B**6 / c**2 / h**5
            tmp_T_dust = (tmp_power / tmp_prefactor)**(1/6)
            NaNs = np.isnan(tmp_T_dust)
            tmp_T_dust[NaNs] = 0.
            self._T_dust = tmp_T_dust

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
        load=True, use_cache=True, energy_units=True):
        """
        (num, num) -> 1darray

        Calculates the luminosity function for dust emissions in galaxies.
        The results are returned in [ergs / s / Hz].

        PARAMETERS

        z: number
            redshift where the luminosity function will be calculated
        
        wave: number
            wavelength (in Angtroms) where the luminosity function will be
            calculated

        RETURNS

        luminosities: 1darray
            luminosity in [ergs / s / Hz] of each galaxy for the given redshift and wavelength
        """
        # TODO add functionality for the keywords: band, window, load, energy_units

        # print("CALLING DUST LUMINOSITY FUNCTION")

        # is cached, we take it from the cache
        if use_cache and (z in self.z) and hasattr(self, '_L_nu'):
            index = np.where(self.z == z)[0][0]
            # Here we have the luminosities in ergs / s / Hz
            luminosities = (self.dust_sed(c / (wave * 1e-8), 0, 1))[1][:, 0, index]

        # is not cached, we calculate everything for the given z and wave
        else:
            # All this code is just to get the dust temperature
            # (if we want to not repeat code maybe need to rewrite the class
            # in a more modular way)

            waves = c / self.frequencies * 1e8
            L_nu = self.pop.synth.Spectrum(waves, \
                    zobs = z, sfh = self.histories['SFR'], tarr = self.histories['t'])      # 2darray galaxy, frequency (in ergs / s / Hz)
            
            R_dust = self.pop.halos.VirialRadius(z, self.pop.get_field(z, 'Mh')) * 0.018    # 1darray galaxy (in kpc)

            kappa_nu = 0.1 * (self.frequencies / 1e12)**2                                   # 1darray frequency (in cm^2/g)

            M_dust = self.pop.get_field(z, 'Md')                                            # 1darray galaxy (in solar masses)

            tau_nu = 3 * (M_dust[:, None] * g_per_msun) * kappa_nu[None, :]\
                 / (4 * np.pi * (R_dust[:, None] * cm_per_kpc)**2)                          # 2darray galaxy, frequency (dim. less)

            f_geom = (1 - np.exp(-tau_nu)) / tau_nu                                         # 2darray galaxy, frequency (dim. less)

            f_star = self.pop.histories['fcov']                                             # just a number (dim. less)

            T_cmb = self.pop.cosm.TCMB(z)                                                   # just a number (in K)

            stellar = L_nu * f_geom * f_star * kappa_nu[None, :] \
                / (R_dust[:, None] * cm_per_kpc)**2                                         # 2darray galaxy, frequency
            cmb = 8 * np.pi * h / c**2 * self.frequencies**3 * kappa_nu \
                / (np.exp(h * self.frequencies / k_B / T_cmb) - 1)                          # 1darray frequency
            
            absorb = stellar + cmb[None,:]                                                  # 2darray galaxy, frequency

            power = simps(absorb, self.frequencies, axis = 1)                               # 1darray galaxy

            prefactor = 64e-25 / 63 * np.pi**7 * k_B**6 / c**2 / h**5                       # just a number, cgs units

            T_dust = (power / prefactor)**(1/6)
            NaNs = np.isnan(T_dust)
            T_dust[NaNs] = 0                                                                # 1darray galaxy

            # Now we can finally calculate the luminosities
            
            nu = c / wave * 1e8
            kappa_nu = 0.1 * (nu / 1e12)**2
            luminosities = 8 * np.pi * h / c**2 * nu**3 * kappa_nu \
                / (np.exp(h * nu / k_B / T_dust) - 1) * (M_dust * g_per_msun)               # 1darray galaxy (ergs / s / Hz)


        if idnum is not None:
            return luminosities[idnum]

        return luminosities