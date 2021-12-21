"""

ClusterPopulation.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jan  3 15:08:08 PST 2018

Description:

"""

import os
import re
import inspect
import numpy as np
from ..util import read_lit
from types import FunctionType
from ..util.Math import interp1d
from .Population import Population
from ..util.ParameterFile import get_pq_pars
from scipy.interpolate import interp1d as interp1d_scipy
from ..phenom.ParameterizedQuantity import ParameterizedQuantity
from ..physics.Constants import s_per_yr, s_per_myr, ev_per_hz, g_per_msun, \
    cm_per_mpc

class ClusterPopulation(Population): # pragma: no cover

    #def LuminosityDensity(self):
    #    pass

    def _sfrd_func(self, z):
        # This is a cheat so that the SFRD spline isn't constructed
        # until CALLED. Used only when linking populations.
        return self.SFRD(z)

    def SFRD(self, z):
        on = self.on(z)
        if not np.any(on):
            return z * on

        sfrd = self.FRD(z=z) * self.Mavg(z) * g_per_msun

        return sfrd / cm_per_mpc**3 / s_per_yr

    def FRD(self, **kwargs):
        if 'z' in kwargs:
            z = kwargs['z']
            on = self.on(z)
            if not np.any(on):
                return z * on
        else:
            on = 1

        return on * self._frd(**kwargs)

    @property
    def _frd(self):
        """
        Formation rate density in # of clusters / yr / cMpc^3.
        """
        if not hasattr(self, '_frd_'):
            if self.pf['pop_frd'] is None:
                self._frd_ = None
            if type(self.pf['pop_frd']) in [int, float, np.float]:
                self._frd_ = lambda **kwargs: self.pf['pop_frd']
            elif type(self.pf['pop_frd']) is FunctionType:
                self._frd_ = self.pf['pop_frd']
            elif inspect.ismethod(self.pf['pop_frd']):
                self._frd_ = self.pf['pop_frd']
            elif isinstance(self.pf['pop_frd'], interp1d_scipy):
                self._frd_ = self.pf['pop_frd']
            elif self.pf['pop_frd'][0:2] == 'pq':
                pars = get_pq_pars(self.pf['pop_frd'], self.pf)
                self._frd_ = ParameterizedQuantity(**pars)
            else:
                tmp = read_lit(self.pf['pop_frd'], verbose=self.pf['verbose'])
                self._frd_ = lambda z: tmp.FRD(z, **self.pf['pop_kwargs'])

        return self._frd_

    def MassFunction(self, **kwargs):
        """
        Return the normalized mass function at redshift z, which includes
        clusters formed at all z' > z.
        """

        iz = np.argmin(np.abs(kwargs['z'] - self.tab_zobs))

        frd = np.array([self.FRD(z=z) for z in self.tab_zobs[:iz]])

        # (redshift, mass)
        mdist = np.array([self._mdist(z=z, M=kwargs['M']) \
            for z in self.tab_zobs[:iz]]) * self._mdist_norm

        y = np.zeros_like(kwargs['M'])
        for i, m in enumerate(kwargs['M']):
            _y = frd * 1e6 * mdist[:,i]
            # Integrate over time for clusters of this mass.
            # Note: we don't not allow clusters to lose mass.
            y[i] = np.trapz(_y, x=self.tab_ages[:iz])

        return y

    @property
    def _tab_massfunc(self):
        if not hasattr(self, '_tab_massfunc_'):
            self._tab_massfunc_ = np.zeros((len(self.tab_zobs), len(self.tab_M)))

            # Loop over formation redshifts.
            for i, z in enumerate(self.tab_zobs):

                frd = np.array([self.FRD(z=zz) \
                    for zz in self.tab_zobs[i:]])
                mdist = np.array([self._mdist(z=zz, M=self.tab_M) \
                    for zz in self.tab_zobs[i:]]) * self._mdist_norm

                for j, M in enumerate(self.tab_M):
                    #self._tab_agefunc_[i,i:] = self.tab_ages
                    self._tab_massfunc_[i,j] = np.trapz(frd * mdist[:,j],
                        x=self.tarr[i:] * 1e6)
                    # 1e6 since tarr in Myr and FRD in yr^-1

                    # Luminosity function integrates along age, not mass.
                    #self._tab_lumfunc[i,i:] = np.trapz()

        return self._tab_massfunc_

    #@property
    #def _tab_agefunc(self):
    #    if not hasattr(self, '_tab_agefunc_'):
    #        self._tab_agefunc_ = np.zeros((len(self.tab_zobs), len(self.tab_zobs)))
    #
    #

    @property
    def _tab_rho_L(self):
        if not hasattr(self, '_tab_rho_L_'):
            lf = self._tab_lf
        return self._tab_rho_L_

    @property
    def _tab_Nc(self):
        if not hasattr(self, '_tab_Nc_'):
            lf = self._tab_lf
        return self._tab_Nc_

    @property
    def _tab_lf(self):
        if not hasattr(self, '_tab_lf_'):

            if self.is_aging:
                self._tab_lf_ = np.zeros((len(self.tab_zobs), len(self.Larr)))
            else:
                self._tab_lf_ = np.zeros((len(self.tab_zobs), len(self.tab_M)))

            # Convert to years
            dt = self.pf['pop_age_res'] * 1e6

            # Number of clusters as a function of (zobs, mass, age)
            # Age is young to old.
            self._tab_Nc_ = np.zeros((len(self.tab_zobs), len(self.tab_M),
                len(self.tab_ages)))

            # Luminosities of these clusters.
            self._tab_Lc_ = np.zeros((len(self.tab_zobs), len(self.tab_M),
                len(self.tab_ages)))

            # These are observed redshifts, so we must integrate over
            # all higher redshifts to get the luminosity function.
            for i, z in enumerate(self.tab_zobs):

                if i == len(self.tab_zobs) - 2:
                    # Do this
                    break

                # If we're not allowing this population to age,
                # things get a lot easier. The luminosity function is just
                # the mass function normalized by the time-integrated FRD.
                if not self.is_aging:

                    frd = self.FRD(z=z)
                    mdist = self._mdist(z=z, M=self.tab_M) * self._mdist_norm

                    # Off by a redshift grid pt? i.e., should be pt corresponding
                    # to dt after they start forming.
                    self._tab_Nc_[i,:,0] = frd * dt * mdist
                    self._tab_lf_[i,:] = self._tab_Nc_[i,:,0]

                    continue

                ##
                # If we're here, it means this population can age.
                # Our goal is to compute the number of clusters as a function
                # of mass and age.
                ##

                # At this redshift of observation, we're seeing clusters of
                # a range of ages. At each age, we need to weight by the
                # formation rate density * mass function at the corresponding
                # birth redshift to get the UV luminosity now.

                # First, calculate number of clusters as a function of
                # mass and age.

                # Redshifts at which clusters have formed so far
                zform = self.tab_zobs[0:i+1]
                # Formation rate density at those redshifts
                frd  = self.FRD(z=zform)

                # This is a unity-normalized mass function as a function
                # of redshift, with shape (z[form:obs], tab_M)
                mdist = np.array([self._mdist(z=z, M=self.tab_M) \
                    for z in zform]) * self._mdist_norm
                # This is now (num formation redshifts, mass)

                # Number of clusters at this zobs (index i), as a function
                # of mass and age
                self._tab_Nc_[i,:,0:i+1] = frd * dt * mdist.T

                # This has shape (M, z)

                # Ages of all clusters formed between now and first formation
                # redshift. Reverse order since large age means high
                # formation redshift, i.e., these are in descending order.
                ages = self.tab_ages[0:i+1][-1::-1]

                # Specific luminosities of clusters as a function of age.
                # Will weight by mass in a sec
                L = np.interp(ages, self.tab_ages, self._tab_L1600)

                tmax = self.tarr[i] - self.tarr[0]

                # Need to normalize such that the integral of
                # the LF is guaranteed to equal the total number of GCs.
                # This means integrating over mass and summing up over
                # all ages. That's what `Nc` is for.

                # Save results
                Nc = 0.0
                for j, age in enumerate(ages):

                    if age > tmax:
                        continue

                    # We store as a function of ascending age, but the
                    # FRD and luminosities are sorted by zform, i.e.,
                    # descending age.
                    k = np.argmin(np.abs(age - self.tab_ages))

                    self._tab_Nc_[i,:,k] = frd[j] * dt * mdist[j,:]
                    self._tab_Lc_[i,:,k] = L[j] * self.tab_M

                    Nc += np.trapz(self._tab_Nc[i,:,k], x=self.tab_M, axis=0)

                # At this point, we have an array Nc_of_M_z that represents
                # the number of clusters as a function of (mass, age).
                # So, we convert from age to luminosity, weight by mass,
                # and then histogram in luminosity.

                # It seems odd to histogram here, but I think we must, since
                # mass and age can combine to produce a continuum of
                # luminosities, i.e., we can't just integrate along one
                # dimension.

                weight = self._tab_Nc_[i].flatten()

                # Histogram: number of clusters in given luminosity bins.
                lf, bin_e = np.histogram(self._tab_Lc_[i].flatten(),
                    bins=self.Larr_e, weights=weight, density=True)

                # Prior to this point, `lf` is normalized to integrate
                # to unity since we set density=True
                self._tab_lf_[i] = lf * Nc

        return self._tab_lf_

    def rho_N(self, Emin=None, Emax=None):
        if not hasattr(self, '_rho_N'):
            self._rho_N = {}

        # If we've already figured it out, just return
        if (Emin, Emax) in self._rho_N:
            return self._rho_N[(Emin, Emax)]

        rho_L = self.rho_L(Emin, Emax)
        return self._rho_N[(Emin, Emax)]

    def rho_L(self, Emin=None, Emax=None):

        if not hasattr(self, '_rho_L'):
            self._rho_L = {}
            self._rho_N = {}

        # If we've already figured it out, just return
        if (Emin, Emax) in self._rho_L:
            return self._rho_L[(Emin, Emax)]

        # Important change needed: not L1600, but integrated luminosity
        # at each age.

        # This is in [erg / s / g]. Must convert to Msun.
        yield_per_M = self.src.rad_yield(Emin, Emax) * g_per_msun
        erg_per_phot = self.src.erg_per_phot(Emin, Emax)

        self._tab_rho_L_ = np.zeros_like(self.tab_zobs)
        self._tab_rho_N_ = np.zeros_like(self.tab_zobs)

        # Loop over redshift
        for i, z in enumerate(self.tab_zobs):

            if not self.is_aging:
                y = np.interp(0.0, self.src.times, yield_per_M)
                N = np.interp(0.0, self.src.times, erg_per_phot)
                self._tab_rho_L_[i] = np.trapz(self._tab_Nc[i,:,0] * self.tab_M * y,
                    x=self.tab_M)
                self._tab_rho_N_[i] = np.trapz(self._tab_Nc[i,:,0] * self.tab_M * N,
                    x=self.tab_M)
                continue

            # This is complicated because objects with the same luminosity
            # will have different spectra at different ages. Basically
            # need to repeat LF calculation...?

            ages = self.tab_ages[0:i+1][-1::-1]
            tmax = self.tarr[i] - self.tarr[0]

            # Save results
            for j, age in enumerate(ages):

                if age > tmax:
                    continue

                k = np.argmin(np.abs(age - self.tab_ages))

                y = np.interp(age, self.src.times, yield_per_M)
                N = np.interp(age, self.src.times, erg_per_phot)

                Mc = self._tab_Nc[i,:,k] * self.tab_M

                self._tab_rho_L_[i] += np.trapz(Mc * y, x=self.tab_M)
                self._tab_rho_N_[i] += np.trapz(Mc * N, x=self.tab_M)

        # Not as general as it could be right now...
        if (Emin, Emax) == (13.6, 24.6):
            self._tab_rho_L_ *= self.pf['pop_fesc']
            self._tab_rho_N_ *= self.pf['pop_fesc']

        self._rho_L[(Emin, Emax)] = interp1d(self.tab_zobs[-1::-1],
            self._tab_rho_L_[-1::-1] / cm_per_mpc**3,
            kind=self.pf['pop_interp_sfrd'], bounds_error=False,
            fill_value=0.0)

        self._rho_N[(Emin, Emax)] = interp1d(self.tab_zobs[-1::-1],
            self._tab_rho_N_[-1::-1] / cm_per_mpc**3,
            kind=self.pf['pop_interp_sfrd'], bounds_error=False,
            fill_value=0.0)

        return self._rho_L[(Emin, Emax)]

    def LuminosityFunction(self, z, x=None, mags=True):
        """
        Compute UV luminosity function at redshift `z`.

        Parameters
        ----------
        x : int, float, array [optional]
            Magnitudes at which to output the luminosity function.
            If None, will return magnitude grid used internally, set
            by mass resolution for cluster mass function and
            age resolution (set by pop_age_res).
        mags : bool
            Must be True for now.

        Returns
        -------
        if x is None:
            Returns tuple of (magnitudes, luminosity function)
        else:
            Returns luminosity function at supplied magnitudes `x`.

        """

        assert mags

        iz = np.argmin(np.abs(self.tab_zobs - z))

        _mags = self.mags(z=z)
        _phi = self._tab_lf[iz]

        if self.is_aging:
            dLdmag = np.diff(self.Larr) / np.diff(_mags)
            phi = _phi[0:-1] * np.abs(dLdmag)

            #return mags[0:-1], phi[0:-1] * np.abs(dLdmag)
        else:
            dMdmag = np.diff(self.tab_M) / np.diff(_mags)
            phi = _phi[0:-1] * np.abs(dMdmag)
            #return mags[0:-1], phi[0:-1] * np.abs(dMdmag)

        if x is not None:
            return np.interp(x, _mags[0:-1][-1::-1], phi[-1::-1],
                left=0., right=0.)
        else:
            return _mags[0:-1], phi

    def rho_GC(self, z):
        mags, phi = self.LuminosityFunction(z)

        return np.trapz(phi, dx=abs(np.diff(mags)[0]))

    @property
    def _mdist_norm(self):
        if not hasattr(self, '_mdist_norm_'):
            ##
            # Wont' work if mdist is redshift-dependent.
            ## HELP
            integ = self._mdist(M=self.tab_M) * self.tab_M
            self._mdist_norm_ = 1. / np.trapz(integ, x=np.log(self.tab_M))

        return self._mdist_norm_

    def test(self):
        """
        Integrate GCLF and make sure we recover FRD * dt.
        """
        integ = self._mdist(M=self.tab_M) * self._mdist_norm

        total = np.trapz(integ * self.tab_M, x=np.log(self.tab_M))

        print(total)

    @property
    def _mdist(self):
        if not hasattr(self, '_mdist_'):
            if self.pf['pop_mdist'] is None:
                self._mdist_ = None
            if type(self.pf['pop_mdist']) in [int, float, np.float]:
                self._mdist_ = lambda **kw: self.pf['pop_mdist']
            elif type(self.pf['pop_mdist']) is FunctionType:
                self._mdist_ = self.pf['pop_mdist']
            elif inspect.ismethod(self.pf['pop_mdist']):
                self._mdist_ = self.pf['pop_mdist']
            elif self.pf['pop_mdist'][0:2] == 'pq':
                pars = get_pq_pars(self.pf['pop_mdist'], self.pf)
                self._mdist_ = ParameterizedQuantity(**pars)
            elif isinstance(self.pf['pop_mdist'], interp1d_scipy):
                self._mdist_ = self.pf['pop_mdist']
            else:
                raise NotImplemented('help')
                tmp = read_lit(self.pf['pop_mdist'], verbose=self.pf['verbose'])
                self._mdist_ = lambda z: tmp.SFRD(z, **self.pf['pop_kwargs'])

        return self._mdist_

    @property
    def Larr(self):
        if not hasattr(self, '_Larr'):
            # Setup array of luminosities spanning full range of possibilities
            # from youngest to oldest, least massive cluster to most massive
            # cluster allowed. Unless we're not allowing this cluster to age,
            # in which case the luminosity is easily related to mass function.

            if self.is_aging:
                Lmin = np.log10(self.tab_M.min() * self._tab_L1600.min())
                Lmax = np.log10(self.tab_M.max() * self._tab_L1600.max())
                dlogL = self.pf['pop_dlogM']
                self._Larr = 10**np.arange(Lmin, Lmax+dlogL, dlogL)
            else:
                self._Larr = self._tab_L1600[0] * self.tab_M

        return self._Larr

    @property
    def Larr_e(self):
        """
        Array of luminosity bin edges.
        """
        if not hasattr(self, '_Larr_e'):
            dlogL = self.pf['pop_dlogM']
            edges = 10**np.arange(np.log10(self.Larr[0]) - 0.5 * dlogL,
                        np.log10(self.Larr[-1]) + 0.5 * dlogL, dlogL)

            self._Larr_e = edges

        return self._Larr_e

    def mags(self, z):
        return self.magsys.L_to_MAB(self.Larr)

    @property
    def tab_M(self):
        if not hasattr(self, '_tab_M'):
            lMmin = np.log10(self.pf['pop_Mmin'])
            lMmax = np.log10(self.pf['pop_Mmax'])
            dlogM = self.pf['pop_dlogM']
            self._tab_M = 10**np.arange(lMmin, lMmax+dlogM, dlogM)

        return self._tab_M

    def Mavg(self, z):
        pdf = self._mdist(z=z, M=self.tab_M) * self._mdist_norm

        return np.trapz(pdf * self.tab_M, x=self.tab_M)

    @property
    def tab_zobs(self):
        if not hasattr(self, '_tab_zobs'):
            ages = self.tab_ages
        return self._tab_zobs

    @property
    def tarr(self):
        """
        Array of times (since Big Bang) corresponding to observed redshifts.
        """
        if not hasattr(self, '_tarr'):
            ages = self.tab_ages
        return self._tarr

    @property
    def tab_ages(self):
        """
        Array of ages corresponding to redshifts at which we tabulate LF.
        """
        if not hasattr(self, '_tab_ages'):
            zf = self.pf['final_redshift']
            ti = self.cosm.t_of_z(self.zform) / s_per_myr
            tf = self.cosm.t_of_z(zf) / s_per_myr

            # Time since Big Bang
            dt = self.pf['pop_age_res']
            self._tarr = np.arange(ti, tf+2*dt, dt)
            self._tab_zobs = self.cosm.z_of_t(self._tarr * s_per_myr)

            if self._tab_zobs[-1] > zf:
                self._tab_zobs[-1] = zf
                self._tarr[-1] = self.cosm.t_of_z(zf) / s_per_myr

            # Of clusters formed at corresponding element of tab_zobs
            self._tab_ages = self._tarr - ti

        return self._tab_ages

    @property
    def _tab_L1600(self):
        if not hasattr(self, '_tab_L1600_'):
            self._tab_L1600_ = np.interp(self.tab_ages, self.src.times,
                self.src.L_per_sfr_of_t())

        return self._tab_L1600_

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

        if not self.is_aging:
            on = self.on(z)
            if not np.any(on):
                return z * on
        else:
            on = 1.

        if self.pf['pop_sed_model'] and (Emin is not None) and (Emax is not None):
            if (Emin > self.pf['pop_Emax']):
                return 0.0
            if (Emax < self.pf['pop_Emin']):
                return 0.0

        if self.is_emissivity_separable:
            # The table is in L1600, so we need to convert to broad-band
            # emissivity.
            rhoL = self.rho_L(Emin=Emin, Emax=Emax)(z)
        else:
            raise NotImplemented('help!')

        #if not self.pf['pop_sed_model']:
        #    if (Emin, Emax) == (10.2, 13.6):
        #        return rhoL * self.pf['pop_Nlw'] * self.pf['pop_fesc_LW']
        #    elif (Emin, Emax) == (13.6, 24.6):
        #        return rhoL * self.pf['pop_Nion'] * self.pf['pop_fesc']
        #    else:
        #        return rhoL

        # Convert from reference band to arbitrary band
        #rhoL *= self._convert_band(Emin, Emax)
        #if (Emax is None) or (Emin is None):
        #    pass
        #elif Emax > 13.6 and Emin < self.pf['pop_Emin_xray']:
        #    rhoL *= self.pf['pop_fesc']
        #elif Emax <= 13.6:
        #    rhoL *= self.pf['pop_fesc_LW']

        if E is not None:
            return rhoL * self.src.Spectrum(E)
        else:
            return rhoL

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

        # erg / s / cm**3
        if self.is_emissivity_scalable:
            rhoL = self.Emissivity(z, E=None, Emin=Emin, Emax=Emax)
            erg_per_phot = self._get_energy_per_photon(Emin, Emax) * erg_per_ev

            return rhoL / erg_per_phot
        else:
            return self.rho_N(Emin, Emax)(z)
