# Thanks, Jason Sun, for most of this!

import os
import re
import pickle
import numpy as np
from ..data import ARES
import scipy.special as sp
from types import FunctionType
from scipy.integrate import quad
from scipy.interpolate import interp1d, Akima1DInterpolator
from ..util.ProgressBar import ProgressBar
from .Constants import rho_cgs, c, cm_per_mpc
from .HaloMassFunction import HaloMassFunction

try:
    import h5py
except ImportError:
    pass

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

try:
    import hankel
    have_hankel = True
    from hankel import HankelTransform, SymmetricFourierTransform
except ImportError:
    have_hankel = False

four_pi = 4 * np.pi

class HaloModel(HaloMassFunction):

    def mvir_to_rvir(self, m):
        return (3. * m / (4. * np.pi * self.pf['halo_delta'] \
            * self.cosm.mean_density0)) ** (1. / 3.)

    def cm_relation(self, m, z, get_rs):
        """
        The concentration-mass relation
        """
        if self.pf['halo_cmr'] == 'duffy':
            return self._cm_duffy(m, z, get_rs)
        elif self.pf['halo_cmr'] == 'zehavi':
            return self._cm_zehavi(m, z, get_rs)
        else:
            raise NotImplemented('help!')

    def _cm_duffy(self, m, z, get_rs=True):
        c = 6.71 * (m / (2e12)) ** -0.091 * (1 + z) ** -0.44
        rvir = self.mvir_to_rvir(m)

        if get_rs:
            return c, rvir / c
        else:
            return c

    def _cm_zehavi(self, m, z, get_rs=True):
        c = ((m / 1.5e13) ** -0.13) * 9.0 / (1 + z)
        rvir = self.mvir_to_rvir(m)

        if get_rs:
            return c, rvir / c
        else:
            return c

    def _dc_nfw(self, c):
        return c** 3. / (4. * np.pi) / (np.log(1 + c) - c / (1 + c))

    def rho_nfw(self, r, m, z):

        c, r_s = self.cm_relation(m, z, get_rs=True)

        x = r / r_s
        rn = x / c

        if np.iterable(x):
            result = np.zeros_like(x)
            result[rn <= 1] = (self._dc_nfw(c) / (c * r_s)**3 / (x * (1 + x)**2))[rn <= 1]

            return result
        else:
            if rn <= 1.0:
                return self._dc_nfw(c) / (c * r_s) ** 3 / (x * (1 + x) ** 2)
            else:
                return 0.0

    def u_nfw(self, k, m, z):
        """
        Normalized Fourier Transform of an NFW profile.

        ..note:: This is Equation 81 from Cooray & Sheth (2002).

        Parameters
        ----------
        k : int, float
            Wavenumber
        m :
        """
        c, r_s = self.cm_relation(m, z, get_rs=True)

        K = k * r_s

        asi, ac = sp.sici((1 + c) * K)
        bs, bc = sp.sici(K)

        # The extra factor of np.log(1 + c) - c / (1 + c)) comes in because
        # there's really a normalization factor of 4 pi rho_s r_s^3 / m,
        # and m = 4 pi rho_s r_s^3 * the log term
        norm = 1. / (np.log(1 + c) - c / (1 + c))

        return norm * (np.sin(K) * (asi - bs) - np.sin(c * K) / ((1 + c) * K) \
            + np.cos(K) * (ac - bc))

    def u_isl(self, k, m, z, rmax):
        """
        Normalized Fourier transform of an r^-2 profile.

        rmax : int, float
            Effective horizon. Distance a photon can travel between
            Ly-beta and Ly-alpha.

        """

        asi, aco = sp.sici(rmax * k)

        return asi / rmax / k

    def u_isl_exp(self, k, m, z, rmax, rstar):
        return np.arctan(rstar * k) / rstar / k

    def u_exp(self, k, m, z, rmax):
        rs = 1.

        L0 = (m / 1e11)**1.
        c = rmax / rs

        kappa = k * rs

        norm = rmax / rs**3

        return norm / (1. + kappa**2)**2.

    def u_cgm_rahmati(self, k, m, z):
        rstar = 0.0025
        return np.arctan((rstar * k) ** 0.75) / (rstar * k) ** 0.75

    def u_cgm_steidel(self, k, m, z):
        rstar = 0.2
        return np.arctan((rstar * k) ** 0.85) / (rstar * k) ** 0.85

    def FluxProfile(self, r, m, z, lc=False):
        return m * self.ModulationFactor(z, r=r, lc=lc) / (4. * np.pi * r**2)

    #@RadialProfile.setter
    #def RadialProfile(self, value):
    #    pass

    def FluxProfileFT(self, k, m, z, lc=False):
        _numerator = lambda r: 4. * np.pi * r**2 * np.sin(k * r) / (k * r) \
            * self.FluxProfile(r, m, z, lc=lc)
        _denominator = lambda r: 4. * np.pi * r**2 *\
            self.FluxProfile(r, m, z, lc=lc)
        _r_LW = 97.39 * self.ScalingFactor(z)
        temp = quad(_numerator, 0., _r_LW)[0] / quad(_denominator, 0., _r_LW)[0]
        return temp

    def ScalingFactor(self, z):
        return (self.cosm.h70 / 0.7)**-1 * (self.cosm.omega_m_0 / 0.27)**-0.5 * ((1. + z) / 21.)**-0.5

    def ModulationFactor(self, z0, z=None, r=None, lc=False):
        """
        Return the modulation factor as a function of redshift or comoving distance
        - Reference: Ahn et al. 2009
        :param z0: source redshift
        :param z: the redshift (whose LW intensity is) of interest
        :param r: the distance from the source in cMpc
        :lc: True or False, including the light cone effect
        :return:
        """
        if z != None and r == None:
            r_comov = self.cosm.ComovingRadialDistance(z0, z)
        elif z == None and r != None:
            r_comov = r
        else:
            raise ValueError('Must specify either "z" or "r".')
        alpha = self.ScalingFactor(z0)
        _a = 0.167
        r_star = c * _a * self.cosm.HubbleTime(z0) * (1.+z0) / cm_per_mpc
        ans = np.maximum(1.7 * np.exp(-(r_comov / 116.29 / alpha)**0.68) - 0.7, 0.0)
        if lc == True:
            ans *= np.exp(-r/r_star)
        return ans

    def _get_ps_integrals(self, k, iz, prof1, prof2, lum1, lum2, mmin1, mmin2,
        term):
        """
        Compute integrals over profile, weighted by bias, dndm, etc.,
        needed for halo model.

        .. note :: This is really just a wrapper around _integrate_over_prof,
            that handles the fact that `k` can be a number or an array.

        """

        if type(k) == np.ndarray:
            integ1 = []; integ2 = []
            for _k in k:
                _integ1, _integ2 = self._integrate_over_prof(_k, iz,
                    prof1, prof2, lum1, lum2, mmin1, mmin2, term)
                integ1.append(_integ1)
                integ2.append(_integ2)

            integ1 = np.array(integ1)
            integ2 = np.array(integ2)
        else:
            integ1, integ2 = self._integrate_over_prof(k, iz,
                prof1, prof2, lum1, lum2, mmin1, mmin2, term)

        return integ1, integ2

    def _integrate_over_prof(self, k, iz, prof1, prof2, lum1, lum2, mmin1,
        mmin2, term):
        """
        Compute integrals over profile, weighted by bias, dndm, etc.,
        needed for halo model.
        """

        p1 = np.abs([prof1(k, M, self.tab_z[iz]) for M in self.tab_M])
        p2 = np.abs([prof2(k, M, self.tab_z[iz]) for M in self.tab_M])

        bias = self.tab_bias[iz]
        rho_bar = self.cosm.rho_m_z0 * rho_cgs
        dndlnm = self.tab_dndlnm[iz] # M * dndm

        if (mmin1 is None) and (lum1 is None):
            fcoll1 = 1.

            # Small halo correction. Make use of Cooray & Sheth Eq. 71
            _integrand = dndlnm * (self.tab_M / rho_bar) * bias
            corr1 = 1. - np.trapz(_integrand, x=np.log(self.tab_M))
        elif lum1 is not None:
            corr1 = 0.0
            fcoll1 = 1.
        else:
            fcoll1 = self.tab_fcoll[iz,np.argmin(np.abs(mmin1-self.tab_M))]
            corr1 = 0.0

        if (mmin2 is None) and (lum2 is None):
            fcoll2 = 1.#self.mgtm[iz,0] / rho_bar
            _integrand = dndlnm * (self.tab_M / rho_bar) * bias
            corr2 = 1. - np.trapz(_integrand, x=np.log(self.tab_M))
        elif lum2 is not None:
            corr2 = 0.0
            fcoll2 = 1.
        else:
            fcoll2 = self.fcoll_2d(z, np.log10(Mmin_2))#self.fcoll_Tmin[iz]
            corr2 = 0.0

        ok = self.tab_fcoll[iz] > 0

        # If luminosities passed, then we must cancel out a factor of halo
        # mass that generally normalizes the integrand.
        if lum1 is None:
            weight1 = self.tab_M
            norm1 = rho_bar * fcoll1
        else:
            weight1 = lum1
            norm1 = 1.

        if lum2 is None:
            weight2 = self.tab_M
            norm2 = rho_bar * fcoll2
        else:
            weight2 = lum2
            norm2 = 1.

        ##
        # Are we doing the 1-h or 2-h term?
        if term == 1:
            integrand = dndlnm * weight1 * weight2 * p1 * p2 / norm1 / norm2

            result = np.trapz(integrand[ok==1], x=np.log(self.tab_M[ok==1]))

            return result, None

        elif term == 2:
            integrand1 = dndlnm * weight1 * p1 * bias / norm1
            integrand2 = dndlnm * weight2 * p2 * bias / norm2

            integral1 = np.trapz(integrand1[ok==1], x=np.log(self.tab_M[ok==1]),
                axis=0)
            integral2 = np.trapz(integrand2[ok==1], x=np.log(self.tab_M[ok==1]),
                axis=0)

            return integral1 + corr1, integral2 + corr2

        else:
            raise NotImplemented('dunno man')

    def _prep_for_ps(self, z, k, prof1, prof2, ztol):
        """
        Basic prep: fill prof1=None or prof2=None with defaults, determine
        the index of the requested redshift in our lookup tables.
        """

        iz = np.argmin(np.abs(z - self.tab_z))

        if abs(self.tab_z[iz] - z) > ztol:
            raise ValueError('Requested z={} not in grid (ztol={}).'.format(z,
                ztol))

        if prof1 is None:
            prof1 = self.u_nfw
        if prof2 is None:
            prof2 = prof1

        if k is None:
            k = self.tab_k_lin

        return iz, k, prof1, prof2

    def _get_ps_lin(self, k, iz):
        """
        Return linear matter power spectrum for requested wavenumber `k`.

        .. note :: Assumes we already know the index of the redshift of interest
            in our lookup tables, `iz`.

        """
        if k is None:
            k = self.tab_k_lin
            ps_lin = self.tab_ps_lin[iz]
        else:
            ps_lin = np.exp(np.interp(np.log(k), np.log(self.tab_k_lin),
                np.log(self.tab_ps_lin[iz])))

        return ps_lin

    def get_ps_1h(self, z, k=None, prof1=None, prof2=None, lum1=None, lum2=None,
        mmin1=None, mmin2=None, ztol=1e-3):
        """
        Compute 1-halo term of power spectrum.
        """

        iz, k, prof1, prof2 = self._prep_for_ps(z, k, prof1, prof2, ztol)

        integ1, none = self._get_ps_integrals(k, iz, prof1, prof2,
            lum1, lum2, mmin1, mmin2, term=1)

        return integ1

    def get_ps_2h(self, z, k=None, prof1=None, prof2=None, lum1=None, lum2=None,
        mmin1=None, mmin2=None, ztol=1e-3):
        """
        Get 2-halo term of power spectrum.
        """

        iz, k, prof1, prof2 = self._prep_for_ps(z, k, prof1, prof2, ztol)

        ps_lin = self._get_ps_lin(k, iz)

        integ1, integ2 = self._get_ps_integrals(k, iz, prof1, prof2,
            lum1, lum2, mmin1, mmin2, term=2)

        ps = integ1 * integ2 * ps_lin

        return ps

    def get_ps_shot(self, z, k=None, lum1=None, lum2=None, mmin1=None, mmin2=None,
        ztol=1e-3):
        """
        Compute the two halo term quickly
        """

        iz, k, _prof1_, _prof2_ = self._prep_for_ps(z, k, None, None, ztol)

        dndlnm = self.tab_dndlnm[iz]
        integrand = dndlnm * lum1 * lum2
        shot = np.trapz(integrand, x=np.log(self.tab_M), axis=0)

        return shot

    def get_ps_tot(self, z, k=None, prof1=None, prof2=None, lum1=None, lum2=None,
        mmin1=None, mmin2=None, ztol=1e-3):
        """
        Return total power spectrum as sum of 1h and 2h terms.
        """
        ps_1h = self.get_ps_1h(z, k, prof1, prof2, lum1, lum2, mmin1, mmin2, ztol)
        ps_2h = self.get_ps_2h(z, k, prof1, prof2, lum1, lum2, mmin1, mmin2, ztol)

        return ps_1h + ps_2h

    def CorrelationFunction(self, z, R, k=None, Pofk=None, load=True):
        """
        Compute the correlation function of the matter power spectrum.

        Parameters
        ----------
        z : int, float
            Redshift of interest.
        R : int, float, np.ndarray
            Scale(s) of interest

        """

        ##
        # Load from table
        ##
        if self.pf['hmf_load_ps'] and load:
            iz = np.argmin(np.abs(z - self.tab_z_ps))
            assert abs(z - self.tab_z_ps[iz]) < 1e-2, \
                'Supplied redshift (%g) not in table!' % z
            if len(R) == len(self.tab_R):
                assert np.allclose(R, self.tab_R)
                return self.tab_cf_mm[iz]

            return np.interp(R, self.tab_R, self.tab_cf_mm[iz])

        ##
        # Compute from scratch
        ##

        # Has P(k) already been computed?
        if Pofk is not None:
            if k is None:
                k = self.tab_k
                assert len(Pofk) == len(self.tab_k), \
                    "Mismatch in shape between Pofk and k!"

        else:
            k = self.tab_k
            Pofk = self.get_ps_tot(z, self.tab_k)

        return self.InverseFT3D(R, Pofk, k)

    def InverseFT3D(self, R, ps, k=None, kmin=None, kmax=None,
        epsabs=1e-12, epsrel=1e-12, limit=500, split_by_scale=False,
        method='clenshaw-curtis', use_pb=False, suppression=np.inf):
        """
        Take a power spectrum and perform the inverse (3-D) FT to recover
        a correlation function.
        """
        assert type(R) == np.ndarray

        if (type(ps) == FunctionType) or isinstance(ps, interp1d) \
           or isinstance(ps, Akima1DInterpolator):
            k = ps.x
        elif type(ps) == np.ndarray:
            # Setup interpolant

            assert k is not None, "Must supply k vector as well!"

            #if interpolant == 'akima':
            #    ps = Akima1DInterpolator(k, ps)
            #elif interpolant == 'cubic':

            ps = interp1d(np.log(k), ps, kind='cubic', assume_sorted=True,
                bounds_error=False, fill_value=0.0)

            #_ps = interp1d(np.log(k), np.log(ps), kind='cubic', assume_sorted=True,
            #    bounds_error=False, fill_value=-np.inf)
            #
            #ps = lambda k: np.exp(_ps.__call__(np.log(k)))

        else:
            raise ValueError('Do not understand type of `ps`.')

        if kmin is None:
            kmin = k.min()
        if kmax is None:
            kmax = k.max()

        norm = 1. / ps(np.log(kmax))

        ##
        # Use Steven Murray's `hankel` package to do the transform
        ##
        if method == 'ogata':
            assert have_hankel, "hankel package required for this!"

            integrand = lambda kk: four_pi * kk**2 * norm * ps(np.log(kk)) \
                * np.exp(-kk * R / suppression)
            ht = HankelTransform(nu=0, N=k.size, h=0.001)

            #integrand = lambda kk: ps(np.log(kk)) * norm
            #ht = SymmetricFourierTransform(3, N=k.size, h=0.001)

            #print(ht.integrate(integrand))
            cf = ht.transform(integrand, k=R, ret_err=False, inverse=True) / norm

            return cf / (2. * np.pi)**3
        else:
            pass
            # Otherwise, do it by-hand.


        ##
        # Optional progress bar
        ##
        pb = ProgressBar(R.size, use=self.pf['progress_bar'] * use_pb,
            name='ps(k)->cf(R)')

        # Loop over R and perform integral
        cf = np.zeros_like(R)
        for i, RR in enumerate(R):

            if not pb.has_pb:
                pb.start()

            pb.update(i)

            # Leave sin(k*R) out -- that's the 'weight' for scipy.
            integrand = lambda kk: norm * four_pi * kk**2 * ps(np.log(kk)) \
                * np.exp(-kk * RR / suppression) / kk / RR

            if method == 'clenshaw-curtis':

                if split_by_scale:
                    kcri = np.exp(ps.x[np.argmin(np.abs(np.exp(ps.x) - 1. / RR))])

                    # Integral over small k is easy
                    lowk = np.exp(ps.x) <= kcri
                    klow = np.exp(ps.x[lowk == 1])
                    plow = ps.y[lowk == 1]
                    sinc = np.sin(RR * klow) / klow / RR
                    integ = norm * four_pi * klow**2 * plow * sinc \
                        * np.exp(-klow * RR / suppression)
                    cf[i] = np.trapz(integ * klow, x=np.log(klow)) / norm

                    kstart = kcri

                    #print(RR, 1. / RR, kcri, lowk.sum(), ps.x.size - lowk.sum())
                    #
                    #if lowk.sum() < 1000 and lowk.sum() % 100 == 0:
                    #    import matplotlib.pyplot as pl
                    #
                    #    pl.figure(2)
                    #
                    #    sinc = np.sin(RR * k) / k / RR
                    #    pl.loglog(k, integrand(k) * sinc, color='k')
                    #    pl.loglog([kcri]*2, [1e-4, 1e4], color='y')
                    #    raw_input('<enter>')

                else:
                    kstart = kmin

                # Add in the wiggly part
                cf[i] += quad(integrand, kstart, kmax,
                    epsrel=epsrel, epsabs=epsabs, limit=limit,
                    weight='sin', wvar=RR)[0] / norm

            else:
                raise NotImplemented('help')

        pb.finish()

        # Our FT convention
        cf /= (2 * np.pi)**3

        return cf

    def FT3D(self, k, cf, R=None, Rmin=None, Rmax=None,
        epsabs=1e-12, epsrel=1e-12, limit=500, split_by_scale=False,
        method='clenshaw-curtis', use_pb=False, suppression=np.inf):
        """
        This is nearly identical to the inverse transform function above,
        I just got tired of having to remember to swap meanings of the
        k and R variables. Sometimes clarity is better than minimizing
        redundancy.
        """
        assert type(k) == np.ndarray

        if (type(cf) == FunctionType) or isinstance(cf, interp1d) \
           or isinstance(cf, Akima1DInterpolator):
            R = cf.x
        elif type(cf) == np.ndarray:
            # Setup interpolant

            assert R is not None, "Must supply R vector as well!"

            #if interpolant == 'akima':
            #    ps = Akima1DInterpolator(k, ps)
            #elif interpolant == 'cubic':
            cf = interp1d(np.log(R), cf, kind='cubic', assume_sorted=True,
                bounds_error=False, fill_value=0.0)

        else:
            raise ValueError('Do not understand type of `ps`.')

        if Rmin is None:
            Rmin = R.min()
        if Rmax is None:
            Rmax = R.max()

        norm = 1. / cf(np.log(Rmin))

        if method == 'ogata':
            assert have_hankel, "hankel package required for this!"

            integrand = lambda RR: four_pi * R**2 * norm * cf(np.log(RR))
            ht = HankelTransform(nu=0, N=k.size, h=0.1)

            #integrand = lambda kk: ps(np.log(kk)) * norm
            #ht = SymmetricFourierTransform(3, N=k.size, h=0.001)

            #print(ht.integrate(integrand))
            ps = ht.transform(integrand, k=k, ret_err=False, inverse=False) / norm

            return ps

        ##
        # Optional progress bar
        ##
        pb = ProgressBar(R.size, use=self.pf['progress_bar'] * use_pb,
            name='cf(R)->ps(k)')

        # Loop over k and perform integral
        ps = np.zeros_like(k)
        for i, kk in enumerate(k):

            if not pb.has_pb:
                pb.start()

            pb.update(i)

            if method == 'clenshaw-curtis':

                # Leave sin(k*R) out -- that's the 'weight' for scipy.
                # Note the minus sign.
                integrand = lambda RR: norm * four_pi * RR**2 * cf(np.log(RR)) \
                    * np.exp(-kk * RR / suppression) / kk / RR

                if split_by_scale:
                    Rcri = np.exp(cf.x[np.argmin(np.abs(np.exp(cf.x) - 1. / kk))])

                    # Integral over small k is easy
                    lowR = np.exp(cf.x) <= Rcri
                    Rlow = np.exp(cf.x[lowR == 1])
                    clow = cf.y[lowR == 1]
                    sinc = np.sin(kk * Rlow) / Rlow / kk
                    integ = norm * four_pi * Rlow**2 * clow * sinc \
                        * np.exp(-kk * Rlow / suppression)
                    ps[i] = np.trapz(integ * Rlow, x=np.log(Rlow)) / norm

                    Rstart = Rcri

                    #if lowR.sum() < 1000 and lowR.sum() % 100 == 0:
                    #    import matplotlib.pyplot as pl
                    #
                    #    pl.figure(2)
                    #
                    #    sinc = np.sin(kk * R) / kk / R
                    #    pl.loglog(R, integrand(R) * sinc, color='k')
                    #    pl.loglog([Rcri]*2, [1e-4, 1e4], color='y')
                    #    raw_input('<enter>')

                else:
                    Rstart = Rmin


                # Use 'chebmo' to save Chebyshev moments and pass to next integral?
                ps[i] += quad(integrand, Rstart, Rmax,
                    epsrel=epsrel, epsabs=epsabs, limit=limit,
                    weight='sin', wvar=kk)[0] / norm


            else:
                raise NotImplemented('help')

        pb.finish()

        #
        return np.abs(ps)

    @property
    def tab_k(self):
        """
        k-vector constructed from mps parameters.
        """
        if not hasattr(self, '_tab_k'):
            dlogk = self.pf['hps_dlnk']
            kmi, kma = self.pf['hps_lnk_min'], self.pf['hps_lnk_max']
            logk = np.arange(kmi, kma+dlogk, dlogk)
            self._tab_k = np.exp(logk)

        return self._tab_k

    @tab_k.setter
    def tab_k(self, value):
        self._tab_k = value

    @property
    def tab_R(self):
        """
        R-vector constructed from mps parameters.
        """
        if not hasattr(self, '_tab_R'):
            dlogR = self.pf['hps_dlnR']
            Rmi, Rma = self.pf['hps_lnR_min'], self.pf['hps_lnR_max']
            logR = np.arange(Rmi, Rma+dlogR, dlogR)
            self._tab_R = np.exp(logR)

        return self._tab_R

    @property
    def tab_z_ps(self):
        """
        Redshift array -- different than HMF redshifts!
        """
        if not hasattr(self, '_tab_z_ps'):
            zmin = self.pf['hps_zmin']
            zmax = self.pf['hps_zmax']
            dz = self.pf['hps_dz']

            Nz = int(round(((zmax - zmin) / dz) + 1, 1))
            self._tab_z_ps = np.linspace(zmin, zmax, Nz)

        return self._tab_z_ps

    @tab_z_ps.setter
    def tab_z_ps(self, value):
        self._tab_z_ps = value

    @tab_R.setter
    def tab_R(self, value):
        self._tab_R = value

        print('Setting R attribute. Should verify it matches PS.')

    def __getattr__(self, name):

        if hasattr(HaloMassFunction, name):
            return HaloMassFunction.__dict__[name].__get__(self, HaloMassFunction)

        if (name[0] == '_'):
            raise AttributeError('This will get caught. Don\'t worry! {}'.format(name))

        if name not in self.__dict__.keys():
            if self.pf['hmf_load']:
                self._load_hmf()
            else:
                # Can generate on the fly!
                if name == 'tab_MAR':
                    self.TabulateMAR()
                else:
                    self.TabulateHMF(save_MAR=False)

            if name not in self.__dict__.keys():
                self._load_ps()

        return self.__dict__[name]

    def _load_ps(self, suffix='hdf5'):
        """ Load table from HDF5 or binary. """

        if self.pf['hps_assume_linear']:
            print("Assuming linear matter PS...")
            self._tab_ps_mm = np.zeros((self.tab_z_ps.size, self.tab_k.size))
            self._tab_cf_mm = np.zeros((self.tab_z_ps.size, self.tab_R.size))
            for i, _z_ in enumerate(self.tab_z_ps):
                iz = np.argmin(np.abs(_z_ - self.tab_z))
                self._tab_ps_mm[i,:] = self._get_ps_lin(_z_, iz)

            return

        fn = '%s/input/hmf/%s.%s' % (ARES, self.tab_prefix_ps(), suffix)

        if re.search('.hdf5', fn) or re.search('.h5', fn):
            f = h5py.File(fn, 'r')
            self.tab_z_ps = f['tab_z_ps'].value
            self.tab_R = f['tab_R'].value
            self.tab_k = f['tab_k'].value
            self.tab_ps_mm = f['tab_ps_mm'].value
            self.tab_cf_mm = f['tab_cf_mm'].value
            f.close()
        elif re.search('.pkl', fn):
            f = open(fn, 'rb')
            self.tab_z_ps = pickle.load(f)
            self.tab_R = pickle.load(f)
            self.tab_k = pickle.load(f)
            self.tab_ps_mm = pickle.load(f)
            self.tab_cf_mm = pickle.load(f)
            f.close()
        else:
            raise IOError('Unrecognized format for hps_table.')

    def tab_prefix_ps(self, with_size=True):
        """
        What should we name this table?

        Convention:
        ps_FIT_logM_nM_logMmin_logMmax_z_nz_

        Read:
        halo mass function using FIT form of the mass function
        using nM mass points between logMmin and logMmax
        using nz redshift points between zmin and zmax

        """

        M1, M2 = self.pf['hmf_logMmin'], self.pf['hmf_logMmax']

        z1, z2 = self.pf['hps_zmin'], self.pf['hps_zmax']

        dlogk = self.pf['hps_dlnk']
        kmi, kma = self.pf['hps_lnk_min'], self.pf['hps_lnk_max']
        #logk = np.arange(kmi, kma+dlogk, dlogk)
        #karr = np.exp(logk)

        dlogR = self.pf['hps_dlnR']
        Rmi, Rma = self.pf['hps_lnR_min'], self.pf['hps_lnR_max']
        #logR = np.arange(np.log(Rmi), np.log(Rma)+dlogR, dlogR)
        #Rarr = np.exp(logR)


        if with_size:
            logMsize = (self.pf['hmf_logMmax'] - self.pf['hmf_logMmin']) \
                / self.pf['hmf_dlogM']
            zsize = ((self.pf['hps_zmax'] - self.pf['hps_zmin']) \
                / self.pf['hps_dz']) + 1

            assert logMsize % 1 == 0
            logMsize = int(logMsize)
            assert zsize % 1 == 0
            zsize = int(round(zsize, 1))

            # Should probably save NFW information etc. too
            return 'hps_%s_logM_%s_%i-%i_z_%s_%i-%i_lnR_%.1f-%.1f_dlnR_%.3f_lnk_%.1f-%.1f_dlnk_%.3f' \
                % (self.hmf_func, logMsize, M1, M2, zsize, z1, z2,
                   Rmi, Rma, dlogR, kmi, kma, dlogk)
        else:
            raise NotImplementedError('help')

    def tab_prefix_ps_check(self, with_size=True):
        """
        A version of the prefix to be used only for checkpointing.

        This just means take the full prefix and hack out the bit with the
        redshift interval.
        """

        prefix = self.tab_prefix_ps(with_size)

        iz = prefix.find('_z_')
        iR = prefix.find('_lnR_')

        return prefix[0:iz] + prefix[iR:]

    @property
    def tab_ps_mm(self):
        if not hasattr(self, '_tab_ps_mm'):
            self._load_ps()
        return self._tab_ps_mm

    @tab_ps_mm.setter
    def tab_ps_mm(self, value):
        self._tab_ps_mm = value

    @property
    def tab_cf_mm(self):
        if not hasattr(self, '_tab_cf_mm'):
            ps = self.tab_ps_mm
        return self._tab_cf_mm

    @tab_cf_mm.setter
    def tab_cf_mm(self, value):
        self._tab_cf_mm = value

    def TabulatePS(self, clobber=False, checkpoint=True, **ftkwargs):
        """
        Tabulate the matter power spectrum as a function of redshift and k.
        """

        pb = ProgressBar(len(self.tab_z_ps), 'ps_dd')
        pb.start()

        # Lists to store any checkpoints that are found
        _z = []
        _ps = []
        _cf = []
        if checkpoint:
            if (not os.path.exists('tmp')):
                os.mkdir('tmp')

            pref = self.tab_prefix_ps_check(True)
            fn = 'tmp/{}.{}.pkl'.format(pref, str(rank).zfill(3))

            if os.path.exists(fn) and (not clobber):

                # Should delete if clobber == True?

                if rank == 0:
                    print("Checkpoints for this model found in tmp/.")
                    print("Re-run with clobber=True to overwrite.")

                f = open(fn, 'rb')
                while True:

                    try:
                        tmp = pickle.load(f)
                    except EOFError:
                        break

                    _z.append(tmp[0])
                    _ps.append(tmp[1])
                    _cf.append(tmp[2])

                if _z != []:
                    print("Processor {} loaded checkpoints for z={}".format(rank, _z))

            elif os.path.exists(fn):
                os.remove(fn)

        # Must collect checkpoints so we don't re-run something another
        # processor did!
        if size > 1 and _z != []:
            _zdone = MPI.COMM_WORLD.reduce(_z, root=0)
            zdone = MPI.COMM_WORLD.bcast(_zdone, root=0)
            _zdone_by = MPI.COMM_WORLD.reduce([rank] * len(_z), root=0)
            zdone_by = MPI.COMM_WORLD.bcast(_zdone_by, root=0)
        else:
            zdone = []
            zdone_by = []

        # Figure out what redshift still need to be done by somebody
        assignments = []
        for k, z in enumerate(self.tab_z_ps):
            if z in zdone:
                continue

            assignments.append(z)

        # Split up the work among processors
        my_assignments = []
        for k, z in enumerate(assignments):
            if k % size != rank:
                continue

            my_assignments.append(z)

        if size > 1:
            if len(assignments) % size != 0:
                print("WARNING: Uneven load: {} redshifts and {} processors!".format(len(assignments), size))

        tab_ps_mm = np.zeros((len(self.tab_z_ps), len(self.tab_k)))
        tab_cf_mm = np.zeros((len(self.tab_z_ps), len(self.tab_R)))
        for i, z in enumerate(self.tab_z_ps):

            # Done but not by me!
            if (z in zdone) and (z not in _z):
                continue

            if z not in my_assignments:
                continue

            ##
            # Calculate from scratch
            ##
            print("Processor {} generating z={} PS and CF...".format(rank, z))

            # Must interpolate back to fine grid (uniformly sampled
            # real-space scales) to do FFT and obtain correlation function
            tab_ps_mm[i] = self.get_ps_tot(z, self.tab_k)

            # Compute correlation function at native resolution to save time
            # later.
            tab_cf_mm[i] = self.InverseFT3D(self.tab_R, tab_ps_mm[i],
                self.tab_k, **ftkwargs)

            pb.update(i)

            if not checkpoint:
                continue

            with open(fn, 'ab') as f:
                pickle.dump((z, tab_ps_mm[i], tab_cf_mm[i]), f)
                #print("Processor {} wrote checkpoint for z={}".format(rank, z))

        pb.finish()

        # Grab checkpoints before writing to disk
        for i, z in enumerate(self.tab_z_ps):

            # Done but not by me! If not for this, Allreduce would sum
            # solutions from different processors.
            if (z in zdone) and (z not in _z):
                continue

            # Two processors did the same redshift (backward compatibility)
            if zdone.count(z) > 1:
                done_by = []
                for ii, zz in enumerate(zdone):
                    if zz != z:
                        continue
                    done_by.append(zdone_by[ii])

                if rank != done_by[0]:
                    continue

            ##
            # Load checkpoint, if one exists.
            ##
            if z in _z:

                j = _z.index(z)
                tab_ps_mm[i] = _ps[j]
                tab_cf_mm[i] = _cf[j]


        # Collect results!
        if size > 1:
            tmp1 = np.zeros_like(tab_ps_mm)
            nothing = MPI.COMM_WORLD.Allreduce(tab_ps_mm, tmp1)
            self.tab_ps_mm = tmp1

            tmp2 = np.zeros_like(tab_cf_mm)
            nothing = MPI.COMM_WORLD.Allreduce(tab_cf_mm, tmp2)
            self.tab_cf_mm = tmp2

        else:
            self.tab_ps_mm = tab_ps_mm
            self.tab_cf_mm = tab_cf_mm

        # Done!

    def SavePS(self, fn=None, clobber=True, destination=None, format='hdf5',
        checkpoint=True, **ftkwargs):
        """
        Save matter power spectrum table to HDF5 or binary (via pickle).

        Parameters
        ----------
        fn : str (optional)
            Name of file to save results to. If None, will use
            self.tab_prefix_ps and value of format parameter to make one up.
        clobber : bool
            Overwrite pre-existing files of the same name?
        destination : str
            Path to directory (other than CWD) to save table.
        format : str
            Format of output. Can be 'hdf5' or 'pkl'

        """

        if destination is None:
            destination = '.'

        # Determine filename
        if fn is None:
            fn = '%s/%s.%s' % (destination, self.tab_prefix_ps(True), format)
        else:
            if format not in fn:
                print("Suffix of provided filename does not match chosen format.")
                print("Will go with format indicated by filename suffix.")

        if os.path.exists(fn):
            if clobber:
                os.system('rm -f %s' % fn)
            else:
                raise IOError('File %s exists! Set clobber=True or remove manually.' % fn)

        # Do this first! (Otherwise parallel runs will be garbage)
        self.TabulatePS(clobber=clobber, checkpoint=checkpoint, **ftkwargs)

        if rank > 0:
            return

        self._write_ps(fn, clobber, format)

    def _write_ps(self, fn, clobber, format=format):

        try:
            import hmf
            hmf_v = hmf.__version__
        except AttributeError:
            hmf_v = 'unknown'

        if os.path.exists(fn):
            if clobber:
                os.system('rm -f %s' % fn)
            else:
                raise IOError('File %s exists! Set clobber=True or remove manually.' % fn)

        if format == 'hdf5':
            f = h5py.File(fn, 'w')
            f.create_dataset('tab_z_ps', data=self.tab_z_ps)
            f.create_dataset('tab_R', data=self.tab_R)
            f.create_dataset('tab_k', data=self.tab_k)
            f.create_dataset('tab_ps_mm', data=self.tab_ps_mm)
            f.create_dataset('tab_cf_mm', data=self.tab_cf_mm)

            f.close()
        # Otherwise, pickle it!
        else:
            f = open(fn, 'wb')
            pickle.dump(self.tab_z_ps, f)
            pickle.dump(self.tab_R, f)
            pickle.dump(self.tab_k, f)
            pickle.dump(self.tab_ps_mm, f)
            pickle.dump(self.tab_cf_mm, f)
            pickle.dump(dict(('hmf-version', hmf_v)))
            f.close()

        print('Wrote %s.' % fn)
        return
