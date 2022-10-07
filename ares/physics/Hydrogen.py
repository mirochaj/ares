"""

Hydrogen.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Mar 12 18:02:07 2012

Description: Container for hydrogen physics stuff.

"""

import scipy
import numpy as np
from types import FunctionType
from scipy.integrate import quad
from scipy.optimize import fsolve, minimize
from ..util.ParameterFile import ParameterFile
from ..util.Math import central_difference, interp1d
from .Constants import A10, T_star, m_p, m_e, erg_per_ev, h_P, c, E_LyA, E_LL, \
    k_B, nu_0_mhz, E10, ev_per_hz, A_LyA, nu_alpha, lam_LyA, m_H

try:
    from scipy.special import gamma
    g23 = gamma(2. / 3.)
    g13 = gamma(1. / 3.)
    c1 = 4. * np.pi / 3. / np.sqrt(3.) / g23
    c2 = 8. * np.pi / 3. / np.sqrt(3.) / g13

    from scipy.special import airy, hyp2f1, erfc

except ImportError:
    pass

try:
    import mpmath
    have_mpmath = True
except ImportError:
    have_mpmath = False

_scipy_ver = scipy.__version__.split('.')

# This keyword didn't exist until version 0.14
if float(_scipy_ver[1]) >= 14:
    _interp1d_kwargs = {'assume_sorted': True}
else:
    _interp1d_kwargs = {}


huge_Ts = 1e10

# Rate coefficients for spin de-excitation - from Zygelman originally

# H-H collisions.
T_HH = \
    np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, \
     25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, \
     90.0, 100.0, 200.0, 300.0, 500.0, 700.0, 1000.0, \
     2000.0, 3000.0, 5000.0, 7000.0, 10000.0])

kappa_HH = \
    np.array([1.38e-13, 1.43e-13, 2.71e-13, 6.60e-13, 1.47e-12, 2.88e-12, \
     9.10e-12, 1.78e-11, 2.73e-11, 3.67e-11, 5.38e-11, 6.86e-11, \
     8.14e-11, 9.25e-11, 1.02e-10, 1.11e-10, 1.19e-10, 1.75e-10, \
     2.09e-10, 2.56e-10, 2.91e-10, 3.31e-10, 4.27e-10, 4.97e-10, \
     6.03e-10, 6.87e-10, 7.87e-10])

# H-e collisions.
T_He = \
    np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0,
     1000.0, 2000.0, 3000.0, 5000.0, 7000.0,
     10000.0, 15000.0, 20000.0])

kappa_He = \
    np.array([2.39e-10, 3.37e-10, 5.30e-10, 7.46e-10, 1.05e-9, 1.63e-9,
     2.26e-9, 3.11e-9, 4.59e-9, 5.92e-9, 7.15e-9, 7.71e-9,
     8.17e-9, 8.32e-9, 8.37e-9, 8.29e-9, 8.11e-9])

tabulated_coeff = \
    {'kappa_H': kappa_HH, 'kappa_e': kappa_He,
     'T_H': T_HH, 'T_e': T_He}

l_LyA = h_P * c / E_LyA / erg_per_ev
tau_21cm0 = 3. * h_P * c**3 * A10 \
    / (32 * np.pi * k_B * (nu_0_mhz * 1e6)**2)

class Hydrogen(object):
    def __init__(self, pf=None, cosm=None, **kwargs):

        if pf is None:
            self.pf = ParameterFile(**kwargs)
        else:
            self.pf = pf

        if cosm is None:
            from .Cosmology import Cosmology
            self.cosm = Cosmology(pf=self.pf, **self.pf)
        else:
            self.cosm = cosm

        self.interp_method = self.pf['interp_cc']
        self.approx_S = self.pf['approx_Salpha']
        self.approx_Ii = self.pf['approx_lya_Ii']

        self.nmax = self.pf['lya_nmax']

        self.tabulated_coeff = \
            {'kappa_H': kappa_HH, 'kappa_e': kappa_He,
             'T_H': T_HH, 'T_e': T_He}

    @property
    def kappa_H(self):
        if not hasattr(self, '_kappa_H_pre'):
            _kappa_H_pre = interp1d(T_HH, kappa_HH,
                kind=self.interp_method, bounds_error=False,
                fill_value=(kappa_HH[0], kappa_HH[-1]),
                left=kappa_HH[0], right=kappa_HH[-1],
                **_interp1d_kwargs).__call__

            tab_T = self.tabulated_coeff['T_H']
            Tlo = tab_T.min()
            def with_extrap_option(Tk):
                """
                Extrapolate kappa outside tabulated range.
                """

                too_cold = Tk < Tlo

                if np.any(too_cold) and self.pf['extrapolate_coupling']:
                    dkap = np.log10(_kappa_H_pre(tab_T[0])) \
                         - np.log10(_kappa_H_pre(tab_T[1]))
                    dT = np.log10(tab_T[1]) - np.log10(tab_T[0])
                    s = dkap / dT
                    T0 = np.log10(tab_T[0])
                    k0 = np.log10(_kappa_H_pre(tab_T[0]))
                    log10kappa_H = k0 + s * (T0 - np.log10(Tk))

                    out = np.zeros_like(Tk)
                    out[too_cold==1] = 10**log10kappa_H[too_cold==1]
                    out[too_cold==0] = _kappa_H_pre(Tk[too_cold==0])

                    return out
                else:
                    return _kappa_H_pre(Tk)

            self._kappa_H_pre = with_extrap_option

        return self._kappa_H_pre

    @property
    def kappa_e(self):
        if not hasattr(self, '_kappa_e_pre'):
            _kappa_e_pre = interp1d(T_He, kappa_He,
                kind=self.interp_method, bounds_error=False,
                fill_value=(kappa_He[0], kappa_He[-1]),
                left=kappa_He[0], right=kappa_He[-1], **_interp1d_kwargs)

            tab_T = self.tabulated_coeff['T_e']
            Tlo = tab_T.min()
            def with_extrap_option(Tk):
                """
                Extrapolate kappa outside tabulated range.
                """

                too_cold = Tk < Tlo

                if np.any(too_cold) and self.pf['extrapolate_coupling']:
                    dkap = np.log10(_kappa_e_pre(tab_T[0])) \
                         - np.log10(_kappa_e_pre(tab_T[1]))
                    dT = np.log10(tab_T[1]) - np.log10(tab_T[0])
                    s = dkap / dT
                    T0 = np.log10(tab_T[0])
                    k0 = np.log10(_kappa_e_pre(tab_T[0]))
                    log10kappa_e = k0 + s * (T0 - np.log10(Tk))

                    out = np.zeros_like(Tk)
                    out[too_cold==1] = 10**log10kappa_e[too_cold==1]
                    out[too_cold==0] = _kappa_e_pre(Tk[too_cold==0])

                    return out
                else:
                    return _kappa_e_pre(Tk)

            self._kappa_e_pre = with_extrap_option

        return self._kappa_e_pre

    @property
    def Tk_hi_H(self):
        if not hasattr(self, '_Tk_hi_H'):
            self._Tk_hi_H = \
                np.logspace(np.log10(tabulated_coeff['T_H'].min()),
                np.log10(tabulated_coeff['T_H'].max()), 1000)

        return self._Tk_hi_H

    @property
    def Tk_hi_e(self):
        if not hasattr(self, '_Tk_hi_e'):
            self._Tk_hi_e = \
                np.logspace(np.log10(tabulated_coeff['T_e'].min()),
                np.log10(tabulated_coeff['T_e'].max()), 1000)

        return self._Tk_hi_e

    @property
    def kH_hi(self):
        if not hasattr(self, '_kH_hi'):
            self._kH_hi = np.array(list(map(self.kappa_H, self.Tk_hi_H)))

        return self._kH_hi

    @property
    def ke_hi(self):
        if not hasattr(self, '_ke_hi'):
            self._ke_hi = np.array(list(map(self.kappa_e, self.Tk_hi_e)))

        return self._ke_hi

    @property
    def _dlogkH_dlogT(self):
        if not hasattr(self, '_dlogkH_dlogT_'):
            Tk_p_H, dkHdT = central_difference(self.Tk_hi_H, self.kH_hi)
            dlogkH_dlogT = dkHdT * Tk_p_H / np.array(list(map(self.kappa_H, Tk_p_H)))
            _kH_spline = interp1d(Tk_p_H, dlogkH_dlogT)
            self._dlogkH_dlogT_ = lambda T: _kH_spline(T)

        return self._dlogkH_dlogT_

    @property
    def _dlogke_dlogT(self):
        if not hasattr(self, '_dlogke_dlogT_'):
            Tk_p_e, dkedT = central_difference(self.Tk_hi_e, self.ke_hi)
            dlogke_dlogT = dkedT * Tk_p_e / np.array(list(map(self.kappa_e, Tk_p_e)))
            _ke_spline = interp1d(Tk_p_e, dlogke_dlogT)
            self._dlogke_dlogT_ = lambda T: _ke_spline(T)

        return self._dlogke_dlogT_

    def photon_energy(self, nu, nl=1):
        """
        Return energy of photon transitioning from nu to nl in eV.
        Defaults to Lyman-series.
        """
        return Ryd * (1. / nl / nl - 1. / nu / nu) / erg_per_ev

    def photon_freq(self, nu, nl=1):
        return self.photon_energy(nu, nl) * erg_per_ev / h

    def zmax(self, z, n):
        return (1. + z) * (1. - (n + 1)**-2) / (1. - n**-2) - 1.

    def beta(self, z, Tk, xHII, ne, Ja):
        return self.beta_d(z, Tk, xHII, ne, Ja)

    def beta_d(self, z, Tk, xHII, ne, Ja):
        xc = self._xc(z, Tk, xHII, ne)
        xt = self._xtot(z, Tk, xHII, ne, Ja)
        return 1. + xc / (xt * (1. + xt))

    def beta_x(self, z, Tk, xHII, ne, Ja):
        xt = self._xtot(z, Tk, Ja, xHII, ne)

        term2 = (self._xc_HH(z, Tk, xHII, ne) - self._xc_eH(z, Tk, xHII, ne)) \
            / (xt * (1. + xt))

        return 1. + term2

    def beta_a(self, z, Tk, xHII, ne, Ja):
        xa = self._xa(z, Tk, xHII, ne, Ja)
        xt = self._xtot(z, Tk, xHII, ne, Ja)
        return xa / (xt * (1. + xt))

    def beta_T(self, z, Tk, xHII, ne, Ja):
        xt = self._xtot(z, Tk, xHII, ne, Ja)
        xc_HH = self._xc_HH(z, Tk, xHII, ne)
        xc_eH = self._xc_eH(z, Tk, xHII, ne)
        Tcmb = self.cosm.TCMB(z)

        term1 = Tcmb / (Tk - Tcmb)
        brackets = xc_eH * self._dlogke_dlogT(Tk) \
                 + xc_HH * self._dlogkH_dlogT(Tk)
        term2 = (1. / (xt * (1. + xt))) * brackets

        return term1 + term2

    def OnePhotonPerHatom(self, z):
        """
        Flux of photons = 1 photon per hydrogen atom assuming Lyman alpha
        frequency.
        """

        return self.cosm.nH0 * (1. + z)**3 * c / 4. / np.pi / self.nu_alpha

    def frec(self, n):
        """ From Pritchard & Furlanetto 2006. """
        if n == 2:    return 1.0
        elif n == 3:  return 0.0
        elif n == 4:  return 0.2609
        elif n == 5:  return 0.3078
        elif n == 6:  return 0.3259
        elif n == 7:  return 0.3353
        elif n == 8:  return 0.3410
        elif n == 9:  return 0.3448
        elif n == 10: return 0.3476
        elif n == 11: return 0.3496
        elif n == 12: return 0.3512
        elif n == 13: return 0.3524
        elif n == 14: return 0.3535
        elif n == 15: return 0.3543
        elif n == 16: return 0.3550
        elif n == 17: return 0.3556
        elif n == 18: return 0.3561
        elif n == 19: return 0.3565
        elif n == 20: return 0.3569
        elif n == 21: return 0.3572
        elif n == 22: return 0.3575
        elif n == 23: return 0.3578
        elif n == 24: return 0.3580
        elif n == 25: return 0.3582
        elif n == 26: return 0.3584
        elif n == 27: return 0.3586
        elif n == 28: return 0.3587
        elif n == 29: return 0.3589
        elif n == 30: return 0.3590
        else:
            raise ValueError('Only know frec for 2 <= n <= 30!')

    @property
    def Tbg(self):
        if not hasattr(self, '_Tbg'):
            if self.pf['Tbg'] is not None:
                if self.pf['Tbg'] == 'pl':
                    p = self.Tbg_pars
                    self._Tbg = lambda z: p[0] * ((1. + z) / (1. + p[1]))**p[2]
                elif type(self.pf['Tbg']) is FunctionType:
                    self._Tbg = self.pf['Tbg']
                else:
                    raise NotImplemented('help')
            else:
                self._Tbg = None

        return self._Tbg

    @Tbg.setter
    def Tbg(self, value):
        """
        Must be a function of redshift.
        """
        self._Tbg = value

    def _xc(self, z, Tk, xHII=0.0, ne=0.0, Ja=0.0):
        return self.CollisionalCouplingCoefficient(z, Tk, xHII, ne)

    def _xa(self, z, Tk=None, xHII=0.0, ne=0.0, Ja=0.0):
        return self.RadiativeCouplingCoefficient(z, Ja, Tk, xHII)

    def _xtot(self, z, Tk, xHII=0.0, ne=0.0, Ja=0.0):
        return self._xc(z, Tk, xHII, ne) + self._xa(z, Tk, xHII, ne, Ja)

    def _xc_HH(self, z, Tk, xHII=0.0, ne=0.0):
        return self.cosm.nH(z) * (1. - xHII) * self.kappa_H(Tk) \
            * T_star / A10 / self.cosm.TCMB(z)

    def _xc_eH(self, z, Tk, xHII=0.0, ne=0.0):
        return ne * self.kappa_e(Tk) * T_star / A10 / self.cosm.TCMB(z)

    @property
    def Tbg_pars(self):
        if not hasattr(self, '_Tbg_pars'):
            self._Tbg_pars = [self.pf['Tbg_p{}'.format(i)] for i in range(5)]
        return self._Tbg_pars

    #def Tref(self, z):
    #    """
    #    Compute background temperature.
    #    """
    #
    #    return self.cosm.TCMB(z) + self.Tbg(z)

    def CollisionalCouplingCoefficient(self, z, Tk, xHII=0.0, ne=0.0, Tr=0.0):
        """
        Parameters
        ----------
        z : float
            Redshift
        Tk : float
            Kinetic temperature of the gas [K]
        xHII : float
            Hydrogen ionized fraction
        ne : float
            Proper electron density [cm**-3]

        References
        ----------
        Zygelman, B. 2005, ApJ, 622, 1356

        """


        kappa_H = self.kappa_H(Tk)
        kappa_e = self.kappa_e(Tk)

        sum_term = self.cosm.nH(z) * (1. - xHII) * kappa_H + ne * kappa_e

        Tref = self.cosm.TCMB(z) + Tr

        return sum_term * T_star / A10 / Tref


    def RadiativeCouplingCoefficient(self, z, Ja, Tk=None, xHII=None, Tr=0.0,
        ne=0.0):
        """
        Return radiative coupling coefficient (i.e., Wouthuysen-Field effect).

        .. note :: If approx_Sa > 3, will return x_a, S_a, and T_S, which are
            determined iteratively.

        """

        if self.approx_S != 4:
            return self.xalpha_tilde(z) \
                * self.Sa(z=z, Tk=Tk, xHII=xHII) * Ja

        ##
        # Must solve iteratively.
        ##

        # Hirata (2006)
        if self.approx_S == 4:

            # This will correctly cause a crash if Tk is an array
            Tk = float(Tk)

            xi = (1e-7 * self.get_tau_GP(z, xHII=xHII))**(1./3.) * Tk**(-2./3.)
            a = lambda Ts: 1. - 0.0631789 / Tk + 0.115995 / Tk**2 \
                - 0.401403 / Ts / Tk + 0.336463 / Ts / Tk**2
            b = 1. + 2.98394 * xi + 1.53583 * xi**2 + 3.85289 * xi**3

            Sa = lambda Ts: a(Ts) / b

            xc = self.CollisionalCouplingCoefficient(z, Tk, xHII, ne, Tr)
            xa = lambda Ts: self.xalpha_tilde(z) * Sa(Ts) * float(Ja) * 1.
            Tcmb = self.cosm.TCMB(z)
            Trad = Tcmb + Tr

            Tr_inv = 1. / Trad
            Tk_inv = 1. / Tk
            Tc_inv = lambda Ts: Tk**-1. + 0.405535 * (Ts**-1. - Tk**-1.) / Tk

            Ts_inv = lambda Ts: (Tr_inv + xa(Ts) * Tc_inv(Ts) + xc * Tk_inv) \
                   / (1. + xa(Ts) + xc)

            to_solve = lambda Ts: np.abs(1. / Ts / Ts_inv(Ts) - 1.)

            assert type(z) is not np.ndarray

            to_solve = lambda Ts: np.abs(Ts * Ts_inv(Ts) - 1.)

            x = fsolve(to_solve, Tcmb, full_output=True, epsfcn=1e-3)
            Ts = abs(float(x[0]))

            return xa(Ts), Sa(Ts), Ts
        else:
            raise NotImplemented('approx_Salpha={} not currently supported!'.format(self.approx_S))

    def get_tau_GP(self, z, xHII=0.):
        """ Gunn-Peterson optical depth. """
        return 1.5 * self.cosm.nH(z) * (1. - xHII) * l_LyA**3 * 50e6 \
            / self.cosm.HubbleParameter(z)

    def get_lya_width(self, Tk, units='ev'):
        """
        Returns Doppler line-width of the Ly-a line in eV.
        """

        w = np.sqrt(2. * k_B * Tk / m_H / c**2)

        if units.lower() == 'ev':
            return w * E_LyA
        elif units.lower() == 'hz':
            return w * nu_alpha
        else:
            raise NotImplemented('No option units={}!'.format(units))

    def xalpha_tilde(self, z):
        """
        Equation 38 in Hirata (2006).
        """
        gamma = 5e7 # 50 MHz
        return 8. * np.pi * l_LyA**2 * gamma * T_star \
            / 9. / A10 / self.cosm.TCMB(z)

    def Sa(self, z=None, Tk=None, xHII=0.0, Ts=None, Ja=0.0):
        """
        Account for line profile effects.
        """

        if self.approx_S == 0:
            delta_nu = self.get_lya_width(Tk, units='Hz')
            tau = self.get_tau_GP(z, xHII=xHII)
            a = A_LyA / 4. / np.pi / delta_nu
            eta = h_P * nu_alpha**2 / (m_H * c**2 * delta_nu)

            integ = lambda y: y**2 * np.exp(2 * eta * y \
                + (2 * np.pi * y**3) / (3 * a * tau))

            S = (2 * np.pi / a / tau) * quad(integ, -np.inf, 0)[0]
        elif self.approx_S == 1:
            S = 1.0
        elif self.approx_S == 2:
            S = np.exp(-0.37 * np.sqrt(1. + z) * Tk**(-2./3.)) \
                / (1. + 0.4 / Tk)
        elif int(self.approx_S) == 3:
            gamma = 1. / self.get_tau_GP(z, xHII=xHII) / (1. + 0.4 / Tk)  # Eq. 4
            alpha = 0.717 * Tk**(-2./3.) * (1e-6 / gamma)**(1. / 3.) # Eq. 20

            # Gamma function approximation: Eq. 19
            if self.approx_S % 1 != 0:
                # c1 = 4. * np.pi / 3. / np.sqrt(3.) / Gamma(2/3)
                # c2 = 8. * np.pi / 3. / np.sqrt(3.) / Gamma(1/3)
                S = 1. - c1 * alpha + c2 * alpha**2 - 4. * alpha**3 / 3.
            # Actual solution: Eq. 18
            else:
                Ai, Aip, Bi, Bip = airy(-2. * alpha / 3.**(1./3.))
                F2 = hyp2f1(1., 4./3., 5./3., -8 * alpha**3 / 27.)
                S = 1. - 4. * alpha \
                    * (3.**(2./3.) * np.pi * Bi + 3 * alpha**2 * F2) / 9.

        elif self.approx_S == 4:
            xa, S, Ts = self.RadiativeCouplingCoefficient(z, Ja, Tk, xHII)
        elif self.approx_S == 5:
            assert have_mpmath, "Need mpmath installed to run approx_Salpha=5!"
            delta_nu = self.get_lya_width(Tk, units='Hz')
            tau = self.get_tau_GP(z, xHII=xHII)
            a = A_LyA / 4. / np.pi / delta_nu
            eta = h_P * nu_alpha**2 / (m_H * c**2 * delta_nu)

            xi_1 = 9 * np.pi / 4. / a / tau / eta**3

            if type(xi_1) == np.ndarray:
                S = np.array([1. - float(mpmath.hyper([1./3,2./3,1],[],-x)) \
                    for x in xi_1])
            else:
                S = 1. - float(mpmath.hyper([1./3,2./3,1],[],-xi_1))
        else:
            raise NotImplementedError('approx_Sa must be in [0,1,2,3,4,5].')

        return np.maximum(S, 0.0)

    def ELyn(self, n):
        """ Return energy of Lyman-n photon in eV. """
        return self.BohrModel(nfrom=n, ninto=1)

    def BohrModel(self, nfrom, ninto, helium=0):
        """ Return energy of photon in eV using Bohr atom. """

        if helium:
            return 4 * E_LL * ((1. / ninto)**2 - 1. / nfrom**2)
        else:
            return E_LL * ((1. / ninto)**2 - 1. / nfrom**2)

    @property
    def Ts_floor_pars(self):
        if not hasattr(self, '_Ts_floor_pars'):
            self._Ts_floor_pars = [self.pf['floor_Ts_p{}'.format(i)] for i in range(6)]
        return self._Ts_floor_pars

    @property
    def Ts_floor(self):
        if not hasattr(self, '_Ts_floor'):
            if not self.pf['floor_Ts']:
                self._Ts_floor = lambda z: 0.0
            else:
                if self.pf['floor_Ts'] == 'pl':
                    pars = self.Ts_floor_pars

                    self._Ts_floor = lambda z: pars[0] * ((1. + z) / pars[1])**pars[2]

                else:
                    raise NotImplemented('sorry!')

        return self._Ts_floor

    @Ts_floor.setter
    def Ts_floor(self, value):
        self._Ts_floor = value

    def Ts(self, z, Tk, Ja, xHII, ne, Tr=0.0):
        """
        Short-hand for calling `SpinTemperature`.
        """
        return self.get_Ts(z, Tk, Ja, xHII, ne, Tr)

    def SpinTemperature(self, z, Tk, Ja, xHII, ne, Tr=0.0):
        return self.get_Ts(z, Tk, Ja, xHII, ne, Tr)

    def get_Ts(self, z, Tk, Ja, xHII, ne, Tr=0.0):
        """
        Returns spin temperature of intergalactic hydrogen.

        Parameters
        ----------
        z : float, np.ndarray
            Redshift
        Tk : float, np.ndarray
            Gas kinetic temperature
        Ja : float, np.ndarray
            Lyman-alpha flux in units of [s**-1 cm**-2 Hz**-1 sr**-1]
        xHII : float, np.ndarray
            Hydrogen ionized fraction
        ne : float, np.ndarray
            Proper electron density in [cm**-3]

        Returns
        -------
        Spin temperature in Kelvin.

        """

        x_c = self.CollisionalCouplingCoefficient(z, Tk, xHII, ne, Tr)
        x_a = self.RadiativeCouplingCoefficient(z, Ja, Tk, xHII, Tr)
        Tc = Tk

        Tref = self.cosm.TCMB(z) + Tr

        if self.approx_S != 4:
            x_a = self.RadiativeCouplingCoefficient(z, Ja, Tk, xHII, Tr, ne)

            if self.pf['spin_exchange']:
                raise NotImplemented("This isn't self-consistent -- should in principle affect ")
                Tse = (nu_0_mhz * 1e6 / nu_alpha)**2 * m_H * c**2 / 9. / k_B
            else:
                Tse = 0.0

            Tref = self.cosm.TCMB(z) + Tr

            Ts = (1.0 + x_c + x_a * Tk / (Tk + Tse)) / \
                (Tref**-1. + x_c * Tk**-1. + x_a * (Tk + Tse)**-1.)
        else:
            if type(z) == np.ndarray:
                x_a = []; S = []; Ts = []
                for i, red in enumerate(z):
                    _xa, _S, _Ts = self.RadiativeCouplingCoefficient(z[i],
                        Ja[i], Tk[i], xHII[i], Tr[i], ne[i])

                    x_a.append(_xa)
                    S.append(_S)
                    Ts.append(_Ts)

                x_a = np.array(x_a)
                S = np.array(S)
                Ts = np.array(Ts)
            else:
                x_a, S, Ts = self.RadiativeCouplingCoefficient(z,
                    Ja, Tk, xHII, Tr, ne)

        return np.maximum(Ts, self.Ts_floor(z=z))

    def get_21cm_tau(self, z, Ts, xavg=0.0, delta=0.0):
        """
        Compute the 21-cm optical depth.
        """

        return tau_21cm0 * (1 - xavg) * self.cosm.nH(z) * (1. + delta) \
            / Ts / (1. + z) / (self.cosm.HubbleParameter(z) / (1. + z))

    def get_21cm_dTb(self, z, Ts, xavg=0.0, Tr=0.0):
        """
        Global 21-cm signature relative to cosmic microwave background in mK.

        Parameters
        ----------
        z : float, np.ndarray
            Redshift
        Ts : float, np.ndarray
            Spin temperature of intergalactic hydrogen [K].
        xavg : float, np.ndarray
            Volume-averaged ionized fraction, i.e., a weighted average
            between the volume of fully ionized gas and the semi-neutral
            bulk IGM beyond.

        Returns
        -------
        Differential brightness temperature in milli-Kelvin.
        """

        # Writing it this way has the disadvantage that if we supply Ts=np.inf,
        # we'll get NaN. So, just replace with absurdly high Ts.
        if np.any(Ts > huge_Ts):
            if type(Ts) == np.ndarray:
                Ts[Ts > huge_Ts] = huge_Ts
            else:
                Ts = huge_Ts

        Tref = self.cosm.get_Tcmb(z) + Tr

        tau = self.get_21cm_tau(z, Ts, xavg=xavg)
        if self.pf['approx_tau_21cm']:
            dTb = (Ts - Tref) * tau / (1. + z)
        else:
            dTb = (Ts - Tref) * (1. - np.exp(-tau)) / (1. + z)

        # convert to mK
        return 1e3 * dTb

    def T0(self, z):
        return 27. * (self.cosm.omega_b_0 * self.cosm.h70**2 / 0.023) * \
            np.sqrt(0.15 * (1.0 + z) / self.cosm.omega_m_0 / self.cosm.h70**2 / 10.)

    @property
    def inits(self):
        if not hasattr(self, '_inits'):
            self._inits = self.cosm._ics.get_inits_rec()
        return self._inits

    def get_21cm_saturated_limit(self, z):
        return self.get_21cm_dTb(z, np.inf)

    def get_21cm_adiabatic_floor(self, z):
        Tk = np.interp(z, self.inits['z'], self.inits['Tk'])
        Ts = self.SpinTemperature(z, Tk, 1e50, 0.0, 0.0)
        return self.get_21cm_dTb(z, Ts)

    def get_21cm_dTb_no_astrophysics(self, z):
        Ts = self.SpinTemperature(z, self.cosm.Tgas(z), 0., 0., 0.)
        return self.get_21cm_dTb(z, Ts)

    def get_voigt_profile(self, x, a=1):
        return (a / np.pi**1.5) * quad(lambda t: np.exp(-t**2) \
            / (a**2 + (x - t)**2), -np.inf, np.inf)[0]

    def get_lya_profile(self, z, Tk, x, continuum=True, xHII=0.0):
        """
        Compute the spectral shape of the Ly-a line.

        These equations appear in several works:
        - Equations 12 and 13 in Furlanetto & Pritchard (2006)
        - Equations 10 and 11 in Mittal & Kulkarni (2021).

        Parameters
        ----------
        z : int, float
            Redshift.
        Tk : int, float
            Kinetic temperature of the gas [K].
        x : int, float
            Dimensionless variable for photon frequency, defined as
            (nu - nu_0) / Delta nu, where nu_0 is central frequency of Ly-a
            line and Delta nu is the Doppler width.
        continuum : bool
            Whether we're referring to continuum or injected photons.
        xHII : int, float
            Ionized fraction of the gas.

        Returns
        -------
        Intensity J(x) / J(x->inf or x=0) for continuum or injected photons,
        depending on the value of `continuum` parameter.

        """

        # Some ingredients we need.
        delta_nu = self.get_lya_width(Tk, units='Hz')
        tau = self.get_tau_GP(z, xHII=xHII)
        a = A_LyA / 4. / np.pi / delta_nu
        eta = h_P * nu_alpha**2 / (m_H * c**2 * delta_nu)

        exp_term = np.exp(-2 * eta * x - (2 * np.pi * x**3) / (3 * a * tau))

        if continuum or (x < 0):
            integ = lambda y: y**2 * np.exp(2 * eta * y \
                + (2 * np.pi * y**3) / (3 * a * tau))

            J = (2 * np.pi / a / tau) * exp_term \
                * quad(integ, -np.inf, x)[0]
        else:
            J = exp_term

        return J

    def get_lya_EW(self, z, Tk, continuum=True, xHII=0.0):
        """
        Compute the area under the curve for Ly-a line profiles. Needed to
        compute heating rate. This is essentially an equivalent width, hence
        the name.

        .. note :: All solutions here employ the wing approximation.
        .. note :: The continuum solution is analytic in this approximation,
            but the injected photon solution is not. If done numerically,
            it slows things down quite a bit, so we also include the power
            series solution for approx_Salpha=3 (Furlanetto & Pritchard 2006).
            Should add the Chuzhoy & Shapiro 2006 approach too.

        Parameters
        ----------
        z : int, float
            Redshift.
        Tk : int, float
            Kinetic temperature of the gas [K].
        continuum : bool
            Whether we're referring to continuum or injected photons.
        xHII : int, float
            Ionized fraction of the gas.

        """

        # Some ingredients we need.
        delta_nu = self.get_lya_width(Tk, units='Hz')
        tau = self.get_tau_GP(z, xHII=xHII)
        a = A_LyA / 4. / np.pi / delta_nu
        eta = h_P * nu_alpha**2 / (m_H * c**2 * delta_nu)

        xi = eta * (4 * a * tau / np.pi)**(1. / 3.)

        if continuum:
            Ai, Aip, Bi, Bip = airy(-xi)

            I = eta * (2 * np.pi**4 * a**2 * tau**2)**(1. / 3.) \
                * (Ai**2 + Bi**2)

            # Expansion in FP06 used for illustrative purposes, but numerically
            # offers no significant advantage, so I'm leaving this as-is for
            # now.

        else:
            assert self.approx_S != 1

            if (self.approx_S in [0, 5]) and (not self.approx_Ii):
                integ = lambda y: np.exp(-2 * eta * y \
                    - (np.pi * y**3) / (6 * a * tau)) \
                    * erfc(np.sqrt(np.pi * y**3 / 2. / a / tau)) / np.sqrt(y)

                S = self.Sa(z=z, Tk=Tk, xHII=xHII)
                I = eta * np.sqrt(a * tau * 0.5) * quad(integ, 0, np.inf)[0] \
                    - S * (1. - S) / 2. / eta
            elif (self.approx_S == 3) or self.approx_Ii:
                # See Eq. 34 in FP06
                gamma = 1. / tau
                beta = eta * (4 * a / np.pi / gamma)**(1. / 3.)

                A = np.array([-0.6979, 2.5424, -2.5645])
                I = (a / gamma)**(1. / 3.) * np.sum(A * beta**np.arange(3))
            else:
                raise NotImplemented('No approximate solution for I_i for approx_Salpha={}'.format(self.approx_S))

        return I

    def get_lya_heating(self, z, Tk, Jc, Ji, xHII=0.0):
        """
        Compute the Ly-a heating rate by summing over continuum and injected
        line profiles.

        .. note :: This is Eq. 47 in Mittal & Kulkarni (2021). However, the
            factor of density and the Boltzmann constant have been ommitted
            as these are applied in the chemistry solver.

        Parameters
        ----------
        z : int, float
            Redshift.
        Tk : int, float
            Kinetic temperature of the gas [K].
        Jc : int, float
            Lyman-alpha background intensity, continuum photons, i.e.,
            flux at x->inf.
        Ji : int, float
            Lyman-alpha background intensity, injected photons, i.e.,
            from cascades (x=0).
        xHII : int, float
            Ionized fraction of the gas.

        Returns
        -------
        Heating rate in erg/s/cm^3.

        """

        if self.approx_S == 1:
            return 0.0

        delta_nu = self.get_lya_width(Tk, units='Hz')
        Ic = self.get_lya_EW(z, Tk, continuum=1, xHII=xHII)
        Ii = self.get_lya_EW(z, Tk, continuum=0, xHII=xHII)

        prefactor = 8 * np.pi * h_P * delta_nu / 3. / l_LyA

        # Note: this will get hit with a factor of H(z) in ChemicalNetwork
        return prefactor * (Ic * Jc + Ii * Ji)
