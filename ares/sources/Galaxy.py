"""

Galaxy.py

Author: Jordan Mirocha
Affiliation: JPL / Caltech
Created on: Fri Jun 30 16:08:51 CEST 2023

Description:

"""

import os
import sys
import itertools
import numpy as np
from ..data import ARES
from ..util import ProgressBar
from scipy.optimize import fmin
from scipy.integrate import quad
from ..core import SpectralSynthesis
from ..physics.Constants import s_per_myr
from .SynthesisModel import SynthesisModel

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

class Galaxy(SynthesisModel):
    """
    Class to handle phenomenological SFH models, e.g., delayed tau, exponential.
    """

    @property
    def _src_csfr(self):
        if not hasattr(self, '_src_csfr_'):
            kw = self.pf.copy()
            kw['source_ssp'] = False
            self._src_csfr_ = Galaxy(cosm=self.cosm, **kw)

        return self._src_csfr_

    @property
    def tH(self):
        if not hasattr(self, '_tH'):
            self._tH = self.cosm.t_of_z(0.) / s_per_myr
        return self._tH

    @property
    def synth(self):
        if not hasattr(self, '_synth'):
            self._synth = SpectralSynthesis(**self.pf)
            self._synth.src = self
            self._synth._src_csfr = self._src_csfr
            self._synth.oversampling_enabled = False
            self._synth.oversampling_below = 30
            self._synth.careful_cache = 1

        return self._synth

    def get_sfr(self, t, tobs, **kwargs):
        """
        Return the star formation rate at time (since Big Bang) `t` [in Myr].

        Parameters
        ----------
        t : np.ndarray
            Time [since Big Bang, in Myr] at which to compute SFR.
        kwargs : dict
            Contains parameters that define the star formation history.

        Returns
        -------
        SFR at all times in Msun/yr.

        """

        assert kwargs != {}, "kwargs are not optional!"

        if 'sfh' in kwargs:
            sfh = kwargs['sfh']
        else:
            sfh = self.pf['source_sfh']

        if sfh == 'exp_decl':
            norm = kwargs['norm']
            tau = kwargs['tau']

            sfr = norm * np.exp(-t / tau)
        elif sfh == 'exp_decl_trunc':
            norm = kwargs['norm']
            tau = kwargs['tau']
            t0 = kwargs['t0']

            sfr = norm * np.exp(-t / tau)
            if type(sfr) == np.ndarray:
                sfr[t < t0] = 0
            else:
                if t < t0:
                    sfr = 0
                else:
                    pass
        elif sfh == 'exp_rise':
            norm = kwargs['norm']
            tau = kwargs['tau']
            sfr = norm * np.exp(-self.tH / tau) * np.exp(t / tau)
        elif sfh == 'const':
            norm = kwargs['norm']
            if 't0' in kwargs:
                t0 = kwargs['t0']
            else:
                t0 = 0

            sfr = norm * np.ones_like(t)
            if type(sfr) == np.ndarray:
                sfr[t < t0] = 0
            elif t < t0:
                sfr = 0
        elif sfh == 'delayed_tau':
            norm = kwargs['norm']
            tau = kwargs['tau']
            t0 = kwargs['t0']

            sfr = norm * (t - t0) * np.exp(-(t - t0) / tau) / tau

            if type(sfr) == np.ndarray:
                sfr[t < t0] = 0
            else:
                if t < t0:
                    sfr = 0
                else:
                    pass

        else:
            raise NotImplemented('help')

        ##
        # Null SFR for times after time of observation!
        # Be careful: if time tobs provided is between grid points, we might
        #
        if type(sfr) == np.ndarray:
            k = np.argmin(np.abs(t - tobs))

            # Ignore this is at edge of array (i.e., tobs=t since Big Bang)
            if k == 0:
                pass
            elif t[k] < tobs:
                k -= 1
            else:
                k -= 2
            sfr[t > t[k]] = 0
        else:
            if t > tobs:
                sfr = 0

        return sfr

    def _get_freturn(self, t):
        """
        t in Myr. This is from Behroozi+ 2013.
        """
        return 0.05 * np.log(1. + t / 1.4)

    def get_kwargs(self, t, mass, sfr, disp=False, mtol=0.05, tau_guess=1e3,
        sfh=None, mass_return=False, tarr=None, **kwargs):
        """
        Determine the free parameters of a model needed to produce stellar mass
        `mass` and star formation rate `sfr` at time `t` [since Big Bang / Myr].
        """

        if sfh is None:
            sfh = self.pf['source_sfh']

        kw = {}
        if sfh == 'exp_decl':

            # For this model, the sSFR uniquely determines tau. Just need to
            # solve for it numerically. Put factor of 10^-6 out front to make
            # sure that the units match input [Msun / yr], despite tau being
            # defined in Myr.

            # One small caveat: if we indicate t0, i.e., star formation begins
            # some time t0 after the Big Bang, then we need to make an
            # adjustment here, and should use 'exp_decl_trunc'
            if 't0' in kwargs:
                raise KeyError("Did you mean to use exp_decl_trunc? Supplied 't0'")

            f_sSFR = lambda logtau: 1e-6 \
                / (10**logtau * (np.exp(t / 10**logtau) - 1.))
            func = lambda logtau: np.abs(np.log10(f_sSFR(logtau) / (sfr / mass)))

            tau = 10**fmin(func, np.log10(tau_guess),
                disp=disp, full_output=disp, ftol=0.01, xtol=0.001)[0]

            # Can analytically solve for normalization once tau in hand.
            norm = sfr / np.exp(-t / tau)

            # Stellar mass = A * tau * (1 - e^(-t / tau))
            # For rising history, mass = A * tau * (e^(t / tau) - 1)
            _mass = 1e6 * norm * tau * (1 - np.exp(-t / tau))

            # Guaranteed?
            _sfr = sfr

            ##
            # Refine if mass_return is on.
            if mass_return:
                def func(pars):
                    logA, logtau = pars
                    sfr0 = self.get_sfr(t, t, norm=10**logA, tau=10**logtau,
                        sfh=sfh, **kwargs)
                    dSFR = np.log10(sfr0 / sfr)

                    mhist = self.get_mass(tarr, t, norm=10**logA, tau=10**logtau,
                        mass_return=True, sfh=sfh, **kwargs)

                    if not np.all(np.diff(tarr) > 0):
                        _mass = 10**np.interp(np.log10(t), np.log10(tarr[-1::-1]),
                            np.log10(mhist[-1::-1]))
                    else:
                        _mass = 10**np.interp(np.log10(t), np.log10(tarr),
                            np.log10(mhist))

                    dMst = np.log10(_mass / mass)

                    return abs(dSFR) + abs(dMst)

                best = fmin(func, [np.log10(norm), np.log10(tau)],
                    disp=disp, full_output=disp, ftol=0.0001, xtol=0.0001)

                norm, tau = 10**best

                mhist = self.get_mass(tarr, t, norm=norm, tau=tau,
                    mass_return=True, sfh=sfh, **kwargs)
                shist = self.get_sfr(tarr, t, norm=norm, tau=tau,
                    sfh=sfh, **kwargs)

                if not np.all(np.diff(tarr) > 0):
                    _mass = 10**np.interp(np.log10(t), np.log10(tarr[-1::-1]),
                        np.log10(mhist[-1::-1]))
                    _sfr = 10**np.interp(np.log10(t), np.log10(tarr[-1::-1]),
                        np.log10(shist[-1::-1]))
                else:
                    _mass = 10**np.interp(np.log10(t), np.log10(tarr),
                        np.log10(mhist))
                    _sfr = 10**np.interp(np.log10(t), np.log10(tarr),
                        np.log10(shist))



            ##
            # Save to dict
            kw = {'norm': norm, 'tau': tau, 'sfh': 'exp_decl'}
            #
        elif sfh == 'exp_decl_trunc':
            assert 't0' in kwargs, "Must provide `t0` for sfh='exp_decl_trunc'!"
            t0 = kwargs['t0']
            f_sSFR = lambda logtau: 1e-6 \
                / (10**logtau * (np.exp(t / 10**logtau) - np.exp(t0 / 10**logtau)))
            func = lambda logtau: np.abs(np.log10(f_sSFR(logtau) / (sfr / mass)))
            tau = 10**fmin(func, np.log10(tau_guess),
                disp=disp, full_output=disp, ftol=0.01, xtol=0.001)[0]

            # Can analytically solve for normalization once tau in hand.
            norm = sfr / np.exp(-t / tau)

            # Guaranteed?
            _sfr = sfr

            # Stellar mass = A * tau * (1 - e^(-t / tau))
            # For rising history, mass = A * tau * (e^(t / tau) - 1)
            _mass = 1e6 * norm * tau * (np.exp(-t0 / tau) - np.exp(-t / tau))

            ##
            # Refine if mass_return is on.
            if mass_return:
                def func(pars):
                    logA, logtau = pars
                    sfr0 = self.get_sfr(t, t, norm=10**logA, tau=10**logtau,
                        sfh=sfh, **kwargs)
                    dSFR = np.log10(sfr0 / sfr)

                    mhist = self.get_mass(tarr, t, norm=10**logA, tau=10**logtau,
                        mass_return=True, sfh=sfh, **kwargs)

                    if not np.all(np.diff(tarr) > 0):
                        _mass = 10**np.interp(np.log10(t), np.log10(tarr[-1::-1]),
                            np.log10(mhist[-1::-1]))
                    else:
                        _mass = 10**np.interp(np.log10(t), np.log10(tarr),
                            np.log10(mhist))

                    dMst = np.log10(_mass / mass)

                    return abs(dSFR) + abs(dMst)

                best = fmin(func, [np.log10(norm), np.log10(tau)],
                    disp=disp, full_output=disp, ftol=0.0001, xtol=0.0001)

                norm, tau = 10**best

                mhist = self.get_mass(tarr, t, norm=norm, tau=tau,
                    mass_return=True, sfh=sfh, **kwargs)
                shist = self.get_sfr(tarr, t, norm=norm, tau=tau,
                    sfh=sfh, **kwargs)

                if not np.all(np.diff(tarr) > 0):
                    _mass = 10**np.interp(np.log10(t), np.log10(tarr[-1::-1]),
                        np.log10(mhist[-1::-1]))
                    _sfr = 10**np.interp(np.log10(t), np.log10(tarr[-1::-1]),
                        np.log10(shist[-1::-1]))
                else:
                    _mass = 10**np.interp(np.log10(t), np.log10(tarr),
                        np.log10(mhist))
                    _sfr = 10**np.interp(np.log10(t), np.log10(tarr),
                        np.log10(shist))

            kw['tau'] = tau
            kw['norm'] = norm
            kw['sfh'] = 'exp_decl_trunc'
            kw['t0'] = t0

        elif sfh == 'exp_rise':
            f_sSFR = lambda logtau: 1e-6 \
                / (10**logtau * (1 - np.exp(-t / 10**logtau)))
            func = lambda logtau: np.abs(np.log10(f_sSFR(logtau) / (sfr / mass)))
            tau = 10**fmin(func, np.log10(tau_guess),
                disp=disp, full_output=disp, ftol=0.001, xtol=0.001)[0]

            # Can analytically solve for normalization once tau in hand.
            norm = sfr / np.exp(t / tau) / np.exp(-self.tH / tau)

            _sfr = sfr

            _mass = 1e6 * norm * np.exp(-self.tH / tau) * \
                tau * (np.exp(t / tau) - 1)

            ##
            # Refine if mass_return is on.
            if mass_return:
                def func(pars):
                    logA, logtau = pars
                    sfr0 = self.get_sfr(t, t, norm=10**logA, tau=10**logtau,
                        sfh=sfh, **kwargs)
                    dSFR = np.log10(sfr0 / sfr)

                    mhist = self.get_mass(tarr, t, norm=10**logA, tau=10**logtau,
                        mass_return=True, sfh=sfh, **kwargs)

                    if not np.all(np.diff(tarr) > 0):
                        _mass = 10**np.interp(np.log10(t), np.log10(tarr[-1::-1]),
                            np.log10(mhist[-1::-1]))
                    else:
                        _mass = 10**np.interp(np.log10(t), np.log10(tarr),
                            np.log10(mhist))

                    dMst = np.log10(_mass / mass)

                    return abs(dSFR) + abs(dMst)

                ##
                # Run minimization
                best = fmin(func, [np.log10(norm), np.log10(tau)],
                    disp=disp, full_output=disp, ftol=0.001, xtol=0.001)

                norm, tau = 10**best

                mhist = self.get_mass(tarr, t, norm=norm, tau=tau,
                    mass_return=True, sfh=sfh, **kwargs)
                shist = self.get_sfr(tarr, t, norm=norm, tau=tau,
                    sfh=sfh, **kwargs)

                if not np.all(np.diff(tarr) > 0):
                    _mass = 10**np.interp(np.log10(t), np.log10(tarr[-1::-1]),
                        np.log10(mhist[-1::-1]))
                    _sfr = 10**np.interp(np.log10(t), np.log10(tarr[-1::-1]),
                        np.log10(shist[-1::-1]))
                else:
                    _mass = 10**np.interp(np.log10(t), np.log10(tarr),
                        np.log10(mhist))
                    _sfr = 10**np.interp(np.log10(t), np.log10(tarr),
                        np.log10(shist))


            # Fools get_sfr routine into doing an exponential rise!
            kw['tau'] = tau
            kw['norm'] = norm
            kw['sfh'] = 'exp_rise'
        elif sfh == 'const':
            if mass_return:
                _kw = kwargs.copy()

                # Means this is a fallback option
                if 't0' in _kw:
                    del _kw['t0']

                def func(pars):
                    logt0 = pars[0]
                    mhist = self.get_mass(tarr, t, norm=sfr, t0=10**logt0,
                        mass_return=True, sfh=sfh, **_kw)

                    if not np.all(np.diff(tarr) > 0):
                        _mass = 10**np.interp(np.log10(t), np.log10(tarr[-1::-1]),
                            np.log10(mhist[-1::-1]))
                    else:
                        _mass = 10**np.interp(np.log10(t), np.log10(tarr),
                            np.log10(mhist))

                    dMst = np.log10(_mass / mass)

                    return abs(dMst)

                best = fmin(func, [np.log10(t*0.5)],
                    disp=disp, full_output=disp, ftol=0.001, xtol=0.001)

                t0 = 10**best[0]

                mhist = self.get_mass(tarr, t, norm=sfr, t0=t0,
                    mass_return=True, sfh=sfh, **_kw)

                if not np.all(np.diff(tarr) > 0):
                    _mass = 10**np.interp(np.log10(t), np.log10(tarr[-1::-1]),
                        np.log10(mhist[-1::-1]))
                else:
                    _mass = 10**np.interp(np.log10(t), np.log10(tarr),
                        np.log10(mhist))

                kw['norm'] = sfr
                kw['t0'] = t0
                kw['sfh'] = 'const'
                kw['tau'] = np.inf
            else:
                kw['norm'] = sfr
                kw['tau'] = np.inf
                kw['t0'] = t - mass / sfr / 1e6
                kw['sfh'] = 'const'

            _mass = mass # guaranteed
            _sfr = sfr
        elif sfh == 'delayed_tau':
            assert 't0' in kwargs, \
                "Must assume a value for t0 for SFH=delayed_tau"

            t0 = kwargs['t0']

            # `norm` will cancel in sSFR, just use functions for convenience.
            #f_sSFR = lambda logtau: self.get_sfr(t, t0=t0, norm=10, tau=10**logtau) \
            #    / self.get_mass(t, t0=t0, norm=10, tau=10**logtau)
            #func = lambda logtau: np.abs(np.log10(f_sSFR(logtau) / (sfr / mass)))
            f_SFR = lambda logtau: self.get_sfr(t, t0=t0, norm=10, tau=10**logtau)

            def func(pars):
                logA, logtau = pars
                dSFR = self.get_sfr(t, t, t0=t0, norm=10**logA, tau=10**logtau) \
                     - sfr
                dMst = self.get_mass(t, t, t0=t0, norm=10**logA, tau=10**logtau) \
                     - mass

                return abs(dSFR) + abs(dMst)

            best = fmin(func, [1, np.log10(tau_guess)],
                disp=disp, full_output=disp, ftol=0.01, xtol=0.01)

            norm, tau = 10**best

            # Can analytically solve for normalization once tau in hand.
            #norm = sfr * tau / np.exp(-(t - t0) / tau) / (t - t0)

            # Stellar mass = A * tau * (1 - e^(-t / tau))
            # For rising history, mass = A * tau * (e^(t / tau) - 1)
            _mass = self.get_mass(t, t, t0=t0, norm=norm, tau=tau)

            ##
            # Refine if mass_return is on.
            if mass_return:
                def func(pars):
                    logA, logtau = pars
                    sfr0 = self.get_sfr(t, t, norm=10**logA, tau=10**logtau,
                        sfh=sfh, **kwargs)
                    dSFR = np.log10(sfr0 / sfr)

                    mhist = self.get_mass(tarr, t, norm=10**logA, tau=10**logtau,
                        mass_return=True, sfh=sfh, **kwargs)

                    if not np.all(np.diff(tarr) > 0):
                        _mass = 10**np.interp(np.log10(t), np.log10(tarr[-1::-1]),
                            np.log10(mhist[-1::-1]))
                    else:
                        _mass = 10**np.interp(np.log10(t), np.log10(tarr),
                            np.log10(mhist))

                    dMst = np.log10(_mass / mass)

                    return abs(dSFR) + abs(dMst)

                ##
                # Run minimization
                best = fmin(func, [np.log10(norm), np.log10(tau)],
                    disp=disp, full_output=disp, ftol=0.001, xtol=0.001)

                norm, tau = 10**best

                mhist = self.get_mass(tarr, t, norm=norm, tau=tau,
                    mass_return=True, sfh=sfh, **kwargs)
                shist = self.get_sfr(tarr, t, norm=norm, tau=tau,
                    sfh=sfh, **kwargs)

                if not np.all(np.diff(tarr) > 0):
                    _mass = 10**np.interp(np.log10(t), np.log10(tarr[-1::-1]),
                        np.log10(mhist[-1::-1]))
                    _sfr = 10**np.interp(np.log10(t), np.log10(tarr[-1::-1]),
                        np.log10(shist[-1::-1]))
                else:
                    _mass = 10**np.interp(np.log10(t), np.log10(tarr),
                        np.log10(mhist))
                    _sfr = 10**np.interp(np.log10(t), np.log10(tarr),
                        np.log10(shist))

            kw['norm'] = norm
            kw['tau'] = tau
            kw['t0'] = t0
            kw['sfh'] = 'delayed_tau'

        else:
            raise NotImplemented("help!")

        ##
        # Check stellar mass -- if way above requested `mass`, then the
        # requested history is inadequate. Switch to something else, potentially.
        merr = abs(np.log10(_mass / mass))
        serr = abs(np.log10(_sfr / sfr))

        if (merr < mtol) and (serr < mtol):
            return kw

        # If we're not allowing a fallback option in the event that this
        # SFH cannot simultaneously satisfy mass and SFR, just return.
        if self.pf['source_sfh_fallback'] is None:
            return kw

        if sfh == 'const':
            print("Failing on const SFH", np.log10(_mass), np.log10(mass), sfr, t)

        if kw['sfh'] != self.pf['source_sfh']:
            #print("Double fail?")
            #print(err, np.log10(_mass), np.log10(mass), sfr, kw)
            #input('enter>')
            sfh_fall = 'const'
        else:
            sfh_fall = self.pf['source_sfh_fallback']
        ##
        # If we're here, we're exploring fallback options.
        print(f"Retrieved mass is off by {merr:.3f} relative to mtol.")
        print(f"Let's try this again with sfh={sfh_fall}...")
        kw = self.get_kwargs(t, mass, sfr, disp=disp, tau_guess=tau_guess,
            mtol=mtol, sfh=sfh_fall, mass_return=mass_return, tarr=tarr,
            **kwargs)

        return kw

    def get_mass(self, t, tobs, mass_return=False, **kwargs):
        """
        Return stellar mass for a given SFH model, integrate analytically
        when possible.
        """

        if 'sfh' in kwargs:
            sfh = kwargs['sfh']
        else:
            sfh = self.pf['source_sfh']

        ##
        # No analytics possible here. Use get_sfr.
        if mass_return:
            func = lambda tt: self.get_sfr(tt, tobs, **kwargs)

            flip = False
            if not np.all(np.diff(t) > 0):
                flip = True
                tasc = t[-1::-1]
            else:
                tasc = t

            smd_ret = []
            for i, _t_ in enumerate(tasc):

                # Re-compute integrand accounting for f_return
                smd_of_t = [func(tasc[k]) \
                    * (1 - self._get_freturn(tasc[i] - tasc[k])) \
                        for k, _tt_ in enumerate(tasc[0:i])]

                smd_ret.append(np.trapz(smd_of_t, x=tasc[0:i] * 1e6))

            if flip:
                return np.array(smd_ret)[-1::-1]
            else:
                return np.array(smd_ret)

        elif sfh == 'exp_decl':
            tau = kwargs['tau']
            norm = kwargs['norm']

            # Factor of 1e6 is to convert tau/Myr -> years
            return norm * 1e6 * tau * (1. - np.exp(-t / tau))
        elif sfh == 'exp_rise':
            tau = kwargs['tau']
            norm = kwargs['norm']

            # Factor of 1e6 is to convert tau/Myr -> years
            return norm * 1e6 * tau * (np.exp(t / tau) - 1)
        elif sfh == 'delayed_tau':
            tau = kwargs['tau']
            norm = kwargs['norm']
            t0 = kwargs['t0']


            func = lambda tt: self.get_sfr(tt, tau=tau, norm=norm, t0=t0)
            #return np.array([quad(func, t0, tt) for tt in t])
            return quad(func, t0, t)[0] * 1e6

            #return norm * 1e6 * tau \
            #    * (np.exp((t0 - t) / tau) * (t0 - t - tau) + tau)
        else:
            raise NotImplemented('help')

    def get_spec(self, zobs, t=None, sfh=None, mass=None, sfr=None, waves=None,
        tau_guess=1e3, use_pbar=True, hist={}, **kwargs):
        """
        Return the rest-frame spectrum of a galaxy at observed redshift, `zobs`.

        There are a few options for how this works:
        1) The user can supply an array of times, `t`, and the corresponding
           star formation history, `sfh`, directly as keyword arguments.

        that has
        stellar mass `mass` and SFR `sfr`.

        Parameters
        ----------
        t : np.ndarray
            Array of times in Myr since Big Bang in Myr.
        """

        if waves is None:
            waves = self.tab_waves_c

        kw = None
        perform_synthesis = True
        if (sfh is not None) and (type(sfh) == np.ndarray):
            assert t is not None
            if np.all(np.diff(t) > 0):
                tasc = t.copy()
                sfh_asc = sfh.copy()
                t = t[-1::-1]
                sfh = sfh[-1::-1]
            else:
                tasc = t[-1::-1]
                sfh_asc = sfh[-1::-1]

        elif kwargs == {}:
            assert (t is not None) and (mass is not None) and (sfr is not None), \
                "Must provide kwargs or (t, mass, sfr)"
            kw = self.get_kwargs(t, mass, sfr, tau_guess=tau_guess)
        else:
            kw = kwargs

        if kw is not None:
            sfh = self.get_sfr(self.tab_t_pop, **kw)
            tasc = self.tab_t_pop[-1::-1]
            sfh_asc = sfh[-1::-1]
            #perform_synthesis = \
            #    ('sfh' not in kwargs) or (kwargs['sfh'] not in ['const', 'burst'])

        if hist != {}:
            if 'Z' in hist:
                perform_synthesis = True

        # General case: synthesize SED
        if perform_synthesis:
            spec = self.synth.get_spec_rest(sfh=sfh_asc, tarr=tasc,
                waves=waves, zobs=zobs, load=False, use_pbar=use_pbar,
                hist=hist)
            return spec

        raise ValueError('This shouldnt happen.')

        ##
        # If using fallback option 'const' or 'ssp', don't need to synthesize!
        if kw['sfh'] == 'const':

            assert np.all(np.diff(sfh[sfh > 0]) == 0)
            sfr = sfh.max()

            src = self._src_csfr
            # Figure out how long this galaxy is "on"
            age = tasc[sfh_asc>0].max() - tasc[sfh_asc>0].min()

            if age > src.tab_t.max():
                return np.interp(waves, src.tab_waves_c, src.tab_sed[:,-1])

            # First, interpolate in age
            ilo = np.argmin(np.abs(age - self.tab_t))
            if src.tab_t[ilo] > age:
                ilo -= 1

            sed_lo = src.tab_sed[:,ilo] * src.tab_dwdn
            sed_hi = src.tab_sed[:,ilo+1] * src.tab_dwdn

            sed = [np.interp(age, src.tab_t[ilo:ilo+2], [sed_lo[i], sed_hi[i]]) \
                for i, wave in enumerate(src.tab_waves_c)]

            # Then, interpolate in wavelength
            return sfr * np.interp(waves, self.tab_waves_c, sed)
        else:
            raise NotImplemented('help')

        return spec

    def get_spec_obs(self):
        pass

    def get_mags(self):
        pass

    #def get_lum_per_sfr(self):
    #    pass

    def get_tab_fn(self):
        """
        Tell us where the output of `generate_sed_tables` is going.
        """

        path = self._litinst._kwargs_to_fn(**self.pf)

        assert 'OUTPUT_POP' in path, \
            "Galaxy class should not be used with CONT SFR model."

        fn = path.replace('OUTPUT_POP', 'OUTPUT_SFH_{}'.format(self.pf['source_sfh']))

        return fn

    def get_sfh_axes(self):
        axes = self.pf['source_sfh_axes']
        axes_names = [ax[0] for ax in axes]
        axes_vals =  [ax[1] for ax in axes]

        return axes_names, axes_vals

    def generate_sed_tables(self, use_pbar=True):
        """
        Create a lookup table for the SED given this SFH model.
        """

        fn = self.get_tab_fn()

        if not os.path.exists(fn[0:fn.rfind('/')]):
            os.mkdir(fn[0:fn.rfind('/')])
        if not os.path.exists(fn + '_checkpoints'):
            os.mkdir(fn + '_checkpoints')

        axes_names, axes_vals = self.get_sfh_axes()

        ##
        # Look for matching file
        if os.path.exists(fn):
            with h5py.File(fn, 'r') as f:
                sed_tab = np.array(f[('seds')])

                # Check that axes match what's in parameter file.
                _axes_names = list(f[('axes_names')])
                _axes_vals = [np.array(f[(f'axes_vals_{k}')]) \
                    for k in range(len(_axes_names))]

                for k in range(len(axes_names)):
                    assert axes_names[k] == _axes_names[k].decode(), \
                        f"Mismatch in axis={k} between parameters and table!"
                    assert np.allclose(axes_vals[k],_axes_vals[k]), \
                        f"Mismatch in axis={k} values between parameters and table!"

            if self.pf['verbose']:
                print("# Loaded {}".format(fn.replace(ARES, '$ARES')))
            return sed_tab

        else:
            print(f"# Did not find {fn}. Will generate from scratch.")

        ##
        # If we didn't find one, generate from scratch
        shape = np.array([vals.size for vals in axes_vals])
        axes_ind  =  [range(shape[i]) for i in range(len(axes_names))]
        ndim = len(axes_names)

        combos = itertools.product(*axes_vals)
        coords = itertools.product(*axes_ind)

        kwargs_flat = []
        for combo in combos:
            tmp = {axes_names[i]:combo[i] for i in range(ndim)}
            kwargs_flat.append(tmp)

        # Create an N+2 dimension SED lookup table, where N is the number of
        # free parameters associated with the SFH.
        deg = self.pf['pop_sfh_degrade']
        tarr = self.tab_t_pop[::deg]
        zarr = self.tab_z_pop[::deg]
        tasc = tarr[-1::-1]
        zdes = zarr[-1::-1]

        #degL = self.pf['pop_sed_degrade']
        waves = self.tab_waves_c#[::degL]

        ##
        # If we didn't find a matching file, make a new one
        if rank == 0:
            print(f"# Will save SED table to {fn}.")

        pb = ProgressBar(len(kwargs_flat) * tasc.size, name='seds|sfhs',
            use=self.pf['progress_bar'] and use_pbar)
        pb.start()

        data = np.zeros([waves.size, tasc.size] + list(shape))
        ind_flat = []
        for i, ind in enumerate(coords):

            kw = kwargs_flat[i]
            ind_flat.append(ind)

            for j, _t_ in enumerate(tasc):

                # Model ID number, just used for progressbar and parellelism
                k = i * len(tasc) + j

                if k % size != rank:
                    continue

                fn_ch = fn + '_checkpoints/t_{:.4f}'.format(np.log10(_t_))
                for l, name in enumerate(axes_names):
                    fn_ch += f'_{name}_{np.log10(kw[name]):.2f}'

                s = slice(None),j, *list(ind)

                # Check for checkpoint file
                if os.path.exists(fn_ch):
                    spec = np.loadtxt(fn_ch, unpack=True)
                    data[s] = spec
                    continue

                ##
                # Otherwise, continue on and generate from scratch
                sfh_asc = self.get_sfr(tasc, **kw)
                spec = self.synth.get_spec_rest(waves,
                    sfh=sfh_asc, tarr=tasc, zobs=zdes[j],
                    load=False, use_pbar=False)

                ##
                # Save checkpoint
                np.savetxt(fn_ch, spec.T)
                data[s] = spec

                pb.update(k)

        pb.finish()

        if size > 1: # pragma: no cover
            data_deg = np.zeros_like(data)
            nothing = MPI.COMM_WORLD.Allreduce(data, data_deg)
        else:
            data_deg = data.copy()

        del data

        ##
        # Let root processor do the rest.
        if rank > 0:
            sys.exit(0)

        ##
        # Interpolate to higher resolution table.
        if deg not in [None, 1]:
            data = np.zeros([waves.size, self.tab_t_pop.size] + list(shape))

            pb = ProgressBar(len(kwargs_flat) * self.tab_t_pop.size * waves.size,
                name='seds|sfhs', use=self.pf['progress_bar'] and use_pbar)
            pb.start()

            ct = 0
            for i, ind in enumerate(ind_flat):
                for j, t in enumerate(self.tab_t_pop):
                    for k, wave in enumerate(waves):

                        s = k,j,*ind
                        sdeg = k,slice(None),*ind
                        data[s] = 10**np.interp(np.log10(t), np.log10(tarr),
                            np.log10(data_deg[sdeg]))

                        pb.update(ct)
                        ct += 1

            pb.finish()
            print("# Done interpolating back to native `t` grid.")
        else:
            data = data_deg

        ##
        # Save to disk.
        with h5py.File(fn, 'w') as f:
            f.create_dataset('seds', data=data)
            f.create_dataset('t', data=self.tab_t_pop)
            f.create_dataset('z', data=self.tab_z_pop)
            f.create_dataset('waves', data=waves)
            f.create_dataset('axes_names', data=axes_names)
            for k in range(len(axes_names)):
                f.create_dataset(f'axes_vals_{k}', data=axes_vals[k])

        print(f"# Wrote {fn}.")

        return data

    @property
    def tab_sed_synth(self):
        if not hasattr(self, '_tab_sed_synth'):
            self._tab_sed_synth = self.generate_sed_tables()
        return self._tab_sed_synth

    def get_lum_for_sfh(self, x=1600., window=1, Z=None, age=None, band=None,
        units='Angstroms', units_out='erg/s/Hz', raw=False, nebular_only=False,
        **kwargs):
        """
        Overhaul usual get_lum_per_sfr by reading from lookup table and
        interpolating to gridpoint appropriate for given SFH (defined by kwargs).
        """

        assert age is not None, "Must provide `age` as time since Big Bang!"

        sed_tab = self.generate_sed_tables()
