"""

NebularEmission.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Sun 21 Jul 2019 14:31:38 AEST

Description:

"""

import numpy as np
from ares.util import ParameterFile, read_lit
from ares.physics.Hydrogen import Hydrogen
from ares.physics.RateCoefficients import RateCoefficients
from ares.physics.Constants import h_p, c, k_B, erg_per_ev, E_LyA, E_LL, Ryd, \
    ev_per_hz, nu_alpha, m_p

try:
    from scipy.special import kn as bessel_2
except ImportError:
    pass

class NebularEmission(object):
    def __init__(self, cosm=None, **kwargs):
        self.pf = ParameterFile(**kwargs)
        self.cosm = cosm

    @property
    def coeff(self):
        if not hasattr(self, '_coeff'):
            self._coeff = RateCoefficients(grid=None)#, rate_src=rate_src,
                    #recombination=recombination, interp_rc=interp_rc)
        return self._coeff

    @property
    def wavelengths(self):
        if not hasattr(self, '_wavelengths'):
            raise AttributeError('Must set `wavelengths` by hand.')
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = value

    @property
    def energies(self):
        if not hasattr(self, '_energies'):
            self._energies = h_p * c / (self.wavelengths / 1e8) / erg_per_ev
        return self._energies

    @property
    def Emin(self):
        return np.min(self.energies)

    @property
    def Emax(self):
        return np.max(self.energies)

    @property
    def frequencies(self):
        if not hasattr(self, '_frequencies'):
            self._frequencies = c / (self.wavelengths / 1e8)
        return self._frequencies

    @property
    def dwdn(self):
        if not hasattr(self, '_dwdn'):
            self._dwdn = self.wavelengths**2 / (c * 1e8)
        return self._dwdn

    @property
    def dE(self):
        if not hasattr(self, '_dE'):
            tmp = np.abs(np.diff(self.energies))
            self._dE = np.concatenate((tmp, [tmp[-1]]))
        return self._dE

    @property
    def hydr(self):
        if not hasattr(self, '_hydr'):
            self._hydr = Hydrogen(pf=self.pf, cosm=self.cosm, **self.pf)
        return self._hydr

    @property
    def _gaunt_avg_fb(self):
        # Karzas & Latter 1961
        return 1.05
    @property
    def _gaunt_avg_ff(self):
        # Karzas & Latter 1961
        return 1.1

    @property
    def _f_k(self):
        return 5.44436e-39

    @property
    def _gamma_fb(self):
        if not hasattr(self, '_gamma_fb_'):
            # Assuming fully-ionized hydrogen-only nebular for now.
            _sum = np.zeros_like(self.frequencies)
            for n in np.arange(2, 15., 1.):
                _xn = Ryd / k_B / self.pf['source_nebular_Tgas'] / n**2
                ok = (Ryd / h_p / n**2) < self.frequencies
                _sum[ok==1] += _xn * (np.exp(_xn) / n) * self._gaunt_avg_fb

            self._gamma_fb_ = self._f_k * _sum

        return self._gamma_fb_

    @property
    def _gamma_ferland(self):
        if not hasattr(self, '_gamma_ferland_'):

            e_ryd, T10, T20 = read_lit('ferland1980')._load()

            assert 1e4 <= self.pf['source_nebular_Tgas'] <= 2e4

            if self.pf['source_nebular_Tgas'] == 1e4:
                coeff = T10
            elif self.pf['source_nebular_Tgas'] == 2e4:
                coeff = T20
            else:
                raise NotImplemented('No interpolation scheme yet.')

            nrg_Ryd = self.energies / (Ryd / erg_per_ev)
            self._gamma_ferland_ = np.zeros_like(self.energies)
            for i in range(len(e_ryd)-1):

                if i % 2 != 0:
                    continue

                x = np.array([e_ryd[i], e_ryd[i+1]])
                y = np.log10([coeff[i], coeff[i+1]])

                m = (y[1] - y[0]) / (x[1] - x[0])

                # Energies stored in descending order in Ferland table.
                ok = np.logical_and(nrg_Ryd < x[0], nrg_Ryd >= x[1])

                self._gamma_ferland_[ok==1] = \
                    10**(m * (nrg_Ryd[ok==1] - x[0]) + y[0])

            self._gamma_ferland_ /= self._p_of_c

        return self._gamma_ferland_

    @property
    def _gamma_ff(self):
        if self.pf['source_nebular_lookup'] == 'ferland1980':
            return 0.0
        return self._f_k * self._gaunt_avg_ff

    @property
    def _p_of_c(self):
        """
        The continuum emission coefficient, over (ne * np * gamma_c).

        .. note :: This is Eq. 6.22 in Dopita & Sutherland (2003).
        .. note :: Pretty sure their F_k in that equation should be f_k.
        .. note :: We don't need the ne * np bit thanks to our assumption
            of photo-ionization equilibrium.

        """

        if not hasattr(self, '_p_of_c_'):
            hnu = self.energies * erg_per_ev
            kT = k_B * self.pf['source_nebular_Tgas']
            self._p_of_c_ = 4. * np.pi * np.exp(-hnu / kT) \
                / np.sqrt(self.pf['source_nebular_Tgas'])

        return self._p_of_c_

    @property
    def _prob_2phot(self):
        # This is Eq. 22 of Fernandez & Komatsu 2006, a fit to the measurements
        # of Brown & Matthews (1970; their Table 4)
        if not hasattr(self, '_prob_2phot_'):

            x = self.energies / E_LyA

            P = np.zeros_like(self.energies)
            # Fernandez & Komatsu 2006
            P[x<1.] = 1.307 \
                    - 2.627 * (x[x<1.] - 0.5)**2 \
                    + 2.563 * (x[x<1.] - 0.5)**4 \
                    - 51.69 * (x[x<1.] - 0.5)**6

            self._prob_2phot_ = P

        return self._prob_2phot_

    @property
    def _ew_wrt_hbeta(self):
        if not hasattr(self, '_ew_wrt_hbeta_'):
            i11 = read_lit('inoue2011')

            waves, ew, ew_std = i11._load(self.pf['source_Z'])

            self._ew_wrt_hbeta_ = waves, ew, ew_std

        return self._ew_wrt_hbeta_

    def f_rep(self, spec, Tgas=2e4, channel='ff', net=False):
        """
        Fraction of photons reprocessed into different channels.

            .. note :: This carries units of Hz^{-1}.
        """

        erg_per_phot = self.energies * erg_per_ev

        Tgas = self.pf['source_nebular_Tgas']
        #A_H = 1. / (1. + self.cosm.y)
        #u = 143.9 / self.wavelengths / (Tgas / 1e6)
        #ne = 1.
        alpha = self.coeff.RadiativeRecombinationRate(0, Tgas)

        #gamma_pre = 2.051e-22 * (Tgas / 1e6)**-0.5 * self.wavelengths**-2. \
        #    * np.exp(-u) * self.dwdn

        ##
        # Read from source?
        lookup = self.pf['source_nebular_lookup']
        if (lookup is not None) and (channel != 'tp'):
            if self.pf['source_nebular_lookup'] == 'ferland1980':
                # Ferland 1980 results have ff+fb as package deal.
                if channel == 'fb':
                    return self._p_of_c * self._gamma_ferland / alpha
                else:
                    return 0.0
            else:
                raise NotImplemented('help')

        # Compute ourselves
        if channel == 'ff':
            frep = self._p_of_c * self._gamma_ff / alpha
        elif channel == 'fb':
            frep = self._p_of_c * self._gamma_fb / alpha
        elif channel == 'tp':
            frep = 2. * self.energies * erg_per_ev * self._prob_2phot / nu_alpha
        else:
            raise NotImplemented("Do not recognize channel `{}`".format(channel))

        if net:
            return np.trapz(frep[-1::-1] * nu[-1::-1], x=np.log(nu[-1::-1]))
        else:
            return frep

    def get_ion_lum(self, spec, species=0):
        if species == 0:
            ion = self.energies >= E_LL
        elif species == 1:
            ion = self.energies >= 24.6
        elif species == 2:
            ion = self.energies >= 4 * E_LL
        else:
            raise NotImplemented('help')

        gt0 = spec > 0
        ok = np.logical_and(ion, gt0)

        return np.trapz(spec[ok==1][-1::-1] * self.frequencies[ok==1][-1::-1],
            x=np.log(self.frequencies[ok==1][-1::-1]))

    def get_ion_num(self, spec, species=0):
        if species == 0:
            ion = self.energies >= E_LL
        elif species == 1:
            ion = self.energies >= 24.6
        elif species == 2:
            ion = self.energies >= 4 * E_LL
        else:
            raise NotImplemented('help')

        gt0 = spec > 0
        ok = np.logical_and(ion, gt0)

        erg_per_phot = self.energies[ok==1][-1::-1] * erg_per_ev
        freq = self.frequencies[ok==1][-1::-1]

        integ = spec[ok==1][-1::-1] * freq / erg_per_phot

        return np.trapz(integ, x=np.log(freq))

    def get_ion_Eavg(self, spec, species=0):
        return self.get_ion_lum(spec, species) \
            / self.get_ion_num(spec, species) / erg_per_ev

    def Continuum(self, spec):
        """
        Add together nebular continuum contributions, i.e., free-free,
        free-bound, and two-photon.

        Parameters
        ----------
        Return L_\nu in [erg/s/Hz]

        """

        fesc = self.pf['source_fesc']
        Tgas = self.pf['source_nebular_Tgas']
        cBd = self.pf['source_nebular_caseBdeparture']
        flya = 2. / 3.
        erg_per_phot = self.energies * erg_per_ev

        # This is in [#/s]
        Nion = self.get_ion_num(spec)

        # Reprocessing fraction in [erg/Hz]
        frep_ff = self.f_rep(spec, Tgas, 'ff')
        frep_fb = self.f_rep(spec, Tgas, 'fb')
        frep_tp = (1. - flya) * self.f_rep(spec, Tgas, 'tp')

        # Amount of UV luminosity absorbed in ISM
        #Nabs = Nion * (1. - fesc)
        if self.pf['source_prof_1h'] is not None:
            Nabs = Nion * (1. - fesc)
        else:
            Nabs = Nion

        tot = np.zeros_like(self.wavelengths)
        if self.pf['source_nebular_ff']:
            tot += frep_ff * Nabs
        if self.pf['source_nebular_fb']:
            tot += frep_fb * Nabs
        if self.pf['source_nebular_2phot']:
            tot += frep_tp * Nabs * cBd

        return tot

    def LineEmission(self, spec):
        """
        Add as many nebular lines as we have models for.

        Parameters
        ----------
        Return L_\nu in [erg/s/Hz]

        """


        fesc = self.pf['source_fesc']
        Tgas = self.pf['source_nebular_Tgas']
        flya = 2. / 3.
        erg_per_phot = self.energies * erg_per_ev

        # This is in [#/s]
        Nion = self.get_ion_num(spec)

        # Amount of UV luminosity absorbed in ISM
        #Nabs = Nion * (1. - fesc)
        if self.pf['source_prof_1h'] is not None:
            Nabs = Nion * (1. - fesc)
        else:
            Nabs = Nion

        #tot = np.zeros_like(self.wavelengths)

        #i_lya = np.argmin(np.abs(self.energies - E_LyA))

        #tot[i_lya] = spec[i_lya] * 10
        if self.pf['source_nebular'] == 2:
            tot =  self.LymanSeries(spec)
            tot += self.BalmerSeries(spec)
        elif self.pf['source_nebular'] == 3:
            _tot = self.BalmerSeries(spec)
            Hb = _tot[np.argmin(np.abs(6569 - self.wavelengths))]

            tot = np.zeros_like(self.wavelengths)

            waves, ew, ew_std = self._ew_wrt_hbeta
            for i, wave in enumerate(waves):
                j = np.argmin(np.abs(wave - self.wavelengths))
                tot[j] = ew[i] * Hb

        else:
            raise NotImplementedError('Unrecognized source_nebular option!')

        return tot

    def LymanSeries(self, spec):
        return self.HydrogenLines(spec, ninto=1)

    def BalmerSeries(self, spec):
        """
        Follow Inoue (2011) with coefficients from Osterbrock & Ferland (2006).
        """

        return self.HydrogenLines(spec, ninto=2)

    @property
    def _jnu_wrt_hbeta(self):
        if not hasattr(self, '_jnu_wrt_hbeta_'):
            _Tg = self.pf['source_nebular_Tgas']

            if _Tg == 1e4:
                self._jnu_wrt_hbeta_ = [2.86, 1., 0.468, 0.259, 0.159, 0.105]
            elif _Tg == 2e4:
                # This is for n = 100 cm^-3
                self._jnu_wrt_hbeta_ = [2.75, 1., 0.475, 0.264, 0.163, 0.107]
            else:
                raise NotImplemented('help')

            self._jnu_wrt_hbeta_ = np.array(self._jnu_wrt_hbeta_)

        return self._jnu_wrt_hbeta_

    @property
    def _gamma_hbeta(self):
        return 1.23e-25 * (self.pf['source_nebular_Tgas'] / 1e4)**-0.9

    def HydrogenLines(self, spec, ninto=1):
        """
        Return spectrum containing only H lines from transitions into `ninto`.
        """

        assert ninto in [1,2], "Only Lyman and Balmer series implemented so far."

        neb = np.zeros_like(self.wavelengths)
        nrg = self.energies
        freq = self.frequencies

        fesc = self.pf['source_fesc']
        _Tg = self.pf['source_nebular_Tgas']
        _cBd = self.pf['source_nebular_caseBdeparture']

        ion = nrg >= E_LL
        gt0 = spec > 0
        ok = np.logical_and(ion, gt0)

        # This will be in [#/s]
        Nion = self.get_ion_num(spec)
        #Nabs = Nion * (1. - fesc)
        if self.pf['source_prof_1h'] is not None:
            Nabs = Nion * (1. - fesc)
        else:
            Nabs = Nion

        sigm = nu_alpha * np.sqrt(k_B * _Tg / m_p / c**2) * h_p

        fout = np.zeros_like(self.wavelengths)
        for i, n in enumerate(range(ninto+1, ninto+7)):

            # Determine resulting photons energy
            En = self.hydr.BohrModel(ninto=ninto, nfrom=n)

            # Need to generalize
            if ninto == 1:
                coeff = 2. / 3.
            elif ninto == 2:
                # Follow Inoue (2011) and compute H-beta, scale other
                # lines using Osterbrock & Ferland (Table 4.4)
                Lbeta = Nabs * self._gamma_hbeta \
                    / self.coeff.RadiativeRecombinationRate(0, _Tg)

                coeff = Lbeta / (En * erg_per_ev * Nabs)
                coeff *= self._jnu_wrt_hbeta[i]

            else:
                raise NotImplemented('help')

            #prof = np.exp(-0.5 * (nrg - E_LyA)**2 / 2. / sigm**2) \
            #     / np.sqrt(2. * np.pi) * erg_per_ev * ev_per_hz / sigm

            # Find correct element in array. Assume delta function
            loc = np.argmin(np.abs(nrg - En))

            # Need to get Hz^-1 units; `freq` in descending order
            dnu = freq[loc] - freq[loc+1]

            # In erg/s
            Lline = Nabs * coeff * En * erg_per_ev
            Lline *= _cBd

            # Currently assuming line is unresolved.
            # Should really do this based on some physical argument.
            fout[loc] = Lline / dnu

            # Only know how to do Ly-a for now.
            if ninto == 1:
                break

        return fout
