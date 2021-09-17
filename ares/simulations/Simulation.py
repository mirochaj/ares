import os
import copy
import pickle
import numpy as np
from types import FunctionType
from ..static import Fluctuations
from .Global21cm import Global21cm
from ..physics.HaloModel import HaloModel
from ..util import ParameterFile, ProgressBar
#from ..analysis.BlobFactory import BlobFactory
from ..analysis.PowerSpectrum import PowerSpectrum as AnalyzePS
from ..physics.Constants import cm_per_mpc, c, s_per_yr, erg_per_ev, \
    erg_per_s_per_nW, h_p, cm_per_m

class Simulation(AnalyzePS): # pragma: no cover
    def __init__(self, pf=None, **kwargs):
        """ Set up a power spectrum calculation. """

        # See if this is a tanh model calculation
        if 'problem_type' not in kwargs:
            kwargs['problem_type'] = 102

        self.tab_kwargs = kwargs

        if pf is None:
            self.pf = ParameterFile(**self.tab_kwargs)
        else:
            self.pf = pf

    @property
    def gs(self):
        if not hasattr(self, '_gs'):
            self._gs = Global21cm(**self.tab_kwargs)
        return self._gs

    @gs.setter
    def gs(self, value):
        """ Set global 21cm instance by hand. """
        self._gs = value

    @property
    def history(self):
        if not hasattr(self, '_history'):
            self._history = {}
        return self._history

    @property
    def mean_intensity(self):
        if not hasattr(self, '_mean_intensity'):
            self._mean_intensity = self.gs.medium.field
        return self._mean_intensity

    def _cache_ebl(self, waves=None, wave_units='mic', flux_units='SI',
        pops=None):
        if not hasattr(self, '_cache_ebl_'):
            self._cache_ebl_ = {}

        # Could be clever and convert units here.
        if (wave_units, flux_units, pops) in self._cache_ebl_:
            _waves, _fluxes = self._cache_ebl_[(wave_units, flux_units, pops)]
            if waves is None:
                return _waves, _fluxes
            elif _waves.size == waves.size:
                if np.all(_waves == waves):
                    return _waves, _fluxes

        return None

    def get_ebl(self, waves=None, wave_units='mic', flux_units='SI',
        pops=None):
        """
        Return the extragalactic background light (EBL) over all wavelengths.

        Parameters
        ----------
        waves : np.ndarray
            If provided, will interpolate fluxes from each source population
            onto common grid.
        wave_units : str
            Current options: 'eV', 'microns', 'Ang'
        flux_units : str
            Current options: 'cgs', 'SI'

        .. note :: 'SI' units means nW / m^2 / sr, 'cgs' means erg/s/Hz/sr.

        Returns
        -------
        Tuple containing (observed wavelength, observed flux). Note that if
        `waves` is not None, the returned flux array will have shape
        (num source populations, num waves). If not, it will be 1-D with
        the same length as output observed wavelengths.

        """

        cached_result = self._cache_ebl(waves, wave_units, flux_units,
            pops)
        if cached_result is not None:
            return cached_result

        if not self.mean_intensity._run_complete:
            self.mean_intensity.run()

        if waves is not None:
            all_x = waves
            all_y = np.zeros((len(self.pops), len(waves)))
        else:
            all_x = []
            all_y = []

        for i in range(len(self.pops)):
            if pops is not None:
                if i not in pops:
                    continue

            zf = self.pops[i].zdead
            E, flux = self.mean_intensity.flux_today(zf=None, popids=i,
                units=flux_units)

            if wave_units.lower() == 'ev':
                x = E
            elif wave_units.lower().startswith('mic'):
                x = 1e4 * c / (E * erg_per_ev / h_p)
            elif wave_units.lower().startswith('ang'):
                x = 1e8 * c / (E * erg_per_ev / h_p)
            else:
                raise NotImplemented('Unrecognized `wave_units`={}'.format(
                    wave_units
                ))

            lo, hi = x.min(), x.max()

            # Check for overlap, warn user if they should use `waves`
            if i > 0:
                lo_all, hi_all = np.min(all_x), np.max(all_x)

                is_overlap = (lo_all <= lo <= hi_all) or (lo_all <= hi <= hi_all)
                if waves is None and is_overlap:
                    print("# WARNING: Overlap in spectral coverage of population #{}. Consider using `waves` keyword argument.".format(i))

            #

            # Either save EBL as potentially-disjointed array of energies
            # OR interpolate to common wavelength grid if `waves` is not None.
            if waves is not None:
                if not np.all(np.diff(x) > 0):
                    all_y[i,:] = np.exp(np.interp(np.log(waves[-1::-1]),
                        np.log(x[-1::-1]), np.log(flux[-1::-1])))[-1::-1]
                else:
                    all_y[i,:] = np.exp(np.interp(np.log(waves), np.log(x),
                        np.log(flux)))
            else:
                all_x.extend(E)
                all_y.extend(flux)

                # Put a gap between chunks to avoid weird plotting artifacts
                all_x.append(-np.inf)
                all_y.append(-np.inf)

        x = np.array(all_x)
        y = np.array(all_y)

        if pops is None:
            hist = self.history # poke
            self._history['ebl'] = x, y

        return x, y

    def get_ps_galaxies(self, scales, waves, wave_units='mic',
        scale_units='arcmin', flux_units='SI', dimensionless=False, pops=None,
        **kwargs):
        """
        Compute power spectrum at some observed wavelength.

        Parameters
        ----------
        scales : int, float, np.ndarray

        waves : int, float, np.ndarray
            Wavelengths at which to compute power spectra in `wave_units`.
            Note that if 2-D, must have shape (number of bins, 2), in which
            case the power spectra will be computed in series of bandpasses.

        wave_units : str
            Current options: 'eV', 'microns', 'Ang'
        flux_units : str
            Current options: 'cgs', 'SI'
        scale_units : str
            Current options: 'arcmin', 'arcsec', 'degrees', 'ell'

        Returns
        -------
        Tuple containing (scales, 2 pi / scales or l*l(+z),
            waves, power spectra).

        Note that the power spectra are return as 2-D arrays with shape
        (len(scales), len(waves))

        """

        # Make sure we do mean background first in case LW feedback is on.
        if not self.mean_intensity._run_complete:
            self.mean_intensity.run()

        # Make sure things are arrays
        if type(scales) != np.ndarray:
            scales = np.array([scales])
        if type(waves) != np.ndarray:
            waves = np.array([waves])

        if waves.ndim == 2:
            assert waves.shape[1] == 2, \
                "If `waves` is 2-D, must have shape (num waves, 2)."

        # Prep scales
        if scale_units.lower() in ['l', 'ell']:
            scales_inv = np.sqrt(scales * (scales + 1))
        else:
            if scale_units.lower().startswith('deg'):
                scale_rad = scales * (np.pi / 180.)
            elif scale_units.lower() == 'arcmin':
                scale_rad = (scales / 60.) * (np.pi / 180.)
            elif scale_units.lower() == 'arcsec':
                scale_rad = (scales / 3600.) * (np.pi / 180.)
            else:
                raise NotImplemented('help')

            scales_inv = np.pi / scale_rad

        if wave_units.lower().startswith('mic'):
            pass
        else:
            raise NotImplemented('help')

        # Do some error-handling if waves is 2-D: means the user provided
        # bandpasses instead of a set of wavelengths.

        ps = np.zeros((len(self.pops), len(scales), len(waves)))

        for i, pop in enumerate(self.pops):

            if pops is not None:
                if i not in pops:
                    continue

            for j, wave in enumerate(waves):
                ps[i,:,j] = pop.get_ps_obs(scales, wave_obs=wave,
                    scale_units=scale_units, **kwargs)


        # Modify PS units before return
        if flux_units.lower() == 'si':
            ps *= cm_per_m**4 / erg_per_s_per_nW**2

        if pops is None:
            hist = self.history # poke
            self._history['ps_nirb'] = scales, scales_inv, waves, ps

        if dimensionless:
            ps *= scales_inv[:,None]**2 / 2. / np.pi**2

        return scales, scales_inv, waves, ps

    @property
    def pops(self):
        return self.gs.medium.field.pops

    @property
    def grid(self):
        return self.gs.medium.field.grid

    @property
    def hydr(self):
        return self.grid.hydr

    @property
    def field(self):
        if not hasattr(self, '_field'):
            self._field = Fluctuations(**self.tab_kwargs)
        return self._field

    @property
    def halos(self):
        if not hasattr(self, '_halos'):
            self._halos = self.pops[0].halos
        return self._halos

    @property
    def tab_z(self):
        if not hasattr(self, '_tab_z'):
            self._tab_z = np.array(np.sort(self.pf['ps_output_z'])[-1::-1],
                dtype=np.float64)
        return self._tab_z

    def run(self):
        """
        Run everything we can.
        """
        pass

    def run_ebl(self):
        pass

    def run_nirb_ps(self):
        pass

    def get_ps_21cm(self):
        if 'ps_21cm' not in self.history:
            self.run_ps_21cm()

        return self.history['ps_21cm']

    def get_gs_21cm(self):
        if 'gs_21cm' not in self.history:
            self.gs.run()

        return self.history['gs_21cm']

    def run_gs_21cm(self):
        self.gs.run()
        self.history['gs_21cm'] = self.gs.history

    def run_ps_21cm(self, z=None, k=None):
        """
        Run a simulation, compute power spectrum at each redshift.

        Returns
        -------
        Nothing: sets `history` attribute.

        """

        if z is None:
            z = self.tab_z
        if k is None:
            k = self.tab_k

        # First, run global signal.
        self.run_gs_21cm()

        N = z.size
        pb = self.pb = ProgressBar(N, use=self.pf['progress_bar'],
            name='ps-21cm')

        all_ps = []
        for i, (z, data) in enumerate(self._step_ps_21cm()):

            # Do stuff
            all_ps.append(data.copy())

            if i == 0:
                keys = data.keys()

            if not pb.has_pb:
                pb.start()

            pb.update(i)

        pb.finish()

        self.all_ps = all_ps

        hist = {}
        for key in keys:

            is2d_k = key.startswith('ps')
            is2d_R = key.startswith('jp') or key.startswith('ev') \
                  or key.startswith('cf')
            is2d_B = (key in ['n_i', 'm_i', 'r_i', 'delta_B', 'bsd'])

            if is2d_k:
                tmp = np.zeros((len(self.tab_z), len(self.tab_k)))
            elif is2d_R:
                tmp = np.zeros((len(self.tab_z), len(self.tab_R)))
            elif is2d_B:
                tmp = np.zeros((len(self.tab_z), len(all_ps[0]['r_i'])))
            else:
                tmp = np.zeros_like(self.tab_z)

            for i, z in enumerate(self.tab_z):
                if key not in all_ps[i].keys():
                    continue

                tmp[i] = all_ps[i][key]

            hist[key] = tmp.copy()

        poke = self.history

        self.history['ps_21cm'] = hist
        self.history['ps_21cm']['z'] = self.tab_z
        self.history['ps_21cm']['k'] = self.tab_k
        self.history['ps_21cm']['R'] = self.tab_R

    @property
    def tab_k(self):
        """
        Wavenumbers to output power spectra.

        .. note :: Can be far more crude than native resolution of
            matter power spectrum.

        """

        if not hasattr(self, '_k'):
            if self.pf['ps_output_k'] is not None:
                self._k = self.pf['ps_output_k']
            else:
                lnk1 = self.pf['ps_output_lnkmin']
                lnk2 = self.pf['ps_output_lnkmax']
                dlnk = self.pf['ps_output_dlnk']
                self._k = np.exp(np.arange(lnk1, lnk2+dlnk, dlnk))

        return self._k

    @property
    def tab_R(self):
        """
        Scales on which to compute correlation functions.

        .. note :: Can be more crude than native resolution of matter
            power spectrum, however, unlike `self.tab_k`, the resolution of
            this quantity matters when converting back to power spectra,
            since that operation requires an integral over R.

        """
        if not hasattr(self, '_R'):
            if self.pf['ps_output_R'] is not None:
                self._R = self.pf['ps_output_R']
            else:
                lnR1 = self.pf['ps_output_lnRmin']
                lnR2 = self.pf['ps_output_lnRmax']
                dlnR = self.pf['ps_output_dlnR']
                #lnR = np.log(self.halos.tab_R)

                self._R = np.exp(np.arange(lnR1, lnR2+dlnR, dlnR))

        return self._R

    @property
    def tab_Mmin(self):
        if not hasattr(self, '_tab_Mmin'):
            self._tab_Mmin = np.ones_like(self.halos.tab_z) * np.inf
            for j, pop in enumerate(self.pops):
                self._tab_Mmin = np.minimum(self._tab_Mmin, pop._tab_Mmin)

        return self._tab_Mmin


    @property
    def tab_zeta(self):
        return self._tab_zeta

    @tab_zeta.setter
    def tab_zeta(self, value):
        self._tab_zeta = value

    def _step_ps_21cm(self):
        """
        Generator for the power spectrum.
        """

        # Set a few things before we get moving.
        self.field.tab_Mmin = self.tab_Mmin

        for i, z in enumerate(self.tab_z):

            data = {}

            ##
            # First, loop over populations and determine total
            # UV and X-ray outputs.
            ##

            # Prepare for the general case of Mh-dependent things
            Nion = np.zeros_like(self.halos.tab_M)
            Nlya = np.zeros_like(self.halos.tab_M)
            fXcX = np.zeros_like(self.halos.tab_M)
            zeta_ion = zeta = np.zeros_like(self.halos.tab_M)
            zeta_lya = np.zeros_like(self.halos.tab_M)
            zeta_X = np.zeros_like(self.halos.tab_M)
            #Tpro = None
            for j, pop in enumerate(self.pops):
                pop_zeta = pop.IonizingEfficiency(z=z)

                if pop.is_src_ion:

                    if type(pop_zeta) is tuple:
                        _Mh, _zeta = pop_zeta
                        zeta += np.interp(self.halos.tab_M, _Mh, _zeta)
                        Nion += pop.src.Nion
                    else:
                        zeta += pop_zeta
                        Nion += pop.pf['pop_Nion']
                        Nlya += pop.pf['pop_Nlw']

                    zeta = np.maximum(zeta, 1.) # why?

                if pop.is_src_heat:
                    pop_zeta_X = pop.HeatingEfficiency(z=z)
                    zeta_X += pop_zeta_X

                if pop.is_src_lya:
                    Nlya += pop.pf['pop_Nlw']
                    #Nlya += pop.src.Nlw

            # Only used if...ps_lya_method==0?
            zeta_lya += zeta * (Nlya / Nion)

            ##
            # Make scalar if it's a simple model
            ##
            if np.all(np.diff(zeta) == 0):
                zeta = zeta[0]
            if np.all(np.diff(zeta_X) == 0):
                zeta_X = zeta_X[0]
            if np.all(np.diff(zeta_lya) == 0):
                zeta_lya = zeta_lya[0]

            self.field.zeta = zeta
            self.field.zeta_X = zeta_X

            self.tab_zeta = zeta

            ##
            # Figure out scaling from ionized regions to heated regions.
            # Right now, only constant (relative) scaling is allowed.
            ##
            asize = self.pf['bubble_shell_asize_zone_0']
            if self.pf['ps_include_temp'] and asize is not None:

                self.field.is_Rs_const = False

                if type(asize) is FunctionType:
                    R_s = lambda R, z: R + asize(z)
                else:
                    R_s = lambda R, z: R + asize

            elif self.pf['ps_include_temp'] and self.pf['ps_include_ion']:
                fvol = self.pf["bubble_shell_rvol_zone_0"]
                frad = self.pf['bubble_shell_rsize_zone_0']

                assert (fvol is not None) + (frad is not None) <= 1

                if fvol is not None:
                    assert frad is None

                    # Assume independent variable is redshift for now.
                    if type(fvol) is FunctionType:
                        frad = lambda z: (1. + fvol(z))**(1./3.) - 1.
                        self.field.is_Rs_const = False
                    else:
                        frad = lambda z: (1. + fvol)**(1./3.) - 1.

                elif frad is not None:
                    if type(frad) is FunctionType:
                        self.field.is_Rs_const = False
                    else:
                        frad = lambda z: frad
                else:
                    # If R_s = R_s(z), must re-compute overlap volumes on each
                    # step. Should set attribute if this is the case.
                    raise NotImplemented('help')

                R_s = lambda R, z: R * (1. + frad(z))


            else:
                R_s = lambda R, z: None
                Th = None

            # Must be constant, for now.
            Th = self.pf["bubble_shell_ktemp_zone_0"]

            self.tab_R_s = R_s
            self.Th = Th


            ##
            # First: some global quantities we'll need
            ##
            Tcmb = self.cosm.TCMB(z)
            hist = self.gs.history

            Tk = np.interp(z, hist['z'][-1::-1], hist['igm_Tk'][-1::-1])
            Ts = np.interp(z, hist['z'][-1::-1], hist['igm_Ts'][-1::-1])
            Ja = np.interp(z, hist['z'][-1::-1], hist['Ja'][-1::-1])
            xHII, ne = [0] * 2

            xa = self.hydr.RadiativeCouplingCoefficient(z, Ja, Tk)
            xc = self.hydr.CollisionalCouplingCoefficient(z, Tk)
            xt = xa + xc

            # Won't be terribly meaningful if temp fluctuations are off.
            C = self.field.TempToContrast(z, Th=Th, Tk=Tk, Ts=Ts, Ja=Ja)
            data['c'] = C
            data['Ts'] = Ts
            data['Tk'] = Tk
            data['xa'] = xa
            data['Ja'] = Ja



            # Assumes strong coupling. Mapping between temperature
            # fluctuations and contrast fluctuations.
            #Ts = Tk


            # Add beta factors to dictionary
            for f1 in ['x', 'd', 'a']:
                func = self.hydr.__getattribute__('beta_%s' % f1)
                data['beta_%s' % f1] = func(z, Tk, xHII, ne, Ja)

            Qi_gs = np.interp(z, self.gs.history['z'][-1::-1],
                self.gs.history['cgm_h_2'][-1::-1])

            # Ionization fluctuations
            if self.pf['ps_include_ion']:

                Ri, Mi, Ni = self.field.BubbleSizeDistribution(z, ion=True)

                data['n_i'] = Ni
                data['m_i'] = Mi
                data['r_i'] = Ri
                data['delta_B'] = self.field._B(z, ion=True)
            else:
                Ri = Mi = Ni = None

            Qi = self.field.MeanIonizedFraction(z)

            Qi_bff = self.field.BubbleFillingFactor(z)

            xibar = Qi_gs


            # Save normalized copy of BSD for easy plotting in post
            dvdr = 4. * np.pi * Ri**2
            dmdr = self.cosm.mean_density0 * (1. + data['delta_B']) * dvdr
            dmdlnr = dmdr * Ri
            dndlnR = Ni * dmdlnr
            V = 4. * np.pi * Ri**3 / 3.
            data['bsd'] = V * dndlnR / Qi

            if self.pf['ps_include_temp']:
                # R_s=R_s(Ri,z)
                Qh = self.field.MeanIonizedFraction(z, ion=False)
                data['Qh'] = Qh
            else:
                data['Qh'] = Qh = 0.0

            # Interpolate global signal onto new (coarser) redshift grid.
            dTb_ps = np.interp(z, self.gs.history['z'][-1::-1],
                self.gs.history['dTb'][-1::-1])

            xavg_gs = np.interp(z, self.gs.history['z'][-1::-1],
                self.gs.history['xavg'][-1::-1])

            data['dTb'] = dTb_ps

            #data['dTb_bulk'] = np.interp(z, self.gs.history['z'][-1::-1],
            #    self.gs.history['dTb_bulk'][-1::-1])


            ##
            # Correct for fraction of ionized and heated volumes
            # and densities!
            ##
            if self.pf['ps_include_temp']:
                data['dTb_vcorr'] = None#(1 - Qh - Qi) * data['dTb_bulk'] \
                    #+ Qh * self.hydr.dTb(z, 0.0, Th)
            else:
                data['dTb_vcorr'] = None#data['dTb_bulk'] * (1. - Qi)

            if self.pf['ps_include_xcorr_ion_rho']:
                pass
            if self.pf['ps_include_xcorr_ion_hot']:
                pass

            # Just for now
            data['dTb0'] = data['dTb']
            data['dTb0_2'] = data['dTb0_1'] = data['dTb_vcorr']

            #if self.pf['include_ion_fl']:
            #    if self.pf['ps_rescale_Qion']:
            #        xibar = min(np.interp(z, self.pops[0].halos.z,
            #            self.pops[0].halos.fcoll_Tmin) * zeta, 1.)
            #        Qi = xibar
            #
            #        xibar = np.interp(z, self.mean_history['z'][-1::-1],
            #            self.mean_history['cgm_h_2'][-1::-1])
            #
            #    else:
            #        Qi = self.field.BubbleFillingFactor(z, zeta)
            #        xibar = 1. - np.exp(-Qi)
            #else:
            #    Qi = 0.



            #if self.pf['ps_force_QHII_gs'] or self.pf['ps_force_QHII_fcoll']:
            #    rescale_Q = True
            #else:
            #    rescale_Q = False

            #Qi = np.mean([QHII_gs, self.field.BubbleFillingFactor(z, zeta)])

            #xibar = np.interp(z, self.mean_history['z'][-1::-1],
            #    self.mean_history['cgm_h_2'][-1::-1])

            # Avoid divide by zeros when reionization is over
            if Qi == 1:
                Tbar = 0.0
            else:
                Tbar = data['dTb0_2']

            xbar = 1. - xibar
            data['Qi'] = Qi
            data['xibar'] = xibar
            data['dTb0'] = Tbar
            #data['dTb_bulk'] = dTb_ps / (1. - xavg_gs)

            ##
            # 21-cm fluctuations
            ##
            if self.pf['ps_include_21cm']:

                data['cf_21'] = self.field.CorrelationFunction(z,
                    R=self.tab_R, term='21', R_s=R_s(Ri,z), Ts=Ts, Th=Th,
                    Tk=Tk, Ja=Ja, k=self.tab_k)

                # Always compute the 21-cm power spectrum. Individual power
                # spectra can be saved by setting ps_save_components=True.
                data['ps_21'] = self.field.PowerSpectrumFromCF(self.tab_k,
                    data['cf_21'], self.tab_R,
                    split_by_scale=self.pf['ps_split_transform'],
                    epsrel=self.pf['ps_fht_rtol'],
                    epsabs=self.pf['ps_fht_atol'])

            # Should just do the above, and then loop over whatever is in
            # the cache and save also. If ps_save_components is True, then
            # FT everything we haven't already.
            for term in ['dd', 'ii', 'id', 'psi', 'phi']:
                # Should change suffix to _ev
                jp_1 = self.field._cache_jp(z, term)
                cf_1 = self.field._cache_cf(z, term)

                if (jp_1 is None and cf_1 is None) and (term not in ['psi', 'phi', 'oo']):
                    continue

                _cf = self.field.CorrelationFunction(z,
                    R=self.tab_R, term=term, R_s=R_s(Ri,z), Ts=Ts, Th=Th,
                    Tk=Tk, Ja=Ja, k=self.tab_k)

                data['cf_{}'.format(term)] = _cf.copy()

                if not self.pf['ps_output_components']:
                    continue

                data['ps_{}'.format(term)] = \
                    self.field.PowerSpectrumFromCF(self.tab_k,
                    data['cf_{}'.format(term)], self.tab_R,
                    split_by_scale=self.pf['ps_split_transform'],
                    epsrel=self.pf['ps_fht_rtol'],
                    epsabs=self.pf['ps_fht_atol'])

            # Always save the matter correlation function.
            data['cf_dd'] = self.field.CorrelationFunction(z,
                term='dd', R=self.tab_R)

            yield z, data

    def save(self, prefix, suffix='pkl', clobber=False, fields=None):
        """
        Save results of calculation. Pickle parameter file dict.

        Notes
        -----
        1) will save files as prefix.history.suffix and prefix.parameters.pkl.
        2) ASCII files will fail if simulation had multiple populations.

        Parameters
        ----------
        prefix : str
            Prefix of save filename
        suffix : str
            Suffix of save filename. Can be hdf5 (or h5) or pkl.
            Anything else will be assumed to be ASCII format (e.g., .txt).
        clobber : bool
            Overwrite pre-existing files of same name?

        """

        self.gs.save(prefix, clobber=clobber, fields=fields)

        fn = '%s.fluctuations.%s' % (prefix, suffix)

        if os.path.exists(fn):
            if clobber:
                os.remove(fn)
            else:
                raise IOError('%s exists! Set clobber=True to overwrite.' % fn)

        if suffix == 'pkl':
            f = open(fn, 'wb')
            pickle.dump(self.history._data, f)
            f.close()

            try:
                f = open('%s.blobs.%s' % (prefix, suffix), 'wb')
                pickle.dump(self.blobs, f)
                f.close()

                if self.pf['verbose']:
                    print('Wrote {}.blobs.{}'.format(prefix, suffix))
            except AttributeError:
                print('Error writing {}.blobs.{}'.format(prefix, suffix))

        elif suffix in ['hdf5', 'h5']:
            import h5py

            f = h5py.File(fn, 'w')
            for key in self.history:
                if fields is not None:
                    if key not in fields:
                        continue
                f.create_dataset(key, data=np.array(self.history[key]))
            f.close()

        # ASCII format
        else:
            f = open(fn, 'w')
            print >> f, "#",

            for key in self.history:
                if fields is not None:
                    if key not in fields:
                        continue
                print >> f, '%-18s' % key,

            print >> f, ''

            # Now, the data
            for i in range(len(self.history[key])):
                s = ''

                for key in self.history:
                    if fields is not None:
                        if key not in fields:
                            continue

                    s += '%-20.8e' % (self.history[key][i])

                if not s.strip():
                    continue

                print >> f, s

            f.close()

        if self.pf['verbose']:
            print('Wrote {}.fluctuations.{}'.format(prefix, suffix))

        #write_pf = True
        #if os.path.exists('%s.parameters.pkl' % prefix):
        #    if clobber:
        #        os.remove('%s.parameters.pkl' % prefix)
        #    else:
        #        write_pf = False
        #        print 'WARNING: %s.parameters.pkl exists! Set clobber=True to overwrite.' % prefix

        #if write_pf:
        #
        #    #pf = {}
        #    #for key in self.pf:
        #    #    if key in self.carryover_kwargs():
        #    #        continue
        #    #    pf[key] = self.pf[key]
        #
        #    if 'revision' not in self.pf:
        #        self.pf['revision'] = get_hg_rev()
        #
        #    # Save parameter file
        #    f = open('%s.parameters.pkl' % prefix, 'wb')
        #    pickle.dump(self.pf, f, -1)
        #    f.close()
        #
        #    if self.pf['verbose']:
        #        print 'Wrote %s.parameters.pkl' % prefix
        #
