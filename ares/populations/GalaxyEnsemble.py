"""

GalaxyEnsemble.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jul  6 11:02:24 PDT 2016

Description:

"""

import os
import gc
import time
import pickle
import numpy as np
from ..data import ARES
from ..util import read_lit
from ..util.Math import smooth
from ..util import ProgressBar
from ..obs.Survey import Survey
from .Halo import HaloPopulation
from ..physics import DustEmission
from scipy.optimize import curve_fit
from .GalaxyCohort import GalaxyCohort
from scipy.interpolate import interp1d
from scipy.integrate import quad, cumtrapz
from ..analysis.BlobFactory import BlobFactory
from ..obs.Photometry import get_filters_from_waves
from ..util.Stats import bin_e2c, bin_c2e, bin_samples, quantify_scatter
from ..static.SpectralSynthesis import SpectralSynthesis
from ..sources.SynthesisModelSBS import SynthesisModelSBS
from ..physics.Constants import rhodot_cgs, s_per_yr, s_per_myr, \
    g_per_msun, c, Lsun, cm_per_kpc, cm_per_pc, cm_per_mpc, E_LL, E_LyA, \
    erg_per_ev, h_p, lam_LyA

try:
    import h5py
except ImportError:
    pass

tiny_MAR = 1e-30
num_types = [int, float, np.float64]

def _linfunc(x, x0, p0, p1):
    return p0 * (x - x0) + p1

def _quadfunc2(x, x0, p0, p1):
    return p0 * (x - x0)**2 + p1

def _quadfunc3(x, x0, p0, p1, p2):
    return p0 * (x - x0)**2 + p1 * (x - x0) + p2

pars_affect_mars = ["pop_MAR", "pop_MAR_interp", "pop_MAR_corr"]
pars_affect_sfhs = ["pop_scatter_sfr", "pop_scatter_sfe", "pop_scatter_mar"]
pars_affect_sfhs.extend(["pop_update_dt", "pop_thin_hist"])

known_lines = 'Ly-a',
known_line_waves = lam_LyA,

class GalaxyEnsemble(HaloPopulation,BlobFactory):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # May not actually need this...
        HaloPopulation.__init__(self, **kwargs)

    def __dict__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]

        raise NotImplemented('help!')

    @property
    def tab_z(self):
        if not hasattr(self, '_tab_z'):
            h = self._gen_halo_histories()
            self._tab_z = h['z']
        return self._tab_z

    @tab_z.setter
    def tab_z(self, value):
        self._tab_z = value

    @property
    def tab_t(self):
        if not hasattr(self, '_tab_t'):
            # Array of times corresponding to all z' > z [years]
            self._tab_t = self.cosm.t_of_z(self.tab_z) / s_per_yr
        return self._tab_t

    @property
    def _b14(self):
        if not hasattr(self, '_b14_'):
            self._b14_ = read_lit('bouwens2014')
        return self._b14_

    @property
    def _c94(self):
        if not hasattr(self, '_c94_'):
            self._c94_ = read_lit('calzetti1994').windows
        return self._c94_

    @property
    def _nircam(self): # pragma: no cover
        if not hasattr(self, '_nircam_'):
            nircam = Survey(cam='nircam')
            nircam_M = nircam._read_nircam(filter_set='M')
            nircam_W = nircam._read_nircam(filter_set='W')

            self._nircam_ = nircam_M, nircam_W
        return self._nircam_

    @property
    def _roman(self): # pragma: no cover
        if not hasattr(self, '_roman_'):
            roman = Survey(cam='roman')
            roman_f = roman._read_roman()
            self._roman_ = roman_f
        return self._roman_

    def run(self):
        return

    def get_sfrd_in_mass_range(self, z, Mlo, Mhi=None):
        """
        Compute cumulative SFRD as a function of lower-mass bound.

        Returns
        -------
        Cumulative *FRACTION* of SFRD in halos above Mh.
        """

        iz = np.argmin(np.abs(z - self.histories['z']))
        _Mh = self.histories['Mh'][:,iz]
        _sfr = self.histories['SFR'][:,iz]
        _w = self.histories['nh'][:,iz]

        if Mhi is None:
            Mhi = _Mh.max()

        ok = np.logical_and(_Mh >= Mlo, _Mh <= Mhi)
        SFRD = np.sum(_sfr[ok==1] * _w[ok==1]) / rhodot_cgs

        return SFRD

    def SFRD(self, z, Mmin=None):
        return self.get_sfrd(z, Mmin=Mmin)

    def get_sfrd(self, z, Mmin=None):
        """
        Will convert to internal cgs units.
        """

        if type(z) in [int, float, np.float64]:

            iz = np.argmin(np.abs(z - self.histories['z']))
            sfr = self.histories['SFR'][:,iz]
            w = self.histories['nh'][:,iz]

            if Mmin is not None:
                _Mh = self.histories['Mh'][:,iz]
                ok = _Mh >= Mmin
            else:
                ok = np.ones_like(sfr)

            # Need to eliminate redundant branches in merger tree
            if 'mask' in self.histories:
                mask = self.histories['mask'][:,iz]
                ok = np.logical_and(ok, np.logical_not(mask))

            # Really this is the number of galaxies that formed in a given
            # differential redshift slice.
            return np.sum(sfr[ok==1] * w[ok==1]) / rhodot_cgs
        else:
            sfrd = np.zeros_like(z)
            for k, _z in enumerate(z):

                iz = np.argmin(np.abs(_z - self.histories['z']))
                _sfr = self.histories['SFR'][:,iz]
                _w = self.histories['nh'][:,iz]

                if Mmin is not None:
                    _Mh = self.histories['Mh'][:,iz]
                    ok = _Mh >= Mmin
                else:
                    ok = np.ones_like(_sfr)

                # Need to eliminate redundant branches in merger tree
                if 'mask' in self.histories:
                    mask = self.histories['mask'][:,iz]
                    ok = np.logical_and(ok, np.logical_not(mask))

                sfrd[k] = np.sum(_sfr[ok==1] * _w[ok==1]) / rhodot_cgs

            return sfrd
        #return np.trapz(sfr[0:-1] * dw, dx=np.diff(Mh)) / rhodot_cgs

    def _sfrd_func(self, z):
        # This is a cheat so that the SFRD spline isn't constructed
        # until CALLED. Used only for tunneling (see `pop_tunnel` parameter).
        return self.SFRD(z)

    def tile(self, arr, thin, renorm=False):
        """
        Expand an array to `thin` times its size. Group elements such that
        neighboring bundles of `thin` elements are objects that formed
        at the same redshift.
        """
        if arr is None:
            return None

        if thin in [0, 1]:
            return arr

        assert thin % 1 == 0

        # First dimension: formation redshifts
        # Second dimension: observed redshifts / times
        #new = np.tile(arr, (int(thin), 1))
        new = np.repeat(arr, int(thin), axis=0)
        #N = arr.shape[0]
        #_new = np.tile(arr, int(thin) * N)
        #new = _new.reshape(N * int(thin), N)

        if renorm:
            return new / float(thin)
        else:
            return new

    def get_noise_normal(self, arr, sigma):
        noise = np.random.normal(scale=sigma, size=arr.size)
        return np.reshape(noise, arr.shape)

    def get_noise_lognormal(self, arr, sigma):
        lognoise = np.random.normal(scale=sigma, size=arr.size)
        #noise = 10**(np.log10(arr) + np.reshape(lognoise, arr.shape)) - arr
        noise = np.power(10, np.log10(arr) + np.reshape(lognoise, arr.shape)) \
              - arr
        return noise

    @property
    def tab_scatter_mar(self):
        if not hasattr(self, '_tab_scatter_mar'):
            self._tab_scatter_mar = np.random.normal(scale=sigma,
                size=np.product(self.tab_shape))
        return self._tab_scatter_mar

    @tab_scatter_mar.setter
    def tab_scatter_mar(self, value):
        assert value.shape == self.tab_shape
        self._tab_scatter_mar = value

    @property
    def tab_shape(self):
        if not hasattr(self, '_tab_shape'):
            raise AttributeError('help')
        return self._tab_shape

    @tab_shape.setter
    def tab_shape(self, value):
        self._tab_shape = value

    @property
    def _cache_halos(self):
        if not hasattr(self, '_cache_halos_'):
            self._cache_halos_ = self._gen_halo_histories()
        return self._cache_halos_

    @_cache_halos.setter
    def _cache_halos(self, value):
        self._cache_halos_ = value

    def _gen_halo_histories(self):
        """
        From a set of smooth halo assembly histories, build a bigger set
        of histories by thinning, and (optionally) adding scatter to the MAR.
        """

        if hasattr(self, '_cache_halos_'):
            return self._cache_halos

        raw = self.load()

        thin = self.pf['pop_thin_hist']

        if thin > 1:
            assert self.pf['pop_histories'] is None, \
                "Set pop_thin_hist=pop_scatter_mar=0 if supplying pop_histories by hand."

        sigma_mar = self.pf['pop_scatter_mar']
        sigma_env = self.pf['pop_scatter_env']

        # Just read in histories in this case.
        if raw is None:
            print('Running halo trajectories...')
            zall, raw = self.guide.Trajectories()
            print('Done with halo trajectories.')
        else:
            zall = raw['z']

        # Should be in ascending redshift.
        assert np.all(np.diff(zall) > 0)

        nh_raw = raw['nh']
        Mh_raw = raw['Mh']

        if type(nh_raw) in [int, float, np.float32, np.float64]:
            nh_raw = nh_raw * np.ones_like(Mh_raw)

        # May have to generate MAR if these are simulated halos
        if ('MAR' not in raw) and ('MAR_acc' not in raw):

            assert thin < 2
            assert sigma_mar == sigma_env == 0

            if self.pf['pop_MAR_from_hist']:
                print("Generating MARs from Mh trajectories...")
                dM = -1. * np.diff(Mh_raw, axis=1)

                t = self.cosm.t_of_z(zall) * 1e6 / s_per_myr

                dt = -1. * np.diff(t) # in yr already

                MAR_z = dM / dt

                zeros = np.ones((Mh_raw.shape[0], 1)) * tiny_MAR
                # Follow ARES convention of forward differencing, so must pad MAR
                # array with zeros at the lowest redshift snapshot.
                mar_raw = np.hstack((zeros, MAR_z))

            else:
                print("Generating MARs from `guide` population...")
                z2d = zall[None,:]
                mar_raw = np.zeros_like(Mh_raw)
                for i, z in enumerate(zall):
                    mar_raw[:,i] = self.guide.MAR(z=z, Mh=Mh_raw[:,i])

        else:
            if 'MAR' in raw:
                mar_raw = raw['MAR']
            elif self.pf['pop_mergers']:
                mar_raw = raw['MAR_acc']
            else:
                if 'MAR_tot' in raw:
                    mar_raw = raw['MAR_tot']
                else:
                    mar_raw = raw['MAR_acc']

        ##
        # Throw away halos with Mh < Mmin or Mh > Mmax
        ##
        if self.pf['pop_synth_minimal'] and (self.pf['pop_histories'] is None):

            Mmin = self.guide.Mmin(zall)

            # Find boundary between halos that never cross Mmin and those
            # that do.
            is_viable = Mh_raw > Mmin[None,:]

            any_viable = np.sum(is_viable, axis=1)

            # Cut out halos that never exist in our mass range of interest.
            ilo = np.min(np.argwhere(any_viable > 0))
            ihi = np.max(np.argwhere(any_viable > 0)) + 1

            # Also cut out some redshift range.
            zok = np.logical_and(zall >= self.pf['pop_synth_zmin'],
                zall <= self.pf['pop_synth_zmax'])
            zall = zall[zok==1]

            # Modify our arrays
            Mh_raw = Mh_raw[ilo:ihi,zok==1]
            nh_raw = nh_raw[ilo:ihi,zok==1]
            mar_raw = mar_raw[ilo:ihi,zok==1]

        ##
        # Could optionally thin out the bins to allow more diversity.
        if thin > 0:
            # Doesn't really make sense to do this unless we're
            # adding some sort of stochastic effects.

            # Remember: first dimension is the SFH identity.
            nh = self.tile(nh_raw, thin, True)
            Mh = self.tile(Mh_raw, thin)
        else:
            nh = nh_raw#.copy()
            Mh = Mh_raw#.copy()

        self.tab_shape = Mh.shape

        ##
        # Allow scatter in things
        ##

        # Two potential kinds of scatter in MAR
        mar = self.tile(mar_raw, thin)
        if sigma_env > 0:
            mar *= (1. + self.get_noise_normal(mar, sigma_env))

        if sigma_mar > 0:
            np.random.seed(self.pf['pop_scatter_mar_seed'])
            noise = self.get_noise_lognormal(mar, sigma_mar)
            mar += noise
            # Normalize by mean of log-normal to preserve mean MAR?
            mar /= np.exp(0.5 * sigma_mar**2)
            del noise

        # SFR = (zform, time (but really redshift))
        # So, halo identity is wrapped up in axis=0
        # In Cohort, formation time defines initial mass and trajectory (in full)
        #z2d = np.array([zall] * nh.shape[0])

        # If loaded from merger tree, these quantities should be
        # numpy masked arrays.
        histories = {'Mh': Mh, 'MAR': mar, 'nh': nh}

        # Add in formation redshifts to match shape (useful after thinning)
        histories['zthin'] = self.tile(zall, thin)

        histories['z'] = zall

        if self.pf['conserve_memory']:
            dtype = np.float32
        else:
            dtype = np.float64

        t = np.array([self.cosm.t_of_z(zall[_i]) \
            for _i in range(zall.size)]) / s_per_myr

        histories['t'] = t.astype(dtype)

        if self.pf['pop_dust_yield'] is not None:
            r = np.reshape(np.random.rand(Mh.size), Mh.shape)
            if self.pf['conserve_memory']:
                histories['rand'] = r.astype(np.float32)
            else:
                histories['rand'] = r
        else:
            pass

        if 'SFR' in raw:
            #assert sigma_mar == sigma_env == 0
            histories['SFR'] = raw['SFR']

        if 'Z' in raw:
            histories['Z'] = raw['Z']

        if 'children' in raw:
            histories['children'] = raw['children']

        if 'pos' in raw:
            histories['pos'] = raw['pos']

        if 'flags' in raw:
            histories['flags'] = raw['flags']

        self.tab_z = zall
        #self._cache_halos = histories

        del raw
        gc.collect()

        return histories

    @property
    def histories(self):
        if not hasattr(self, '_histories'):
            self._histories = self.RunSAM()
        return self._histories

    @histories.setter
    def histories(self, value):

        assert type(value) is dict

        must_flip = False
        if 'z' in value:
            if np.all(np.diff(value['z']) > 0):
                must_flip = True

        if must_flip:
            for key in value:
                if not type(value[key]) == np.ndarray:
                    continue

                if value[key].ndim == 1:
                    value[key] = value[key][-1::-1]
                else:
                    value[key] = value[key][:,-1::-1]

        self._histories = value

    def Trajectories(self):
        return self.RunSAM()

    def RunSAM(self):
        """
        Run models. If deterministic, will just return pre-determined
        histories. Otherwise, will do some time integration.
        """

        if self.pf['pop_sam_method'] == 0:
            return self._gen_prescribed_galaxy_histories()
        elif self.pf['pop_sam_method'] == 1:
            return self._gen_active_galaxy_histories()
        else:
            raise NotImplemented('Unrecognized      pop_sam_method={}.'.format(self.pf['pop_sam_method']))

    @property
    def guide(self):
        if not hasattr(self, '_guide'):
            if self.pf['pop_guide_pop'] is not None:
                self._guide = self.pf['pop_guide_pop']
            else:
                tmp = self.pf.copy()
                tmp['pop_ssp'] = False
                self._guide = GalaxyCohort(**tmp)

        return self._guide

    #def cmf(self, M):
    #    # Allow ParameterizedQuantity here
    #    pass
#
    #@property
    #def tab_cmf(self):
    #    if not hasattr(self, '_tab_cmf'):
    #        pass
#
    #@property
    #def _norm(self):
    #    if not hasattr(self, '_norm_'):
    #        mf = lambda logM: self.ClusterMF(10**logM)
    #        self._norm_ = quad(lambda logM: mf(logM) * 10**logM, -3, 10.,
    #            limit=500)[0]
    #    return self._norm_
#
    #def ClusterMF(self, M, beta=-2, Mmin=50.):
    #    return (M / Mmin)**beta * np.exp(-Mmin / M)
#
    #@property
    #def tab_Mcl(self):
    #    if not hasattr(self, '_tab_Mcl'):
    #        self._tab_Mcl = np.logspace(-1., 8, 10000)
    #    return self._tab_Mcl
#
    #@tab_Mcl.setter
    #def tab_Mcl(self, value):
    #    self._tab_Mcl = value
#
    #@property
    #def tab_cdf(self):
    #    if not hasattr(self, '_tab_cdf'):
    #        mf = lambda logM: self.ClusterMF(10**logM)
    #        f_cdf = lambda M: quad(lambda logM: mf(logM) * 10**logM, -3, np.log10(M),
    #            limit=500)[0] / self._norm
    #        self._tab_cdf = np.array(map(f_cdf, self.tab_Mcl))
#
    #    return self._tab_cdf
#
    #@tab_cdf.setter
    #def tab_cdf(self, value):
    #    assert len(value) == len(self.tab_Mcl)
    #    self._tab_cdf = value
#
    #def ClusterCDF(self):
    #    if not hasattr(self, '_cdf_cl'):
    #        self._cdf_cl = lambda MM: np.interp(MM, self.tab_Mcl, self.tab_cdf)
#
    #    return self._cdf_cl
#
    #@property
    #def Mcl(self):
    #    if not hasattr(self, '_Mcl'):
    #        mf = lambda logM: self.ClusterMF(10**logM)
    #        self._Mcl = quad(lambda logM: mf(logM) * (10**logM)**2, -3, 10.,
    #            limit=500)[0] / self._norm
#
    #    return self._Mcl
#
    #@property
    #def tab_imf_me(self):
    #    if not hasattr(self, '_tab_imf_me'):
    #        self._tab_imf_me = 10**bin_c2e(self.src.pf['source_imf_bins'])
    #    return self._tab_imf_me
#
    #@property
    #def tab_imf_mc(self):
    #    if not hasattr(self, '_tab_imf_mc'):
    #        self._tab_imf_mc = 10**self.src.pf['source_imf_bins']
    #    return self._tab_imf_mc

    def _cache_ehat(self, key):
        if not hasattr(self, '_cache_ehat_'):
            self._cache_ehat_ = {}

        if key in self._cache_ehat_:
            return self._cache_ehat_[key]

        return None

    def _TabulateEmissivity(self, E=None, Emin=None, Emax=None, wave=None):
        """
        Compute emissivity over a grid of redshifts and setup interpolant.
        """

        dz = self.pf['pop_synth_dz']
        zarr = np.arange(self.pf['pop_synth_zmin'],
            self.pf['pop_synth_zmax'] + dz, dz)

        if (Emin is not None) and (Emax is not None):
            # Need to send off in Angstroms
            band = (1e8 * h_p * c / (Emax * erg_per_ev),
                    1e8 * h_p * c / (Emin * erg_per_ev))
        else:
            band = None

        if (band is not None) and (E is not None):
            raise ValueError("You're being confusing! Supply `E` OR `Emin` and `Emax`")

        if wave is not None:
            raise NotImplemented('careful')

        hist = self.histories

        tab = np.zeros_like(zarr)
        for i, z in enumerate(zarr):

            # This will be [erg/s]
            L = self.synth.Luminosity(sfh=hist['SFR'], zobs=z, band=band,
                zarr=hist['z'], extras=self.extras)

            # OK, we've got a whole population here.
            nh = self.get_field(z, 'nh')
            Mh = self.get_field(z, 'Mh')

            # Modify by fesc
            if band is not None:
                if Emin in [13.6, E_LL]:
                    # Doesn't matter what Emax is
                    fesc = self.guide.fesc(z=z, Mh=Mh)
                elif (Emin, Emax) in [(10.2, 13.6), (E_LyA, E_LL)]:
                    fesc = self.guide.fesc_LW(z=z, Mh=Mh)
                else:
                    fesc = 1.
            else:
                fesc = 1.

            # Integrate over halo population.
            tab[i] = np.sum(L * fesc * nh)

            if np.isnan(tab[i]):
                tab[i] = 0

        return zarr, tab / cm_per_mpc**3

    def get_emissivity(self, z, E=None, Emin=None, Emax=None):
        """
        Compute the emissivity of this population as a function of redshift
        and (potentially) rest-frame photon energy [eV].

        .. note :: If Emin and Emax are supplied, this is a luminosity density,
            and will have units of erg/s/(co-moving cm)^3. If `E` is supplied,
            will also carry units of eV^-1.

        Parameters
        ----------
        z : int, float
            Redshift.

        Returns
        -------
        Emissivity in units of erg / s / c-cm**3 [/ eV].

        """

        on = self.on(z)
        if not np.any(on):
            return z * on

        # Need to build an interpolation table first.
        # Cache also by E, Emin, Emax

        cached_result = self._cache_ehat((E, Emin, Emax))
        if cached_result is not None:
            func = cached_result
        else:

            zarr, tab = self._TabulateEmissivity(E, Emin, Emax)

            tab[np.logical_or(tab <= 0, np.isinf(tab))] = 1e-70

            func = interp1d(zarr, np.log10(tab), kind='cubic',
                bounds_error=False, fill_value=-np.inf)

            self._cache_ehat_[(E, Emin, Emax)] = func#zarr, tab

        return 10**func(z)
        #return self._cache_ehat_[(E, Emin, Emax)](z)

    def get_photon_density(self, z, E=None, Emin=None, Emax=None):
        # erg / s / cm**3
        rhoL = self.get_emissivity(z, E=E, Emin=Emin, Emax=Emax)
        erg_per_phot = self._get_energy_per_photon(Emin, Emax) * erg_per_ev

        return rhoL / np.mean(erg_per_phot)

    def _gen_stars(self, idnum, Mh): # pragma: no cover
        """
        Take draws from cluster mass function until stopping criterion met.

        Return the amount of mass formed in this burst.
        """

        raise NotImplemented('this will need a lot of fixing.')

        z = self._arr_z[idnum]
        t = self._arr_t[idnum]
        dt = (self._arr_t[idnum+1] - self._arr_t[idnum]) # in Myr

        E_h = self.halos.BindingEnergy(z, Mh)

        # Statistical approach from here on out.
        Ms = 0.0

        Mg = self._arr_Mg_c[idnum] * 1.

        # Number of supernovae from stars formed previously
        N_SN_p = self._arr_SN[idnum] * 1.
        E_UV_p = self._arr_UV[idnum] * 1.
        # Number of supernovae from stars formed in this timestep.
        N_SN_0 = 0.
        E_UV_0 = 0.

        N_MS_now = 0
        m_edg = self.tab_imf_me
        m_cen = self.tab_imf_mc
        imf = np.zeros_like(m_cen)

        fstar_gmc = self.pf['pop_fstar_cloud']

        delay_fb_sne = self.pf['pop_delay_sne_feedback']
        delay_fb_rad = self.pf['pop_delay_rad_feedback']

        vesc = self.halos.EscapeVelocity(z, Mh) # cm/s

        # Form clusters until we use all the gas or blow it all out.
        ct = 0
        Mw = 0.0
        Mw_rad = 0.0
        while (Mw + Mw_rad + Ms) < Mg * fstar_gmc:

            r = np.random.rand()
            Mc = np.interp(r, self.tab_cdf, self.tab_Mcl)

            # If proposed cluster would take up all the rest of our
            # gas (and then some), don't let it happen.
            if (Ms + Mc + Mw) >= Mg:
                break

            ##
            # First things first. Figure out the IMF in this cluster.
            ##

            # Means we're doing cheap spectral synthesis.
            if self.pf['pop_sample_imf']:
                # For now, just scale UV luminosity with Nsn?

                # Uniform IMF placeholder
                #r2 = 0.1 + np.random.rand(1000) * (200. - 0.1)
                r2 = self._stars.draw_stars(1000000)

                # Integrate until we get Mc.
                m2 = np.cumsum(r2)

                # What if, by chance, first star drawn is more massive than
                # Mc?

                cut = np.argmin(np.abs(m2 - Mc)) + 1

                if cut >= len(r2):
                    cut = len(r2) - 1
                    #print(r2.size, Mc, m2[-1] / Mc)
                    #raise ValueError('help')

                hist, edges = np.histogram(r2[0:cut+1], bins=m_edg)
                imf += hist

                N_MS = np.sum(hist[m_cen >= 8.])

                # Ages of the stars

                #print((Mw + Ms) / Mg, Nsn, Mc / 1e3)

            else:
                # Expected number of SNe if we were forming lots of clusters.
                lam = Mc * self._stars.nsn_per_m
                N_MS = np.random.poisson(lam)
                hist = 0.0

            ##
            # Now, take stars and let them feedback on gas.
            ##

            ##
            # Increment stuff
            ##
            Ms += Mc
            imf += hist

            # Move along.
            if N_MS == 0:
                ct += 1
                continue

            ##
            # Can delay feedback or inject it instantaneously.
            ##
            if delay_fb_sne == 0:
                N_SN_0 += N_MS
            elif delay_fb_sne == 1:
                ##
                # SNe all happen at average delay time from formation.
                ##

                if self.pf['pop_sample_imf']:
                    raise NotImplemented('help')

                # In Myr
                avg_delay = self._stars.avg_sn_delay

                # Inject right now if timestep is long.
                if dt > avg_delay:
                    N_SN_0 += N_MS
                else:
                    # Figure out when these guys will blow up.
                    #N_SN_next += N_MS

                    tnow = self._arr_t[idnum]
                    tfut = self._arr_t[idnum:]

                    iSNe = np.argmin(np.abs((tfut - tnow) - avg_delay))
                    #if self._arr_t[idnum+iSNe] < avg_delay:
                    #    iSNe += 1

                    print('hey 2', self._arr_t[idnum+iSNe] - tnow)

                    self._arr_SN[idnum+iSNe] += N_MS

            elif delay_fb_sne == 2:
                ##
                # Actually spread SNe out over time according to DTD.
                ##
                delays = self._stars.draw_delays(N_MS)

                tnow = self._arr_t[idnum]
                tfut = self._arr_t[idnum:]

                # Could be more precise since closest index may be
                # slightly shorter than delay time.
                iSNe = np.array([np.argmin(np.abs((tfut - tnow) - delay)) \
                    for delay in delays])

                # Add SNe one by one.
                for _iSNe in iSNe:
                    self._arr_SN[idnum+_iSNe] += 1

                # Must make some of the SNe happen NOW!
                N_SN_0 += sum(iSNe == 0)

            ##
            # Make a wind
            ##
            Mw = 2 * (N_SN_0 + N_SN_p) * 1e51 * self.pf['pop_coupling_sne'] \
               / vesc**2 / g_per_msun

            ##
            # Stabilize with some radiative feedback?
            ##
            if not self.pf['pop_feedback_rad']:
                ct += 1
                continue

            ##
            # Dump in UV from 'average' massive star.
            ##
            if self.pf['pop_delay_rad_feedback'] == 0:

                # Mask out low-mass stuff? Only because scaling by N_MS
                massive = self._stars.Ms >= 8.

                LUV = self._stars.tab_LUV

                Lavg = np.trapz(LUV[massive==1] * self._stars.tab_imf[massive==1],
                    x=self._stars.Ms[massive==1]) \
                     / np.trapz(self._stars.tab_imf[massive==1],
                    x=self._stars.Ms[massive==1])

                life = self._stars.tab_life
                tavg = np.trapz(life[massive==1] * self._stars.tab_imf[massive==1],
                    x=self._stars.Ms[massive==1]) \
                     / np.trapz(self._stars.tab_imf[massive==1],
                    x=self._stars.Ms[massive==1])

                corr = np.minimum(tavg / dt, 1.)

                E_UV_0 += Lavg * N_MS * dt * corr * 1e6 * s_per_yr

                #print(z, 2 * E_rad / vesc**2 / g_per_msun / Mw)

                Mw_rad += 2 * (E_UV_0 + E_UV_p) * self.pf['pop_coupling_rad'] \
                    / vesc**2 / g_per_msun

            elif self.pf['pop_delay_rad_feedback'] >= 1:
                raise NotImplemented('help')

                delays = self._stars.draw_delays(N_MS)

                tnow = self._arr_t[idnum]
                tfut = self._arr_t[idnum:]

                # Could be more precise since closest index may be
                # slightly shorter than delay time.
                iSNe = np.array([np.argmin(np.abs((tfut - tnow) - delay)) \
                    for delay in delays])

                # Add SNe one by one.
                for _iSNe in iSNe:
                    self._arr_SN[idnum+_iSNe] += 1

            # Count clusters
            ct += 1

        ##
        # Back outside loop over clusters.
        ##

        return Ms, Mw, imf

    def deposit_in(self, tnow, delay):
        """
        Determin index of time-array in which to deposit some gas, energy,
        etc., given current time and requested delay.
        """

        inow = np.argmin(np.abs(tnow - self._arr_t))

        tfut = self._arr_t[inow:]

        ifut = np.argmin(np.abs((tfut - tnow) - delay))

        return inow + ifut

    def get_fstar(self, z, Mh):
        return self.guide.get_fstar(z=z, Mh=Mh)

    def _gen_prescribed_galaxy_histories(self, zstop=0):
        """
        Take halo histories and paint on galaxy histories in deterministic way.
        """

        # First, grab halos
        halos = self._gen_halo_histories()

        ##
        # Simpler models. No need to loop over all objects individually.
        ##

        # Eventually generalize
        assert self.pf['pop_update_dt'].startswith('native')
        native_sampling = True

        Nhalos = halos['Mh'].shape[0]

        # Flip arrays to be in ascending time.
        z = halos['z'][-1::-1]
        z2d = z[None,:]
        t = halos['t'][-1::-1]
        Mh = halos['Mh'][:,-1::-1]
        nh = halos['nh'][:,-1::-1]

        # Will have been corrected for mergers in `load` if pop_mergers==1
        MAR = halos['MAR'][:,-1::-1]

        # 't' is in Myr, convert to yr
        dt = np.abs(np.diff(t)) * 1e6
        dt_myr = dt / 1e6

        ##
        # OK. We've got a bunch of halo histories and we need to integrate them
        # to get things like stellar mass, metal mass, etc. This means we need
        # to make an assumption about how halos grow between our grid points.
        # If we assume smooth histories, we should be trying to do a trapezoidal
        # integration in log-space. However, given that we often add noise to
        # MARs, in that case we should probably just assume flat MARs within
        # the timestep.
        #
        # Note also that we'll need to zero-pad these arrays to keep their
        # shapes the same after we integrate (just so we don't have indexing
        # problems later). Since we're doing a simple sum, we'll fill the
        # elements corresponding to the lowest redshift bin with zeros since we
        # can't compute luminosities after that point (no next grid pt to diff
        # with).
        ##

        fb = self.cosm.fbar_over_fcdm
        fZy = self.pf['pop_mass_yield'] * self.pf['pop_metal_yield']

        if self.pf['pop_dust_yield'] is not None:
            fd = self.guide.dust_yield(z=z2d, Mh=Mh)
            have_dust = np.any(fd > 0)
        else:
            fd = 0.0
            have_dust = False

        if self.pf['pop_dust_growth'] is not None:
            fg = self.guide.dust_growth(z=z2d, Mh=Mh)
        else:
            fg = 0.0

        fmr = self.pf['pop_mass_yield']
        fml = (1. - fmr)

        # Integrate (crudely) mass accretion rates
        #_Mint = cumtrapz(_MAR[:,:], dx=dt, axis=1)
        #_MAR_c = 0.5 * (np.roll(MAR, -1, axis=1) + MAR)
        #_Mint = np.cumsum(_MAR_c[:,1:] * dt, axis=1)

        if 'SFR' in halos:
            SFR = halos['SFR'][:,-1::-1]
        else:
            iz = np.argmin(np.abs(6. - z))
            SFR = self.guide.get_fstar(z=z2d, Mh=Mh)
            np.multiply(SFR, MAR, out=SFR)
            SFR *= fb

        ##
        # Duty cycle effects
        ##
        if self.pf['pop_fduty'] is not None:

            np.random.seed(self.pf['pop_fduty_seed'])

            fduty = self.guide.fduty(z=z2d, Mh=Mh)
            T_on = self.pf['pop_fduty_dt']

            if T_on is not None:

                fduty_avg = np.mean(fduty, axis=1)

                # Create random bursts with length `T_on`
                dt_tot = t.max() - t.min() # Myr

                i_on = int(T_on / dt_myr[0])

                # t is in ascending order
                Nt = float(t.size)
                on = np.zeros_like(Mh, dtype=bool)

                for i in range(Nhalos):

                    if np.all(SFR[i] == 0):
                        continue

                    r = np.random.rand(t.size)
                    on_prop = r < fduty[i]

                    if fduty_avg[i] == 1:
                        on[i,:] = True
                    else:
                        ct = 0
                        while (on[i].sum() / Nt) < fduty_avg[i]:
                            j = np.random.randint(low=0, high=int(Nt))

                            on[i,j:j+i_on] = True

                            ct += 1

                off = np.logical_not(on)

            else:
                # Random numbers for all mass and redshift points
                r = np.reshape(np.random.rand(Mh.size), Mh.shape)

                off = r >= fduty

            SFR[off==True] = 0

        # Never do this!
        if self.pf['conserve_memory']:
            raise NotImplemented('this is deprecated')
            dtype = np.float32
        else:
            dtype = np.float64

        zeros_like_Mh = np.zeros((Nhalos, 1), dtype=dtype)

        ##
        # Cut out Mh < Mmin galaxies
        if self.pf['pop_Mmin'] is not None:
            above_Mmin = Mh >= self.pf['pop_Mmin']
        else:
            Mmin = self.halos.VirialMass(z2d, self.pf['pop_Tmin'])
            above_Mmin = Mh >= Mmin

        # Bye bye guys
        SFR[above_Mmin==False] = 0

        ##
        # Introduce some by-hand quenching.
        if self.pf['pop_quench'] is not None:
            zreion = self.pf['pop_quench']
            if type(zreion) in [np.ndarray, np.ma.core.MaskedArray]:
                assert zreion.size == Mh.size, \
                    "Supplied `is_quenched` mask is the wrong size!"

                is_quenched = self.pf['pop_quench']

                assert np.unique(is_quenched).size == 2, \
                    "`pop_quench` should be a mask of ones and zeros!"
            else:
                # In this case, the supplied function isn't zreion, it tells
                # us whether halos are quenched or not.
                is_quenched = zreion(z=z2d, Mh=Mh)

            # Print some quenched fraction vs. redshift to help debug?
            if self.pf['debug']:

                for _z_ in [10, 8, 6, 5, 4]:

                    k = np.argmin(np.abs(z - _z_))
                    print('# Quenched fraction at z={}: {:.4f}'.format(z[k],
                        np.sum(is_quenched[:,k]) / float(Nhalos)))
                    print('# Quenched number at z={}: {}'.format(z[k],
                        np.sum(is_quenched[:,k])))

                    Mh_active = Mh[~is_quenched[:,k],k]
                    print("# Least massive active halos at z={}: {:.2e}".format(
                        z[k], Mh_active[Mh_active>0].min()))
                    print("# Most massive active halos at z={}: {:.2e}".format(
                        z[k], Mh_active[Mh_active>0].max()))
                    print("# Most massive halo (period) at z={}: {:.2e}".format(
                        z[k], Mh[:,k].max()))

                    if self.pf['pop_Mmin'] == 0:
                        continue

                    hfrac = Mh[:,k] < self.pf['pop_Mmin']
                    print('# Halo fraction below Mmin = {}'.format(hfrac.sum() / float(hfrac.size)))

            # Bye bye guys
            SFR[is_quenched==True] = 0

            # Sanity check.
            SFR_eq0 = SFR == 0.0
            if SFR[SFR == 0].size < is_quenched.sum():
                err = "SFR should be == 0 for all quenched galaxies."
                err += " Only {}/{} SFR elements are zero, ".format(SFR_eq0.sum(),
                    SFR.size)
                err += "despite {} quenched elements".format(is_quenched.sum())

                # See if this is just a masking thing.
                ok = SFR.mask == 0
                if sum(SFR[ok==1] == 0) < sum(is_quenched[ok==1]):
                    raise ValueError(err)


        # Stellar mass should have zeros padded at the 0th time index
        Ms = np.hstack((zeros_like_Mh,
            np.cumsum(SFR[:,0:-1] * dt * fml, axis=1)))

        #Ms = np.zeros_like(Mh)

        if self.pf['pop_flag_sSFR'] is not None:
            sSFR = SFR / Ms

        ##
        # Dust
        ##
        if have_dust:

            delay = self.pf['pop_dust_yield_delay']

            if np.all(fg == 0):
                if type(fd) in [int, float, np.float64] and delay == 0:
                    Md = fd * fZy * Ms
                else:

                    if delay > 0:

                        assert np.allclose(np.diff(dt_myr), 0.0,
                            rtol=1e-5, atol=1e-5)

                        shift = int(delay // dt_myr[0])

                        # Need to fix so Mh-dep fd can still work.
                        assert type(fd) in [int, float, np.float64]

                        DPR = np.roll(SFR, shift, axis=1)[:,0:-1] \
                            * dt * fZy * fd
                        DPR[:,0:shift] = 0.0
                    else:
                        DPR = SFR[:,0:-1] * dt * fZy * fd[:,0:-1]

                    Md = np.hstack((zeros_like_Mh,
                        np.cumsum(DPR, axis=1)))
            else:

                # Handle case where growth in ISM is included.
                if type(fg) in [int, float, np.float64]:
                    fg = fg * np.ones_like(SFR)
                if type(fd) in [int, float, np.float64]:
                    fd = fd * np.ones_like(SFR)

                # fg^-1 is like a rate coefficient [has units yr^-1]
                Md = np.zeros_like(SFR)
                for k, _t in enumerate(t[0:-1]):

                    # Dust production rate
                    Md_p = SFR[:,k] * fZy * fd[:,k]

                    # Dust growth rate
                    Md_g = Md[:,k] / fg[:,k]

                    Md[:,k+1] = Md[:,k] + (Md_p + Md_g) * dt[k]

            # Dust surface density.
            Rd = self.guide.dust_scale(z=z2d, Mh=Mh)
            Sd = np.divide(Md, np.power(Rd, 2.)) \
                / 4. / np.pi

            iz = np.argmin(np.abs(6. - z))

            # Can add scatter to surface density
            if self.pf['pop_dust_scatter'] is not None:
                sigma = self.guide.dust_scatter(z=z2d, Mh=Mh)
                noise = np.zeros_like(Sd)
                np.random.seed(self.pf['pop_dust_scatter_seed'])
                for _i, _z in enumerate(z):
                    noise[:,_i] = self.get_noise_lognormal(Sd[:,_i], sigma[:,_i])

                Sd += noise

            # Convert to cgs. Do in two steps in case conserve_memory==True.
            Sd *= g_per_msun / cm_per_kpc**2

            if self.pf['pop_dust_fcov'] is not None:
                fcov = self.guide.dust_fcov(z=z2d, Mh=Mh)
            else:
                fcov = 1.

        else:
            Md = Sd = 0.
            Rd = np.inf
            fcov = 1.0

        # Metal mass
        if 'Z' in halos:
            Z = halos['Z']
            Mg = MZ = 0.0
        else:
            if self.pf['pop_enrichment']:
                MZ = Ms * fZy

                # Gas mass
                Mg = np.hstack((zeros_like_Mh,
                    np.cumsum((MAR[:,0:-1] * fb - SFR[:,0:-1]) * dt, axis=1)))

                if self.pf['pop_enrichment'] == 2:
                    Vd = 4. * np.pi * self.guide.dust_scale(z=z2d, Mh=Mh)**3 / 3.
                    rho_Z = MZ / Vd
                    Vg = 4. * np.pi * self.halos.VirialRadius(z2d, Mh)**3 / 3.
                    rho_g = Mg / Vg
                    Z = rho_Z / rho_g / self.pf['pop_fpoll']
                else:
                    Z = MZ / Mg / self.pf['pop_fpoll']

                Z[Mg==0] = 1e-3
                Z = np.maximum(Z, 1e-3)

            else:
                MZ = Mg = Z = 0.0

        ##
        # Merge halos, sum stellar masses, SFRs.
        # Only add masses after progenitors are absorbed.
        # Need to add luminosity from progenitor history even after merger.
        # NOTE: no transferrance of gas, metals, or stars, as of yet.
        ##
        if self.pf['pop_mergers'] > 0: # pragma: no cover
            children = halos['children']
            iz, iM, is_main = children.T
            uni = np.all(Mh.mask == False, axis=1)
            merged = np.logical_and(iz >= 0, uni == True)

            pos = halos['pos'][:,-1::-1,:]

            for i in range(iz.size):
                if iz[i] < 0:
                    continue

                # May not be necessary
                if not merged[i]:
                    continue

                # Fill-in positions of merged parents
                #pos[i,0:iz[i],:] = pos[iM[i],iz[i],:]

                # Add SFR so luminosities include that of parent halos
                #SFR[iM[i],:] += SFR[i,:]
                # Looks like a potential double-counting issue but SFR
                # will have been set to zero post-merger.

                # Add stellar mass of progenitors
                Ms[iM[i],iz[i]:] += np.max(Ms[i,:])

                #Mg[iM[i],iz[i]:] += max(Mg[i,:])

                # Add dust mass of progenitors
                if have_dust:
                    Md[iM[i],iz[i]:] += np.max(Md[i,:])
                    #MZ[iM[i],iz[i]:] += max(MZ[i,:])

            # Re-compute dust surface density
            if have_dust and (self.pf['pop_dust_scatter'] is not None):
                Sd = Md / 4. / np.pi \
                    / self.guide.dust_scale(z=z2d, Mh=Mh)**2
                Sd += noise
                Sd *= g_per_msun / cm_per_kpc**2

        # Limit to main branch
        elif self.pf['pop_mergers'] == -1: # pragma: no cover
            children = halos['children'][:,-1::-1]
            iz, iM, is_main = children.T
            main_branch = is_main == 1

            nh = nh[main_branch==1]
            Ms = Ms[main_branch==1]
            Mh = Mh[main_branch==1]
            #MAR = MAR[main_branch==1]
            #MZ = MZ[main_branch==1]
            Md = Md[main_branch==1]
            Sd = Sd[main_branch==1]
            #fcov = fcov[main_branch==1]
            #Mg = Mg[main_branch==1]
            #Z = Z[main_branch==1]
            SFR = SFR[main_branch==1]
            zeros_like_Mh = zeros_like_Mh[main_branch==1]

            if 'pos' in halos:
                pos = halos['pos'][main_branch==1,-1::-1,:]
            else:
                pos = None
        else:
            children = None

            if 'pos' in halos:
                pos = halos['pos'][:,-1::-1,:]
            else:
                pos = None

        del z2d

        # Pack up
        results = \
        {
         'nh': nh,
         'Mh': Mh,
         'MAR': MAR,  # May use this
         't': t,
         'z': z,
         'children': children,
         'zthin': halos['zthin'][-1::-1],
         #'z2d': z2d,
         'SFR': SFR,
         'Ms': Ms,
         'MZ': MZ,
         'Md': Md,
         'Sd': Sd,
         'fcov': fcov,
         'Mh': Mh,
         'Mg': Mg,
         'Z': Z,
         'bursty': zeros_like_Mh,
         'pos': pos,
         #'imf': np.zeros((Mh.shape[0], self.tab_imf_mc.size)),
         'Nsn': zeros_like_Mh,
        }

        if 'flags' in halos.keys():
            results['flags'] = halos['flags'][:,-1::-1]

        if self.pf['pop_dust_yield'] is not None:
            results['rand'] = halos['rand'][:,-1::-1]

        # Reset attribute!
        self.histories = results

        return results

    def _gen_active_galaxy_histories(self):
        """
        This is called when pop_sam_method == 1.

        The main difference between this model and pop_sam_method == 0 is that
        we try to 'do feedback properly', i.e., rather than prescribing the
        SFE as a function of redshift and mass, we dump energy/momentum in,
        with possible delay, and compare to binding energy etc. on a timestep
        by timestep basis. Should recover minimalist model when delay is zero.

        """
        # First, grab halos
        halos = self._gen_halo_histories()

        # Eventually generalize
        assert self.pf['pop_update_dt'].startswith('native')
        native_sampling = True

        ##
        # Grab some basic info about the halos
        Nhalos = halos['Mh'].shape[0]
        zeros_like_Mh = np.zeros((Nhalos, 1))

        # Flip arrays to be in ascending time.
        z = halos['z'][-1::-1]
        z2d = z[None,:]
        t = halos['t'][-1::-1]
        Mh = halos['Mh'][:,-1::-1]
        nh = halos['nh'][:,-1::-1]

        # Will have been corrected for mergers in `load` if pop_mergers==1
        MAR = np.maximum(halos['MAR'][:,-1::-1], 0)

        t_ind = np.arange(0, t.size, 1)

        # 't' is in Myr, convert to yr
        dt = np.abs(np.diff(t)) * 1e6
        dt_myr = dt / 1e6

        fb = self.cosm.fbar_over_fcdm
        fmr = self.pf['pop_mass_yield']
        fml = (1. - fmr)
        fZy = fmr * self.pf['pop_metal_yield']

        #######################################################################

        #
        # Time to actually run the SAM.

        # Loop over time.
        # Determine gas mass in SF phase
        # Determine energy injection from SNe.
        # Unbind some gas.
        # Continue

        dt_over = self.pf['pop_sfh_oversample']
        if dt_over > 0:
            raise NotImplemented('help')


        Mg = np.zeros((Nhalos, len(t)))
        SFR = np.zeros((Nhalos, len(t)))
        SNR = np.zeros((Nhalos, len(t)))
        EIR = np.zeros((Nhalos, len(t)))
        Ms = np.zeros((Nhalos, len(t)))
        MZ = np.zeros((Nhalos, len(t)))

        vesc = self.halos.EscapeVelocity(z2d, Mh) # in cm/s
        # Some Mh = 0, need to prevent NaNs from muddying the waters.
        vesc[Mh==0] = np.inf

        if type(self.pf['pop_fshock']) is str:
            fshock = self.guide.fshock(z=z2d, Mh=Mh)
        else:
            fshock = np.ones_like(z2d) * self.pf['pop_fshock']

        fstar = self.pf['pop_fstar']
        fstar_max = self.pf['pop_fstar_max']
        fb_delay = self.pf['pop_delay_sne_feedback']

        # Hard-coding some stuff for now
        A_r = 0.02

        for i, _t_ in enumerate(t[0:-1]):

            # At the start, assume we've got a cosmic baryon fraction's worth
            # of gas.
            if i == 0:
                Mg[:,i] = fb * Mh[:,i]

            # Myr -> yr
            # Use halo properties from sims to compute this?
            torb = 18. * (A_r / 0.02) * (7. / (1. + z[i]))**-1.5 * 1e6

            SFR[:,i] = Mg[:,i] * np.minimum(fstar / torb, fstar_max)

            # Cap SFR?
            #SFR[:,i] = np.minimum(SFR[:,i], )


            # Increment stellar mass
            Ms[:,i] += SFR[:,i] * dt[i]

            # Set when supernovae happen.


            if self.pf['pop_delay_sne_feedback']:

                if type(fb_delay) in num_types:

                    j = np.argmin(np.abs(t[i] + fb_delay - t))

                    EIR[:,j] = SFR[:,i] * self.pf['pop_omega_51'] * 1e51

                else:

                    raise NotImplemented('help')



            else:
                # Energy injection rate from SNe.
                # This is like assuming 10% of stars blow up and their
                # typical mass is 10 Msun.
                EIR[:,i] = SFR[:,i] * self.pf['pop_omega_51'] * 1e51

            # Mass ejection rates
            Mg_ej = self.pf['pop_coupling_sne'] * EIR[:,i] * dt[i] \
                / vesc[:,i]**2 / g_per_msun


            ##
            # Updates for next timestep below.
            ##

            # Gas in halo (less new stars) at the end of this timestep
            Mg_in = Mg[:,i] + fb * MAR[:,i] * fshock[:,i] * dt[i] \
                  - SFR[:,i] * dt[i]

            # Can't eject more gas than we've got.
            Mg_ej = np.minimum(Mg_in, Mg_ej)

            # Set gas mass for next time-step.
            Mg[:,i+1] = Mg_in - Mg_ej


        # Pack up
        results = \
        {
         'nh': nh,
         'Mh': Mh,
         'MAR': MAR,  # May use this
         't': t,
         'z': z,
         #'child': child,
         'zthin': halos['zthin'][-1::-1],
         #'z2d': z2d,
         'SFR': SFR,
         'Ms': Ms,
         #'MZ': MZ,
         #'Md': Md,
         #'Sd': Sd,
         #'fcov': fcov,
         'Mh': Mh,
         'Mg': Mg,
         #'Z': Z,
         #'bursty': zeros_like_Mh,
         #'pos': pos,
         #'imf': np.zeros((Mh.shape[0], self.tab_imf_mc.size)),
         'Nsn': zeros_like_Mh,
        }

        return results

    def get_field(self, z, field):
        iz = np.argmin(np.abs(z - self.histories['z']))
        return self.histories[field][:,iz]

    def StellarMassFunction(self, z, bins=None, units='dex'):
        return self.get_smf(z, bins=bins, units=units)

    def get_smf(self, z, bins=None, units='dex', Mbin=0.1):
        """
        Could do a cumulative sum to get all stellar masses in one pass.

        For now, just do for the redshift requested.

        Parameters
        ----------
        z : int, float
            Redshift of interest.
        bins : array, float
            log10 stellar masses at which to evaluate SMF. Bin *centers*.
        Mbin : float
            Alternatively, just provide log10 bin width.

        Returns
        -------
        Tuple containing (stellar mass bins in log10(Msun), number densities).

        """

        cached_result = self._cache_smf(z, bins)
        if cached_result is not None:
            return cached_result

        iz = np.argmin(np.abs(z - self.histories['z']))
        Ms = self.histories['Ms'][:,iz]
        nh = self.histories['nh'][:,iz]
        ok = Ms > 0

        if (bins is None) or (type(bins) is not np.ndarray):
            binw = Mbin
            bin_c = np.arange(0., 13.+binw, binw)
        else:
            dx = np.diff(bins)
            assert np.allclose(np.diff(dx), 0)
            binw = dx[0]
            bin_c = bins

        bin_e = bin_c2e(bin_c)

        # Make sure binning range covers the range of SFRs
        assert np.log10(Ms[ok==1]).min() > bin_e.min(), \
            "Bins do not span full range in stellar mass!"
        assert np.log10(Ms[ok==1]).max() < bin_e.max(), \
            "Bins do not span full range in stellar mass!"

        phi, _bins = np.histogram(Ms, bins=10**bin_e, weights=nh)

        if units == 'dex':
            # Convert to dex**-1 units
            phi /= binw
        else:
            raise NotImplemented('help')

        self._cache_smf_[z] = bin_c, phi

        return self._cache_smf(z, bin_c)

    def get_xhm(self, z, field='Ms', bins=None, return_mean_only=False,
        Mbin=0.1, method_avg='median'):
        """
        Generic routine for retrieving the X -- halo-mass relation, where
        X is some phase of galaxies in our model, e.g., gas mass, metal mass,
        SFR, etc.

        Parameters
        ----------
        z : int, float
            Redshift.
        field: str
            String describing field X in XMHM relation, e.g., stellar mass
            is 'Ms', gas mass is 'Mg'. See contents of `histories` attribute
            for more ideas of what's available.
        bins : np.ndarray
            Optional: if provided, array of log10(halo mass) bins to use in
            determining the relation. Must be evenly spaced in log10.
        return_mean_only : bool
            By default (False), will return bins, the X--halo-mass fractions,
            and scatter in each bin. If True, will just return X/HM fractions.
        Mbin : float
            If Mh=None (default), will construct array of halo mass bins using
            this log10 spacing.

        Returns
        -------
        X / Mh fraction; also see `return_mean_only` above.

        """
        iz = np.argmin(np.abs(z - self.histories['z']))

        _X = self.histories[field][:,iz]
        _Mh = self.histories['Mh'][:,iz]

        Xfrac = _X / _Mh

        if (bins is None) or (type(bins) is not np.ndarray):
            bin_c = np.arange(6., 14.+Mbin, Mbin)
        else:
            dx = np.diff(bins)
            assert np.allclose(np.diff(dx), 0)
            Mbin = dx[0]
            bin_c = bins

        nh = self.get_field(z, 'nh')
        x, y, z, N = quantify_scatter(np.log10(_Mh), np.log10(Xfrac), bin_c,
            weights=nh, method_avg=method_avg)

        if return_mean_only:
            return y

        return x, y, z

    def get_smhm(self, z, bins=None, return_mean_only=False, Mbin=0.1):
        """
        Compute stellar mass -- halo mass relation at given redshift `z`.

        .. note :: Just a wrapper around `get_xmhm`; see above.

        Parameters
        ----------
        z : int, float
            Redshift of interest
        bins : int, np.ndarray
            Halo mass bins (their centers) in to use for histogram.
            Must be evenly spaced in log10. Optional -- will use `Mbin` to
            create if Mh not supplied (default).

        """

        return self.get_xhm(z, field='Ms', bins=bins,
            return_mean_only=return_mean_only, Mbin=Mbin)

    def get_sfr_df(self, z, bins=None, return_mean_only=False, sfrbin=0.1):
        """
        Compute SFR distribution function at given redshift `z`.

        .. note :: This is like a UVLF just without converting SFRs to
            luminosities and without doing any dust corrections.

        Parameters
        ----------
        z : int, float
            Redshift of interest
        bins : int, np.ndarray
            SFR bins (their centers) to use for histogram.
            Must be evenly spaced in log10. Optional -- will use `sfrbin` to
            create if Mh not supplied (default).

        """

        sfr = self.get_field(z, 'SFR')
        nh = self.get_field(z, 'nh')

        ok = sfr > 0

        if bins is None:
            bins = np.arange(-8, 6+sfrbin, sfrbin)
        else:
            sfrbin = np.diff(bins)
            assert np.allclose(np.diff(sfrbin), 0)
            sfrbin = sfrbin[0]

        # Make sure binning range covers the range of SFRs
        assert np.log10(sfr[ok==1]).min() > bins.min(), \
            "Bins do not span full range in SFR!"
        assert np.log10(sfr[ok==1]).max() < bins.max(), \
            "Bins do not span full range in SFR!"

        hist, bin_histedges = np.histogram(np.log10(sfr[ok==1]),
            weights=nh[ok==1], bins=bin_c2e(bins), density=True)

        N = np.sum(nh[ok==1])
        phi = hist * N

        return bins, phi

    def get_main_sequence(self, z, bins=None, Mbin=0.1, method_avg='median'):
        """
        Compute the star-forming main sequence, i.e., stellar mass v. SFR.

        Parameters
        ----------
        z : int, float
            Redshift.
        bins : np.ndarray or None
            If provided, must be array of log10(stellar mass) bin *centers*.

        """

        Ms = self.get_field(z, 'Ms')
        sfr = self.get_field(z, 'SFR')

        if bins is None:
            bins = np.arange(6., 14.+Mbin, Mbin)
        else:
            dx = np.diff(bins)
            assert np.allclose(np.diff(dx), 0)
            Mbin = dx[0]

        nh = self.get_field(z, 'nh')
        x, y, z, N = quantify_scatter(np.log10(Ms), np.log10(sfr), bins,
            weights=nh, method_avg=method_avg)

        return x, y, z

    def get_uvsm(self, z, bins=None, magbin=None, method_avg='median'):
        """
        Get relationship between UV magnitude and stellar mass.

        z : int, float
            Redshift of interest
        bins : int, np.ndarray
            MUV bins (their centers) to use for histogram. Assumes absolute
            AB magnitude corresponding to rest-frame 1600 Angstrom. If None,
            will use parameters `pop_mag_min`, `pop_mag_max`, and potentially
            `pop_mag_bin` (see below).
        magbin : int, float
            Can instead provide bin size. If None, will revert to value of
            parameter `pop_mag_bin` in self.pf.



        """

        filt, MUV = self.get_mags(z, wave=1600.)
        Mst = self.get_field(z, 'Ms')

        if bins is None:
            magbin = magbin if magbin is not None else self.pf['pop_mag_bin']
            bins = np.arange(self.pf['pop_mag_min'],
                self.pf['pop_mag_max']+magbin, magbin)

        else:
            dx = np.diff(bins)
            assert np.allclose(np.diff(dx), 0)
            magbin = dx[0]

        nh = self.get_field(z, 'nh')
        ok = Mst > 0

        x, y, z, N = quantify_scatter(MUV[ok==1], np.log10(Mst[ok==1]), bins,
            weights=nh[ok==1], method_avg=method_avg)

        return x, y, z

    @property
    def _stars(self):
        if not hasattr(self, '_stars_'):
            self._stars_ = SynthesisModelSBS(**self.src_kwargs)
        return self._stars_

    def _cache_L(self, key):
        if not hasattr(self, '_cache_L_'):
            self._cache_L_ = {}

        if key in self._cache_L_:
            return self._cache_L_[key]

        return None

    def _cache_lf(self, z, bins=None, wave=None):
        if not hasattr(self, '_cache_lf_'):
            self._cache_lf_ = {}

        if (z, wave) in self._cache_lf_:

            _x, _phi = self._cache_lf_[(z, wave)]

            if self.pf['debug']:
                print("# Read LF from cache at (z={}, wave={})".format(
                    z, wave
                ))

            # If no x supplied, return bin centers
            if bins is None:
                return _x, _phi

            assert type(bins) == np.ndarray, "Must supply LF bins as array!"

            if _x.size == bins.size:
                if np.allclose(_x, bins):
                    return bins, _phi

            if self.pf['debug']:
                print("# Will interpolate to new MUV array.")

            #_func_ = interp1d(_x, np.log10(_phi), kind='cubic',
            #    fill_value=-np.inf)
            #
            #return 10**_func_(x)

            return bins, 10**np.interp(bins, _x, np.log10(_phi),
                left=-np.inf, right=-np.inf)

        return None

    def _cache_smf(self, z, Ms):
        if not hasattr(self, '_cache_smf_'):
            self._cache_smf_ = {}

        if z in self._cache_smf_:
            _x, _phi = self._cache_smf_[z]

            if Ms is None:
                return _x, _phi
            elif type(Ms) != np.ndarray:
                k = np.argmin(np.abs(Ms - _x))
                if abs(Ms - _x[k]) < 1e-3:
                    return _x[k], _phi[k]
                else:
                    return Ms, 10**np.interp(Ms, _x, np.log10(_phi),
                        left=-np.inf, right=-np.inf)
            elif _x.size == Ms.size:
                if np.allclose(_x, Ms):
                    return _x, _phi

            return Ms, 10**np.interp(Ms, _x, np.log10(_phi),
                left=-np.inf, right=-np.inf)

        return None

    def get_history(self, i):
        """
        Extract a single halo's trajectory from full set, return in dict form.
        """

        # These are kept in ascending redshift just to make life difficult.
        raw = self.histories

        hist = {'t': raw['t'], 'z': raw['z'],
            'SFR': raw['SFR'][i], 'Mh': raw['Mh'][i],
            'bursty': raw['bursty'][i], 'Nsn': raw['Nsn'][i]}

        if self.pf['pop_dust_yield'] is not None:
            hist['rand'] = raw['rand'][i]
            hist['Sd'] = raw['Sd'][i]

        if self.pf['pop_enrichment']:
            hist['Z'] = raw['Z'][i]

        if 'child' in raw:
            if raw['child'] is not None:
                hist['child'] = raw['child'][i]
            else:
                hist['child'] = None

        return hist

    #def get_histories(self, z):
    #    for i in range(self.histories['Mh'].shape[0]):
    #        yield self.get_history(i)

    @property
    def synth(self):
        if not hasattr(self, '_synth'):
            self._synth = SpectralSynthesis(**self.pf)
            self._synth.src = self.src
            self._synth.oversampling_enabled = self.pf['pop_ssp_oversample']
            self._synth.oversampling_below = self.pf['pop_ssp_oversample_age']
            self._synth.careful_cache = self.pf['pop_synth_cache_level']

        return self._synth

    def Magnitude(self, z, MUV=None, wave=1600., cam=None, filters=None,
        filter_set=None, dlam=20., method='gmean', idnum=None, window=1,
        load=True, presets=None, absolute=True):
        """
        For backward compatibility as we move to get_* method model.

        See `get_mags` below.
        """
        return self.get_mags(z, MUV=MUV, wave=wave, cam=cam, filters=filters,
            filter_set=filter_set, dlam=dlam, method=method, idnum=idnum,
            window=window, load=load, presets=presets, absolute=absolute)

    def get_mags(self, z, MUV=None, wave=1600., cam=None, filters=None,
        filter_set=None, dlam=20., method='closest', idnum=None, window=1,
        load=True, presets=None, absolute=True):
        """
        Return the magnitude of objects at specified wavelength or
        as-estimated via given photometry.

        Parameters
        ----------
        z : int, float
            Redshift of object(s)
        wave : int, float
            If `cam` and `filters` aren't supplied, return the monochromatic
            AB magnitude at this wavelength [Angstroms].
        absolute : bool
            If True, return absolute magnitude. [Default: True]
        cam : str, tuple
            Single camera or tuple of cameras that contain the filters named
            in `filters`, e.g., cam=('wfc', 'wfc3')
        filters : tuple
            List (well, tuple) of filters to be used in estimating the
            magnitude of objects.
        method : str
            How to combined photometric measurements to estimate magnitude?
            Can compute geometric mean (method='gmean'), can interpolate
            between photometry to estimate magnitude at specified wavelength
            `wave` (method='interp'), use filter closest to wavelength provided
            (method='closest'), or return monochromatic magnitude (method='mono')
        dlam : int, float
            Wavelength resolution (in Angstrom) with which to sample underlying
            spectra of objects. 20 is optimized for speed and accuracy.
        window : int
            Can optionally compute magnitude as the intrinsic spectrum
            centered at `wave` but convolved with a `window`-pixel boxcar.

        Returns
        -------
        Tuple containing the (photometric filters, magnitudes). If you set
        method != None, or if you're not doing photometry, the first entry
        of this tuple will be None. The keyword argument 'absolute' determines
        if the output magnitudes are apparent or absolute AB magnitudes.

        """

        if presets is not None:
            filter_set = None
            cam, filters = self._get_presets(z, presets)

        if type(filters) is dict:
            filters = filters[round(z)]

        if type(filters) == str:
            filters = (filters, )

        # Don't put any binning stuff in here!
        kw = {'z': z, 'cam': cam, 'filters': filters, 'window': window,
            'filter_set': filter_set, 'dlam':dlam, 'method': method,
            'wave': wave, 'absolute': absolute}

        kw_tup = tuple(kw.items())

        if load:
            cached_result = self._cache_mags(kw_tup)
        else:
            cached_result = None

        # Compute magnitude correction factor
        dL = self.cosm.LuminosityDistance(z) / cm_per_pc
        magcorr = 5. * (np.log10(dL) - 1.) - 2.5 * np.log10(1. + z)

        # Either load previous result or compute from scratch
        fil = filters
        if cached_result is not None:
            M, mags, xph = cached_result
        else:
            # Take monochromatic (or within some window) MUV
            L = self.get_lum(z, wave=wave, window=window, load=load)

            M = self.magsys.L_to_MAB(L)
            # May or may not use this.

            ##
            # Compute apparent magnitudes from photometry
            if (filters is not None) or (filter_set is not None):
                assert cam is not None

                hist = self.histories

                if type(cam) not in [tuple, list]:
                    cam = [cam]

                mags = []
                xph  = []
                fil = []
                for j, _cam in enumerate(cam):
                    _filters, xphot, dxphot, ycorr = \
                        self.synth.get_photometry(zobs=z, sfh=hist['SFR'],
                        zarr=hist['z'], hist=hist, dlam=dlam,
                        cam=_cam, filters=filters, filter_set=filter_set,
                        idnum=idnum, extras=self.extras, rest_wave=None)

                    mags.extend(list(np.array(ycorr)))
                    xph.extend(xphot)
                    fil.extend(_filters)

                mags = np.array(mags)
            else:
                xph = None
                mags = M + magcorr

            if hasattr(self, '_cache_mags_'):
                self._cache_mags_[kw_tup] = M, mags, xph

        ##
        # Interpolate etc.
        ##
        xout = None
        if (filters is not None) or (filter_set is not None):
            hist = self.histories

            # Can return all photometry
            if method is None:
                xout = fil
                Mg = mags
            elif len(filters) == 1:
                xout = filters
                Mg = mags.squeeze()
            # Or combine in some way below
            elif method == 'gmean':
                if len(mags) == 0:
                    Mg = -99999 * np.ones(hist['SFR'].shape[0])
                else:
                    Mg = np.nanprod(np.abs(mags), axis=0)**(1. / float(len(mags)))

                    if not (np.all(mags < 0) or np.all(mags > 0)):
                        raise ValueError('If geometrically averaging magnitudes, must all be the same sign!')

                    Mg = -1 * Mg if np.all(mags < 0) else Mg

                Mg = Mg.squeeze()

            elif method == 'closest':
                if len(mags) == 0:
                    Mg = -99999 * np.ones(hist['SFR'].shape[0])
                else:
                    # Get closest to specified rest-wavelength
                    rphot = np.array(xph) * 1e4 / (1. + z)
                    k = np.argmin(np.abs(rphot - wave))
                    Mg = mags[k,:]
            elif method == 'interp':
                if len(mags) == 0:
                    Mg = -99999 * np.ones(hist['SFR'].shape[0])
                else:
                    rphot = np.array(xph) * 1e4 / (1. + z)
                    kall = np.argsort(np.abs(rphot - wave))
                    _k1 = kall[0]#np.argmin(np.abs(rphot - wave))

                    if len(kall) == 1:
                        Mg = mags[_k1,:]
                    else:
                        _k2 = kall[1]

                        if rphot[_k2] < rphot[_k1]:
                            k1 = _k2
                            k2 = _k1
                        else:
                            k1 = _k1
                            k2 = _k2

                        dy = mags[k2,:] - mags[k1,:]
                        dx = rphot[k2] - rphot[k1]
                        m = dy / dx

                        Mg = mags[k1,:] + m * (wave - rphot[k1])

            elif method == 'mono':
                if len(mags) == 0:
                    Mg = -99999 * np.ones(hist['SFR'].shape[0])
                else:
                    Mg = M
            else:
                raise NotImplemented('method={} not recognized.'.format(method))

            if MUV is not None:
                Mout = np.interp(MUV, M[-1::-1], Mg[-1::-1])
            else:
                Mout = Mg
        else:
            Mout = mags

        if absolute:
            M_final = Mout - magcorr
        else:
            M_final = Mout

        return xout, M_final

    def Luminosity(self, z, wave=1600., band=None, idnum=None, window=1,
        load=True, energy_units=True):
        """
        For backward compatibility as we move to get_* method model.

        See `get_lum` below.
        """
        return self.get_lum(z, wave=wave, band=band, idnum=idnum,
            window=window, load=load, energy_units=energy_units)

    def _dlam_check(self, dlam):
        if self.pf['pop_sed_degrade'] is None:
            pass
        else:
            s = "`dlam` provided is finer than native SED resolution! "
            s += 'See `pop_sed_degrade` parameter, and set to value <= desired `dlam`.'
            assert (dlam >= self.pf['pop_sed_degrade']), s

    def get_waves_for_line(self, waves, dlam=1., window=3):
        """
        If `waves` is a string, e.g., 'Ly-a', convert to array of wavelengths.

        .. note :: This is used as a convenience routine to let the user
            retrieve the flux from a given spectral line rather than having to
            specify wavelengths by hand.

        Parameters
        ----------
        waves : str or np.ndarray
            If str, figure out what line user wants and create an array
            of wavelengths in [Angstrom]. Otherwise, just return.
        dlam : int
            Resolution to sample spectrum around line. [Angstrom]
        window : int
            Number of pixels to include around line. Must be odd!


        Returns
        -------
        Array of wavelengths to use in, e.g., `get_spec_obs` or `get_flux`.

        """
        # Special mode: retrieve line luminosity.
        if isinstance(waves, str):
            assert waves in known_lines, \
                "Unrecognized line={}. Options={}".format(waves, known_lines)

            self._dlam_check(dlam)

            i = known_lines.index(waves)
            l0 = known_line_waves[i]
            if window == 1:
                waves = np.array([l0])
            else:
                assert window % 2 == 1, "`window` must be odd!"
                w = (window - 1) // 2
                waves = np.arange(l0 - w * dlam, l0 + (w + 1) * dlam, dlam)

            return waves
        else:
            return waves

    def get_line_lum(self, z, line):
        """
        Get line luminosity in [erg/s/Hz].

        Parameters
        ----------
        z : int, float
            Redshift of galaxies.
        line : str
            String representation of line of interest. Currently, only option
            is "Ly-a".

        Returns
        -------
        Tuple containing (rest wavelengths [Angstrom], flux [erg/s/Hz]).

        """
        waves = self.get_waves_for_line(line)
        L = np.array([self.get_lum(z, wave) for wave in waves])
        return waves, L

    def get_line_flux(self, z, line, integrate=True, redden=True):
        """
        Compute line flux at z=0.

        .. note :: This computes the intrinsic line luminosity and integrates
            [optionally] there in order to avoid redshifting effects, i.e.,
            because we inject lines as delta functions, if we do the integral
            in the observer frame, we'll get too much dilution, since the
            wavelength range between bin edges is bigger.

        Parameters
        ----------
        z : int, float
            Redshift of galaxies.
        line : str
            String representation of line of interest. Currently, only option
            is "Ly-a".
        integrate : bool
            If True, integrate flux to obtain result in erg/s. If False,
            returned value will be flux in [erg/s/Hz].

        Returns
        -------
        Tuple containing (observed wavelengths [micron],
            flux [units set by value of `integrate`; see above]).

        """

        waves, L = self.get_line_lum(z, line)

        imid = (waves.size - 1) // 2
        owaves = waves * 1e-4 * (1. + z)
        line_wave = owaves[imid]

        dL = self.cosm.LuminosityDistance(z)

        flux = L / (4. * np.pi * dL**2)
        # dnu_rest/dnu_obs
        flux *= (1. + z)

        if integrate:
            # `waves` are bin centers
            waves_e = bin_c2e(waves)
            freq_e = c / (waves_e * 1e-8)
            dnu = -1 * np.diff(freq_e)

            flux = flux[imid] * dnu[imid]
        else:
            flux = flux[imid]

        return line_wave, flux

    def get_spec_obs(self, z, waves):
        """
        Generate z=0 observed spectrum for all sources.

        Parameters
        ----------
        z : int, float
            Redshift.
        waves : np.ndarray
            Array of rest-wavelengths to probe (in Angstrom).

        Returns
        -------
        A tuple containing (observed wavelengths [microns], flux [erg/s/Hz]).

        Note that the flux array is 2-D, with the first axis corresponding to
        halo mass bins.

        """

        owaves, flux = self.synth.get_spec_obs(z, hist=self.histories,
            waves=waves, sfh=self.histories['SFR'], extras=self.extras)

        return owaves, flux

    def get_lum(self, z, wave=1600., band=None, idnum=None, window=1,
        load=True, energy_units=True):
        """
        Return the luminosity for one or all sources at wavelength `wave`.

        Parameters
        ----------
        z : int, float
            Redshift of observation.
        wave : int, float
            Rest wavelength of interest [Angstrom]
        band : tuple
            Can alternatively request the average luminosity in some wavelength
            interval (again, rest wavelengths in Angstrom).
        window : int
            Can alternatively retrive the average luminosity at specified
            wavelength after smoothing intrinsic spectrum with a boxcar window
            of this width (in pixels).
        idnum : int
            If supplied, will only determine the luminosity for a single object
            (the one at this position in the array).
        cache : bool
            If False, don't save luminosities to cache. Needed sometimes to
            conserve memory if, e.g., computing luminosity for a ton of
            wavelengths for many (millions) of halos.

        Returns
        -------
        Luminosity (or luminosities if idnum=None) of object(s) in the model.


        """
        cached_result = self._cache_L((z, wave, band, idnum, window,
            energy_units))
        if load and (cached_result is not None):
            return cached_result

        #if band is not None:
        #    assert self.pf['pop_dust_yield'] in [0, None], \
        #        "Going to get weird answers for L(band != None) if dust is ON."

        raw = self.histories
        if (wave is not None) and (wave > self.src.wavelengths.max()):
            L = self.dust.Luminosity(z=z, wave=wave, band=band, idnum=idnum,
                window=window, load=load, energy_units=energy_units)
        else:
            L = self.synth.get_lum(wave=wave, zobs=z, hist=raw,
                extras=self.extras, idnum=idnum, window=window, load=load,
                band=band, energy_units=energy_units)

        self._cache_L_[(z, wave, band, idnum, window, energy_units)] = L.copy()

        return L

    def get_bias(self, z, limit=None, wave=1600., cam=None, filters=None,
        filter_set=None, dlam=20., method='closest', idnum=None, window=1,
        load=True, presets=None, cut_in_flux=False, cut_in_mass=False,
        absolute=False, factor=1, limit_is_lower=True, limit_lower=None,
        depths=None, color_cuts=None, logic='or'):
        """
        Compute the linear bias of sources above some limiting magnitude or
        flux.

        Parameters
        ----------
        z : int, float
            Redshift.
        limit : int, float
            Limiting magnitude of survey.
        limit_is_lower : bool
            If True (default), assumes we're interested in objects brighter
            than `limit`.
        limit_lower : int, float
            If `limit_is_lower` is False, then we need to provide another
            limiting magnitude to bracket the range of interest, i.e.,
            `limit_lower` > `limit`.
        depths : tuple
            Can supply a set of limiting magnitudes instead of a single
            limiting magnitude if photometry in multiple bands is retrieved
            (via `filters`). Use `logic` keyword argument to control how many
            bands an object must be detected in to be included in sample.
        logic : str, int
            If 'or', a detection in any filter is sufficient for inclusion. If
            'and', a detection in ALL `filters` is required. If an integer,
            only objects with a detection in N>=`logic` bands will be included
            in the sample.
        cut_in_flux : bool
            Not yet implemented.
        color_cuts : tuple
            Not yet implemented (really).

        Returns
        -------
        Galaxy bias at redshift `z`.

        """

        assert (limit is not None) or (depths is not None), \
            "Must supply `limit` or `depths`!"

        # In this case, just use GalaxyCohort class's version.
        if cut_in_mass:
            return self.guide.get_bias(z, limit, cut_in_mass=True)

        # Otherwise, use machinery here.
        _nh = self.get_field(z, 'nh')
        _Mh = self.get_field(z, 'Mh')

        _Lh = self.get_lum(z, wave=wave, window=window)

        iz = np.argmin(np.abs(z - self.halos.tab_z))

        bh = np.interp(np.log10(_Mh), np.log10(self.halos.tab_M),
            self.halos.tab_bias[iz,:])

        if cut_in_flux:
            raise NotImplemented('help')
        else:
            filt, mags = self.get_mags(z, wave=wave, cam=cam,
                filters=filters, filter_set=filter_set, dlam=dlam, method=method,
                idnum=idnum, window=window, load=load, presets=presets,
                absolute=absolute)

            if depths is not None:
                assert len(depths) == len(filt)
                assert method is None

                _ok = np.zeros(mags.shape[1])
                for i, limit in enumerate(depths):
                    _ok_ = np.logical_and(mags[i] <= limit, np.isfinite(mags[i]))
                    _ok += _ok_

                if logic == 'and':
                    ok = _ok == len(depths)
                elif logic == 'or':
                    ok = _ok > 0
                else:
                    assert isinstance(logic, int)
                    ok = _ok > logic

            elif limit_is_lower:
                ok = np.logical_and(mags <= limit, np.isfinite(mags))
            else:
                assert limit_lower is not None, \
                    "Provide `limit_lower` if isolating faint population."

                ok = np.logical_and(mags >= limit, mags <= limit_lower)
                ok = np.logical_and(ok, np.isfinite(mags))
            #else:
            #    if mags.ndim == 2:
            #        ok = np.ones(mags.shape[1])
            #    else:
            #        ok = np.ones_like(mags)

            if color_cuts is not None:
                assert depths is not None

                if type(color_cuts) != list:
                    color_cuts = [color_cuts]

                # Augment `ok`
                for cut in color_cuts:
                    filt1, _filt2 = cut.split('-')
                    if '>' in _filt2:
                        is_lo = True
                        filt2, thresh = _filt2.split('>')
                    else:
                        is_lo = False
                        filt2, thresh = _filt2.split('<')

                    color = mags[filt.index(filt1)] - mags[filt.index(filt2)]

                    print('color cut {} - {}'.format(filt1, filt2))
                    print(filt1, filt2, thresh, sum(color < float(thresh)), color.size)

                    if is_lo:
                        ok[color < float(thresh)] = 0
                    else:
                        ok[color > float(thresh)] = 0

        ##
        # Apply cut and integrate
        integ_top = bh[ok==1] * _nh[ok==1]
        integ_bot = _nh[ok==1]

        # Don't do trapz -- not a continuous curve like in GalaxyCohort.
        b = np.sum(integ_top * _Mh[ok==1]) / np.sum(integ_bot * _Mh[ok==1])

        return b

    def get_bias_from_scaling_relations(self, z, smhm, uvsm, limit,
        return_funcs=False, use_dpl_smhm=False, Mpeak=None, extrap=True):
        """
        Compute the galaxy bias from stellar-mass-halo-mass (SMHM) relation
        and the UV magnitude -- stellar mass relation (UVSM).

        .. note :: Limiting magnitude must be provided as absolute AB mag.

        .. note :: Will first construct fitting functions if scaling laws
            provided as discrete points.

        Parameters
        ----------
        z : int, float
            Redshift.
        smhm : tuple, FunctionType
            Contains arrays of (stellar mass, stellar mass / halo mass).
        uvsm : tuple, FunctionType
            Contains arrays of (MUV, stellar mass / Msun).
        limit : int, float
            Limiting magnitude of survey (absolute, AB).

        Returns
        -------
        Average bias of galaxies.

        """

        # Convert points in (Mh, Ms/Mh) to function.
        if type(smhm) == tuple:
            _Mh, _smhm = smhm

            ok = _smhm > 0
            _Mh = _Mh[ok==1]
            _smhm = _smhm[ok==1]

            smhm_max = max(_smhm)

            # Fit to Mh -- Mstell. Anchor to Mh=1e10
            def func(x, p0, p1):
                return _linfunc(x, 10, p0, p1)

            popt1, pcov1 = curve_fit(func, np.log10(_Mh), np.log10(_Mh * _smhm),
                p0=[1., 8.], maxfev=10000)

            if use_dpl_smhm:
                assert Mpeak is not None, \
                    "Must supply `Mpeak` if `use_dpl_smhm`=True!"

                from ..phenom.ParameterizedQuantity import DoublePowerLaw

                _Ms_of_Mh = DoublePowerLaw(pq_func_var='Mh',
                    pq_func_par0=10**popt1[1], pq_func_par1=Mpeak,
                    pq_func_par2=popt1[0], pq_func_par3=-1, pq_func_par4=1e10)

                Ms_of_Mh = lambda Mh: _Ms_of_Mh(Mh=Mh) #* Mh

            else:
                _Ms_of_Mh = lambda Mh: 10**_linfunc(np.log10(Mh), 10, popt1[0],
                    popt1[1])

                def Ms_of_Mh(Mh):

                    if extrap:
                        return _Ms_of_Mh(Mh)

                    if Mpeak is not None:
                        if type(Mh) == np.ndarray:
                            ok = Mh < Mpeak
                        elif Mh > Mpeak:
                            ok = 0
                        else:
                            ok = 1
                    else:
                        ok = 1

                    return np.minimum(_Ms_of_Mh(Mh), smhm_max * Mh) * ok

        else:
            assert type(smhm) == FunctionType
            Ms_of_Mh = lambda Mh: Mh * smhm(Mh)


        if type(uvsm) == tuple:
            _MUV, _Mst = uvsm

            ok = _Mst > 0
            _MUV = _MUV[ok==1]
            _Mst = _Mst[ok==1]

            # Fit to MUV -- Mstell. Anchor to Mstell=1e8
            def func(x, p0, p1):
                return _linfunc(x, 8, p0, p1)

            popt2, pcov2 = curve_fit(func, np.log10(_Mst), _MUV,
                p0=[1., -22], maxfev=10000)
            MUV_of_Ms = lambda Ms: _linfunc(np.log10(Ms), 8, popt2[0], popt2[1])
        else:
            MUV_of_Ms = uvsm

        # Need to map MUV onto Mh so that we can determine the galaxies
        # brighter than `limit`.

        # Map functions of Mh onto tabulated halo arrays
        iz = np.argmin(np.abs(z - self.halos.tab_z))

        nh = self.halos.tab_dndm[iz,:]
        bh = self.halos.tab_bias[iz,:]
        tab_M = self.halos.tab_M

        tab_Ms = Ms_of_Mh(Mh=tab_M)
        tab_MUV = MUV_of_Ms(tab_Ms)
        ok = tab_MUV <= limit

        integ_top = bh[ok==1] * nh[ok==1]
        integ_bot = nh[ok==1]

        # Integrate in log-space
        b = np.trapz(integ_top * tab_M[ok==1]**2, x=np.log(tab_M[ok==1])) \
          / np.trapz(integ_bot * tab_M[ok==1]**2, x=np.log(tab_M[ok==1]))


        if return_funcs:
            return b, Ms_of_Mh, MUV_of_Ms
        else:
            return b

    def get_uvlf(self, z, bins):
        """
        Compute what people usually mean by the UVLF.
        """
        return self.get_lf(z, bins, use_mags=True, wave=1600., window=51.,
            absolute=True)

    def get_irlf(self):
        pass

    def LuminosityFunction(self, z, bins=None, use_mags=True, wave=1600.,
        window=1,
        band=None, cam=None, filters=None, filter_set=None, dlam=20.,
        method='closest', load=True, presets=None, absolute=True, total_IR=False):

        return self.get_lf(z, bins=bins, use_mags=use_mags,
            wave=wave, window=window, band=band, cam=cam, filters=filters,
            filter_set=filter_set, dlam=dlam, method=method,
            load=load, presets=presets, absolute=absolute,
            total_IR=total_IR)

    def get_lf(self, z, bins=None, use_mags=True, wave=1600.,
        window=1, band=None, cam=None, filters=None, filter_set=None,
        dlam=20., method='closest', load=True, presets=None, absolute=True,
        total_IR=False):
        """
        Compute the luminosity function from discrete histories.

        If given redshift not exactly in the grid, we'll compute the LF
        at the closest redshift *below* that requested.

        Parameters
        ----------

        z: int, float
            Redshift of interest.
        use_mags : boolean
            if True: returns bin centers in AB magnitudes, whether
            absolute or apparent depends on value of `absolute` parameter.
            if False: returns bin centers in log(L / Lsun)

        wave :  int, float
            wavelength in Angstroms to be looked at. If wave > 3e5, then
            the luminosity function comes from the dust in the galaxies.

        window : int
            Can alternatively retrive the average luminosity at specified
            wavelength after smoothing intrinsic spectrum with a boxcar window
            of this width (in pixels).

        band : tuple
            Can alternatively request the average luminosity in some wavelength
            interval (again, rest wavelengths in Angstrom).

        total_IR : boolean
            if False: returns luminosity function at the given wavelength
            if True: returns the total infrared luminosity function for wavelengths
            between 8 and 1000 microns.
            Note: if True, ignores wave and band keywords, and always returns in log(L / Lsun)

        """
        if total_IR:
            wave = 'total'

        cached_result = self._cache_lf(z, bins, wave)

        if (cached_result is not None) and load:
            print("WARNING: should we be doing this?")
            return cached_result

        # These are kept in descending redshift just to make life difficult.
        # [Useful for SAM to proceed in ascending time order]
        raw = self.histories
        keys = raw.keys()

        # Find the z grid point closest to that requested.
        # Must be >= z requested.
        izobs = np.argmin(np.abs(raw['z'] - z))
        if raw['z'][izobs] > z:
            # Go to one grid point lower redshift
            izobs += 1

        # Currently izobs is such that the requested redshift is
        # just higher in redshift, i.e., (izobs, izobs+1) brackets
        # the requested redshift.

        # Make sure we don't overshoot end of array.
        # Choices about fwd vs. backward differenced MARs will matter here.
        # Note that this only used for `nh` below, ultimately a similar thing
        # happens inside self.synth.Luminosity.
        izobs = min(izobs, len(raw['z']) - 1)

        ##
        # Run in batch.
        #if total_IR:
        #    L = self.dust.Luminosity(z, total_IR=True)
        #    mags = False
        #else:
        #    L = self.Luminosity(z, wave=wave, band=band, window=window)

        # Need to be more careful here as nh can change when using
        # simulated halos
        w = raw['nh'][:,izobs] # used to be izobs+1, I belive in error.

        if use_mags:
            #_MAB = self.magsys.L_to_MAB(L)
            filt, mags = self.get_mags(z, wave=wave, cam=cam,
                filters=filters, presets=presets, dlam=dlam, window=window,
                method=method, absolute=absolute, load=load)

            #z, MUV=None, wave=1600., cam=None, filters=None,
            #    filter_set=None, dlam=20., method='closest', idnum=None, window=1,
            #    load=True, presets=None, absolute=True

            if mags.shape[0] == 1:
                mags = mags[0,:]
            else:
                assert mags.ndim == 1
        else:
            L = self.get_lum(z, wave=wave, band=band, window=window, load=load)

        #elif total_IR:
        #    _MAB = np.log10(L / Lsun)
        #else:
        #    _MAB = np.log10(L * c / (wave * 1e-8) / Lsun)

        if use_mags:
            if (self.pf['dustcorr_method'] is not None) and absolute:
                y = self.dust.Mobs(z, mags)
            else:
                y = mags

            yok = np.isfinite(mags)
        else:
            #raise NotImplemented('help')
            y = L
            yok = np.logical_and(L > 0, np.isfinite(L))

        # If L=0, MAB->inf. Hack these elements off if they exist.
        # This should be a clean cut, i.e., there shouldn't be random
        # spikes where L==0, all L==0 elements should be a contiguous chunk.

        # Always bin to setup cache, interpolate from then on.
        if bins is not None:
            x = bins
        elif use_mags:
            ymin = x.min()
            ymax = x.max()
            if absolute:
                x = np.arange(ymin*0.5, ymax*2, self.pf['pop_mag_bin'])
            else:
                x = np.arange(ymin*0.5, ymax*2, self.pf['pop_mag_bin'])
        elif not total_IR:
            x = np.arange(4, 12, 0.25)
        else:
            x = np.arange(6.5, 14, 0.25)

        if yok.sum() == 0:
            return x, np.zeros_like(x)

        # Make sure binning range covers the range of luminosities/magnitudes
        if use_mags:
            mi, ma = y[yok==1].min(), y[yok==1].max()
            assert mi > x.min(), "{} NOT > {}".format(mi, x.min())
            assert ma < x.max(), "{} NOT < {}".format(ma, x.max())
        else:
            assert y[yok==1].min() < x.min()
            assert y[yok==1].max() > x.max()

        hist, bin_histedges = np.histogram(y[yok==1],
            weights=w[yok==1], bins=bin_c2e(x), density=True)

        N = np.sum(w[yok==1])
        phi = hist * N

        #self._cache_lf_[(z, wave)] = x, phi

        return x, phi

    def _cache_beta(self, kw_tup):

        if not hasattr(self, '_cache_beta_'):
            self._cache_beta_ = {}

        if kw_tup in self._cache_beta_:
            return self._cache_beta_[kw_tup]

        return None

    def _cache_mags(self, kw_tup):

        if not hasattr(self, '_cache_mags_'):
            self._cache_mags_ = {}

        if kw_tup in self._cache_mags_:
            return self._cache_mags_[kw_tup]

        return None

    @property
    def extras(self):
        if not hasattr(self, '_extras'):
            if self.pf['pop_dust_yield'] is not None:
                self._extras = {'kappa': self.guide.dust_kappa}
            else:
                self._extras = {}
        return self._extras

    def _get_presets(self, z, presets, for_beta=True, wave_range=None):
        """
        Convenience routine to retrieve `cam` and `filters` via short-hand.

        Parameters
        ----------
        z : int, float
            Redshift of interest.
        presets : str
            Name of presets package to use, e.g., 'hst', 'jwst', 'hst+jwst'.
        for_beta : bool
            If True, will restrict to filters used to compute UV slope.
        wave_range : tuple
            If for_beta==False, can restrict photometry to specified range
            of rest wavelengths (in Angstroms). If None, will return all
            photometry.

        Returns
        -------
        Tuple containing (i) list of cameras, and (ii) list of filters.

        """

        zstr = int(round(z))

        if ('hst' in presets.lower()) or ('hubble' in presets.lower()):
            hst_shallow = self._b14.filt_shallow
            hst_deep = self._b14.filt_deep

            if zstr >= 7:
                filt_hst = hst_deep
            else:
                filt_hst = hst_shallow

        if ('jwst' in presets.lower()) or ('nircam' in presets.lower()):
            nircam_M, nircam_W = self._nircam

        ##
        # Hubble only
        if presets.lower() in ['hst', 'hubble']:
            if for_beta:
                cam = ('wfc', 'wfc3')
                filters = filt_hst[zstr]
            else:
                raise NotImplemented('help')

        # JWST only
        elif ('jwst' in presets.lower()) or ('nircam' in presets.lower()): # pragma: no cover

            if for_beta:
                # Override
                if z < 4:
                    raise ValueError("JWST too red for UV slope measurements at z<4!")

                cam = ('nircam',)

                wave_lo, wave_hi = np.min(self._c94), np.max(self._c94)

                if presets.lower() in ['jwst-m', 'jwst', 'nircam-m', 'nircam']:
                    filters = list(get_filters_from_waves(z, nircam_M, wave_lo,
                        wave_hi))

                    ct = 1
                    while len(filters) < 2:
                        filters = get_filters_from_waves(z, nircam_M, wave_lo,
                            wave_hi + 10 * ct)

                        ct += 1

                    if ct > 1:
                        print("For JWST M filters at z={}, extended wave_hi to {}A".format(z,
                            wave_hi + 10 * (ct - 1)))

                else:
                    filters = []

                if presets.lower() in ['jwst-w', 'jwst', 'nircam-w', 'nircam']:
                    nircam_W_fil = get_filters_from_waves(z, nircam_W, wave_lo,
                        wave_hi)

                    ct = 1
                    while len(nircam_W_fil) < 2:
                        nircam_W_fil = get_filters_from_waves(z, nircam_W, wave_lo,
                            wave_hi + 10 * ct)

                        ct += 1

                    if ct > 1:
                        print("For JWST W filters at z={}, extended wave_hi to {}A".format(z,
                            wave_hi + 10 * (ct - 1)))

                    filters.extend(list(nircam_W_fil))

                filters = tuple(filters)

                if len(filters) < 2:
                    raise ValueError('Need at least 2 filters to compute slope.')

            else:
                raise NotImplemented('help')

        # Combo
        elif presets == 'hst+jwst':

            if for_beta:
                cam = ('wfc', 'wfc3', 'nircam') if zstr <= 8 else ('nircam', )
                filters = filt_hst[zstr] if zstr <= 8 else None

                if filters is not None:
                    filters = list(filters)
                else:
                    filters = []

                wave_lo, wave_hi = np.min(self._c94), np.max(self._c94)
                filters.extend(list(get_filters_from_waves(z, nircam_M, wave_lo,
                    wave_hi)))
                filters.extend(list(get_filters_from_waves(z, nircam_M, wave_lo,
                    wave_hi)))
                filters = tuple(filters)
            else:
                 raise NotImplemented('help')

        elif presets.lower() in ['c94', 'calzetti', 'calzetti1994']:
            return ('calzetti', ), self._c94
        elif presets.lower() in ['roman', 'rst', 'wfirst']:
            cam = 'roman',
            wave_lo, wave_hi = np.min(self._c94), np.max(self._c94)
            filters = tuple((get_filters_from_waves(z, self._roman, wave_lo,
                wave_hi)))
        else:
            raise NotImplemented('No presets={} option yet!'.format(presets))

        if self.pf['debug']:
            print("# Filters (z={}): {}".format(z, filters))

        ##
        # Done!
        return cam, filters

    def get_lae_fraction(self, z, bins, absolute=True, model=1, Tcrit=0.7,
        wave=1600., cam=None, filters=None, filter_set=None, dlam=20.,
        method='closest', window=1, load=True, presets=None):
        """
        Compute Lyman-alpha emitter (LAE) fraction vs. UV magnitude relation.

        .. note :: For now, this is based entirely on the empirical model
            described in Mirocha, Mason, & Stark (2020). Could be
            generalized in the future.

        Parameters
        ----------
        z : int, float
            Redshift.
        model : int, str
            Currently, only acceptable value is 1 (for MMS 2020 approach).
        bins : tuple
            Magnitude bins in which to compute LAE fraction. Whether
            absolute or apparent AB mags depends on `absolute` parameter.
        absolute: bool
            If True, assume `maglim` magnitudes are absolute, otherwise,
            apparent. Also controls type of magnitudes returned.

        Returns
        -------
        Tuple containing (bins, LAE fractions, scatter in LAE fraction within bin).

        """

        assert model == 1, "Haven't implemented any other LAE models!"

        nh = self.get_field(z, 'nh')

        filt, mags = self.get_mags(z, absolute=absolute, wave=wave, cam=cam,
            filters=filters, filter_set=filter_set, dlam=dlam, method=method,
            window=window, load=load, presets=presets)

        tau = self.get_dust_opacity(z, wave=wave)

        is_LAE = np.exp(-tau) > Tcrit

        ok = np.isfinite(mags)
        _x, _y, _err, _N = bin_samples(mags[ok==1], is_LAE[ok==1],
            bins, weights=nh[ok==1])

        return _x, _y, _err

    def get_dust_opacity(self, z, wave):
        """
        Compute dust opacity for every halo using Mirocha, Mason, & Stark (2020).
        """

        Mh = self.get_field(z, 'Mh')

        if self.pf['pop_dust_yield'] is None:
            return np.zeros_like(Mh)
        if self.pf['pop_dust_yield'] == 0:
            return np.zeros_like(Mh)

        kappa = self.guide.dust_kappa(wave=wave, Mh=Mh, z=z)
        Sd = self.get_field(z, 'Sd')
        return kappa * Sd

    def Beta(self, z, **kwargs):
        return self.get_beta(z, **kwargs)

    def get_beta(self, z, **kwargs):
        return self.get_uv_slope(z, **kwargs)

    def get_uv_slope(self, z, waves=None, rest_wave=None, cam=None,
        filters=None, filter_set=None, dlam=20., method='linear', magmethod='gmean',
        return_binned=False, Mbins=None, Mwave=1600., MUV=None, Mstell=None,
        return_scatter=False, load=True, massbins=None, return_err=False,
        presets=None):
        """
        Compute UV slope for all objects in model.

        Parameters
        ----------
        z : int, float
            Redshift.
        MUV : int, float, np.ndarray
            Optional. Set of magnitudes at which to return Beta.
            Note: these need not be at the same wavelength as that used
                  to compute Beta (see Mwave).
        wave : int, float
            Wavelength at which to compute Beta.
        Mwave : int, float
            Wavelength assumed for input MUV. Allows us to return, e.g., the
            UV slope at 2200 Angstrom corresponding to input 1600 Angstrom
            magnitudes.
        dlam : int, float
            Interval over which to average UV slope [Angstrom]
        return_binned : bool
            If True, return binned version of MUV(Beta), including
            standard deviation of Beta in each MUV bin.

        Returns
        -------
        if MUV is None:
            Tuple containing (magnitudes, slopes)
        else:
            Slopes at corresponding user-supplied MUV (assumed to be at
            wavelength `wave_MUV`).
        """

        if presets is not None:
            cam, filters = self._get_presets(z, presets)

        if type(filters) is dict:
            filters = filters[round(z)]

        # Don't put any binning stuff in here!
        kw = {'z':z, 'waves':waves, 'rest_wave':rest_wave, 'cam': cam,
            'filters': filters, 'filter_set': filter_set,
            'dlam':dlam, 'method': method, 'magmethod': magmethod}

        kw_tup = tuple(kw.items())

        if load:
            cached_result = self._cache_beta(kw_tup)
        else:
            cached_result = None

        if cached_result is not None:
            if len(cached_result) == 2:
                beta_r, beta_rerr = cached_result
            else:
                beta_r = cached_result

        else:
            raw = self.histories

            ##
            # Run in batch.
            _beta_r = self.synth.Slope(zobs=z, sfh=raw['SFR'], waves=waves,
                zarr=raw['z'], hist=raw, dlam=dlam, cam=cam, filters=filters,
                filter_set=filter_set, rest_wave=rest_wave, method=method,
                extras=self.extras, return_err=return_err)

            if return_err:
                beta_r, beta_rerr = _beta_r
            else:
                beta_r = _beta_r

            ##
            if hasattr(self, '_cache_beta_'):
                self._cache_beta_[kw_tup] = _beta_r

        # Can return binned (in MUV) version
        if return_binned:
            if Mbins is None:
                Mbins = np.arange(-30, -10, 1.)

            nh = self.get_field(z, 'nh')

            # This will be the geometric mean of magnitudes in `cam` and
            # `filters` if they are provided!
            if (presets == 'calzetti') or (cam == 'calzetti'):
                assert magmethod == 'mono', \
                    "Known issues with magmethod!='mono' and Calzetti approach."

            _filt, _MAB = self.get_mags(z, wave=Mwave, cam=cam,
                filters=filters, method=magmethod, presets=presets)

            if np.all(np.diff(np.diff(nh)) == 0):
                Mh = self.get_field(z, 'Mh')
                # L may be zero (and so MUV -inf) even for elements with Mh>0
                # because we generally (should) mask out first timestep MAR.
                ok = np.logical_and(Mh > 0, np.isfinite(_MAB))
            else:
                ok = np.ones_like(_MAB)

            # Hack for the time being
            ok = np.logical_and(ok, np.isfinite(beta_r))

            # Generally happens if our HMF tables don't extend to
            # this redshift.
            if ok.sum() == 0:
                print("# WARNING: all Magnitudes flagged.")
                print("# (z={} outside available HMF range?)".format(z))
                bad = -99999 * np.ones(ok.size)
                if return_scatter:
                    return bad, bad
                if return_err:
                    return bad, bad
                else:
                    return bad

            MAB, beta, _std, N1 = bin_samples(_MAB[ok==1], beta_r[ok==1],
                Mbins, weights=nh[ok==1])

        else:
            beta = beta_r

        if MUV is not None:
            return np.interp(MUV, MAB, beta, left=-99999, right=-99999)
        if Mstell is not None:
            Ms_r = self.get_field(z, 'Ms')
            nh_r = self.get_field(z, 'nh')
            x1, y1, err1, N1 = bin_samples(np.log10(Ms_r), beta_r, massbins,
                weights=nh_r)
            return np.interp(np.log10(Mstell), x1, y1, left=0., right=0.)

        # Get out.
        if return_scatter:
            assert return_binned

            return beta, _std

        if return_err:
            assert not return_scatter
            assert not return_binned

            return beta, beta_rerr
        else:
            return beta

    def AUV(self, z, Mwave=1600., cam=None, MUV=None, Mstell=None,
        magbins=None,
        massbins=None, return_binned=False, filters=None, dlam=20.):
        """
        For backward compatibility -- see `get_AUV` below.
        """

        return self.get_AUV(z, Mwave=Mwave, cam=cam, MUV=MUV, Mstell=Mstell,
        magbins=magbins, massbins=massbins, return_binned=return_binned,
        filters=filters, dlam=dlam)

    def get_AUV(self, z, Mwave=1600., cam=None, MUV=None, Mstell=None,
        magbins=None, massbins=None, return_binned=False, filters=None,
        dlam=20.):
        """
        Compute rest-UV extinction.

        Parameters
        ----------
        z : int, float
            Redshift.
        Mstell : int, float, np.ndarray
            Can return AUV as function of provided stellar mass.
        MUV : int, float, np.ndarray
            Can also return AUV as function of UV magnitude.
        magbins : np.ndarray
            Either the magnitude bins or log10(stellar mass) bins to use if
            return_binned==True.

        """

        if self.pf['pop_dust_yield'] is None:
            return None

        tau = self.get_dust_opacity(z, Mwave)

        AUV_r = np.log10(np.exp(-tau)) / -0.4

        # Just do this to get MAB array of same size as Mh
        _filt, MAB = self.Magnitude(z, wave=Mwave, cam=cam, filters=filters,
            dlam=dlam)

        if return_binned:
            if magbins is None:
                magbins = np.arange(-30, -10, 0.25)

            nh = self.get_field(z, 'nh')
            _x, _y, _z, _N = bin_samples(MAB, AUV_r, magbins, weights=nh)

            MAB = _x
            AUV = _y
            std = _z
        else:
            std = None
            AUV = AUV_r

            assert MUV is None
            MAB = None

        # May specify a single magnitude at which to return AUV
        if MUV is not None:
            return np.interp(MUV, MAB, AUV, left=0., right=0.)

        if Mstell is not None:
            Ms_r = self.get_field(z, 'Ms')
            nh_r = self.get_field(z, 'nh')

            x1, y1, err1, N1 = bin_samples(np.log10(Ms_r), AUV_r, massbins,
                weights=nh_r)
            return np.interp(np.log10(Mstell), x1, y1, left=0., right=0.)

        # Otherwise, return raw (or binned) results
        return MAB, AUV

    def get_dBeta_dMUV(self, z, magbins, presets=None, model='exp',
        return_funcs=False, maglim=None, dlam=20., magmethod='gmean',
        Mwave=1600.):
        """
        Compute gradient in UV slope with respect to UV magnitude.

        Parameters
        ----------

        """

        _beta, _std = self.get_beta(z, presets=presets, dlam=dlam,
            magmethod=magmethod, return_scatter=True, Mbins=magbins,
            return_binned=True)

        ok = np.isfinite(_beta)
        if maglim is not None:
            _ok = np.logical_and(magbins >= maglim[0], magbins <= maglim[1])
            ok = np.logical_and(ok, _ok)

        _x = magbins[ok==1]
        _y = _beta[ok==1]
        _err = _std[ok==1]

        if not np.any(ok):
            print("# All elements masked for dBeta/dMUV at z={}".format(z))
            print("# _y1=", _y1)
            return (None, None, None) if return_funcs else None

        # Arbitrary pivot magnitude
        x0 = -16.

        # Compute slopes wrt MUV
        if model == 'exp':
            def func(x, p0, p1, p2):
                return np.exp((x / p0)**p2) + p1

            popt, pcov = curve_fit(func, _x, _y, p0=np.array([-10, -2.5, 1]),
                maxfev=100000)

            recon = np.exp((_x / popt[0])**popt[2]) + popt[1]
            eder = popt[2] * (_x / popt[0])**(popt[2] - 1) \
                * np.exp((_x / popt[0])**popt[2]) / popt[0]
        elif model == 'linear':
            def func(x, p0, p1):
                return _linfunc(x, x0, p0, p1)

            popt, pcov = curve_fit(func, _x, _y, p0=[-0.5, -2.],
                maxfev=10000)
            recon = _linfunc(_x, x0, popt[0], popt[1])
            eder = popt[0] * np.ones_like(_x)
        else:
            raise NotImplementedError('Unrecognized model={}.'.format(model))

        # Create interpolants for Beta and its derivative
        _interp_ = lambda xx: np.interp(xx, _x, recon)
        _interpp_ = lambda xx: np.interp(xx, _x, eder)

        dBeta = np.array([_interpp_(_x_) for _x_ in magbins])

        if return_funcs:
            return dBeta, _interp_, _interpp_
        else:
            return dBeta

    def dBeta_dMstell(self, z, dlam=20., Mstell=None, massbins=None,
        model='quad3', return_funcs=False, masslim=None):
        """
        Compute gradient in UV color at fixed stellar mass.

        Parameters
        ----------
        z : int, float
            Redshift of interest.
        dlam : int
            Wavelength resolution to use for Beta calculation.
        Mstell : int, float, np.ndarray
            Stellar mass at which to evaluate slope. Can be None, in which
            case we'll return over a grid with resolution 0.5 dex.
        model : str
            Fit Beta(Mstell) with this function and then
            determine slope at given Mstell via interpolation (after
            analytic differentiation). Options: 'quad2', 'quad3',
            or 'linear'.
        """

        zstr = round(z)
        calzetti = read_lit('calzetti1994').windows

        # Compute raw beta and compare to Mstell
        beta_c94 = self.Beta(z, Mwave=1600., return_binned=False,
            cam='calzetti', filters=calzetti, dlam=dlam, rest_wave=None,
            magmethod='mono')

        Ms_r = self.get_field(z, 'Ms')
        nh_r = self.get_field(z, 'nh')

        # Compute binned version of Beta(Mstell).
        _x1, _y1, _err, _N = bin_samples(np.log10(Ms_r), beta_c94, massbins,
            weights=nh_r)

        ok = np.isfinite(_y1)

        if masslim is not None:
            _ok = np.logical_and(_x1 >= masslim[0], _x1 <= masslim[1])
            ok = np.logical_and(ok, _ok)

        _x1 = _x1[ok==1]
        _y1 = _y1[ok==1]
        _err = _err[ok==1]

        # Pivot at log10(Mstell) = 7
        x0 = 7.

        # Compute slopes with Mstell
        if model == 'quad2':
            def func(x, p0, p1):
                return _quadfunc2(x, x0, p0, p1)

            popt, pcov = curve_fit(func, _x1, _y1, p0=[0.1, -2.],
                sigma=_err, maxfev=10000)
            recon = popt[0] * (_x1 - x0)**2 + popt[1]
            eder = 2 * popt[0] * (_x1 - x0)
        elif model == 'quad3':
            def func(x, p0, p1, p2):
                return _quadfunc3(x, x0, p0, p1, p2)

            popt, pcov = curve_fit(func, _x1, _y1, p0=[0.1, -2., 0.],
                sigma=_err, maxfev=10000)

            recon = popt[0] * (_x1 - x0)**2 + popt[1] * (_x1 - x0) + popt[2]
            eder = 2 * popt[0] * (_x1 - x0) + popt[1]
        elif model == 'linear':
            def func(x, p0, p1):
                return _linfunc(x, x0, p0, p1)

            popt, pcov = curve_fit(func, _x1, _y1, p0=[0.1, -2.],
                sigma=_err, maxfev=10000)
            recon = popt[0] * (_x1 - x0) + popt[1]

        else:
            raise NotImplemented('Unrecognized model={}.'.format(model))

        # Create interpolants for Beta and its derivative
        _interp_ = lambda xx: np.interp(xx, _x1, recon)
        _interpp_ = lambda xx: np.interp(xx, _x1, eder)

        norm = []
        dBMstell = []
        for _x in _x1:
            dBMstell.append(_interpp_(_x))

        if return_funcs:
            return np.array(dBMstell), _interp_, _interpp_
        else:
            return np.array(dBMstell)

    def prep_hist_for_cache(self):
        keys = ['nh', 'MAR', 'Mh', 't', 'z']
        hist = {key:self.histories[key][-1::-1] for key in keys}
        return hist

    def SurfaceDensity(self, z, bins, dz=1., dtheta=1., wave=1600.,
        cam=None, filters=None, filter_set=None, depths=None, dlam=20.,
        method='closest', window=1, load=True, presets=None, absolute=True,
        use_mags=True):
        """
        For backward compatibility. See `get_surface_density`.
        """
        return self.get_surface_density(z=z, bins=bins, dz=dz, dtheta=dtheta,
            wave=wave, cam=cam, filters=filters, filter_set=filter_set,
            dlam=dlam, method=method, use_mags=use_mags, depths=depths,
            window=window, load=load, presets=presets, absolute=absolute)

    def get_surface_density(self, z, bins=None, dz=1., dtheta=1., wave=1600.,
        cam=None, filters=None, filter_set=None, depths=None, dlam=20.,
        method='closest', window=1, load=True, presets=None, absolute=False,
        use_mags=True, use_central_z=True, zstep=0.1, return_evol=False,
        use_volume=False, save_by_band=False):
        """
        Compute surface density of galaxies [number / deg^2 / dz]

        Returns
        -------
        Observed magnitudes, then, projected surface density of galaxies in
        `dz` thick shell, in units of cumulative number of galaxies per
        square degree.
        """

        if type(filters) not in [list, tuple]:
            filters = filters,

        if depths is not None:
            assert len(depths) == len(filters)

        Ngal = np.zeros((len(filters), bins.size))
        nltm = np.zeros((len(filters), bins.size))
        for i, filt in enumerate(filters):

            # Simplest thing: take central redshift, assume same UVLF throughout
            # dz interval along LOS.
            if use_central_z:

                # First, compute the luminosity function.
                x, phi = self.get_lf(z, bins=bins, wave=wave, cam=cam,
                    filters=filt, filter_set=filter_set, dlam=dlam, method=method,
                    window=window, load=load, presets=presets, absolute=absolute,
                    use_mags=use_mags)

                # Compute the volume of the shell we're looking at [cMpc^3]
                if use_volume:
                    vol = 1
                else:
                    vol = self.cosm.ProjectedVolume(z, angle=dtheta, dz=dz)

                # Get total number of galaxies in volume in each bin.
                Ngal[i,:] = phi * vol
            else:
                assert depths is None, "Not implemented!"

                # Sub-sample redshift interval
                zbin_e = np.arange(z - 0.5 * dz, z + 0.5 * dz, zstep)

                phi = np.zeros((zbin_e.size, bins.size))
                vol = np.zeros_like(zbin_e)
                for j, ze in enumerate(zbin_e):
                    zmid = ze + 0.5 * zstep

                    # Compute LF at midpoint of this bin.
                    x, phi[j] = self.get_lf(zmid, bins=bins, wave=wave, cam=cam,
                        filters=filt, filter_set=filter_set, dlam=dlam, method=method,
                        window=window, load=load, presets=presets, absolute=absolute,
                        use_mags=use_mags)

                    # Compute the volume of the shell we're looking at [cMpc^3]
                    if use_volume:
                        vol[j] = 1
                    else:
                        vol[j] = self.cosm.ProjectedVolume(zmid, angle=dtheta,
                            dz=zstep)

                # Integrate over the redshift interval
                Ngal[i,:] = np.sum(phi * vol[:,None], axis=0)

            # Faint to bright
            Ngal_asc = Ngal[i,-1::-1]
            x_asc = x[-1::-1]

            # At this point, magnitudes are in ascending order, i.e., bright to
            # faint.

            # Cumulative surface density of galaxies *brighter than*
            # some corresponding magnitude
            assert Ngal[i,0] == 0, "Broaden binning range?"
            #ntot = np.trapz(Ngal[i,:], x=x)
            nltm[i,:] = cumtrapz(Ngal[i,:], x=x, initial=Ngal[i,0])

        # Can just return *maximum* number of galaxies detected,
        # regardless of band. Equivalent to requiring only single-band
        # detection.
        if not save_by_band:
            nltm = np.max(nltm, axis=0)

        if return_evol and (not use_central_z):
            return x, nltm, zbin_e, phi, vol
        else:
            return x, nltm

    def get_volume_density(self, z, bins=None, wave=1600.,
        cam=None, filters=None, filter_set=None, dlam=20., method='closest',
        window=1, load=True, presets=None, absolute=False, use_mags=True,
        use_central_z=True, zstep=0.1, return_evol=False):
        """
        Return volume density of galaxies in given `dz` chunk.

        .. note :: Just a wrapper around `get_surface_density`, with
            hack parameter `use_volume` set to True and `use_central_z` to
            True.


        """

        return self.get_surface_density(z, bins=bins, wave=wave,
            cam=cam, filters=filters, filter_set=filter_set, dlam=dlam,
            method=method, window=window, load=load, presets=presets,
            absolute=absolute, use_mags=use_mags, use_central_z=True,
            zstep=zstep, return_evol=return_evol, use_volume=True)

    def load(self):
        """
        Load results from past run.
        """

        fn_hist = self.pf['pop_histories']

        # Look for results attached to hmf table
        if fn_hist is None:
            prefix = self.guide.halos.tab_prefix_hmf(True)
            fn = self.guide.halos.tab_name

            suffix = fn[fn.rfind('.')+1:]
            path = ARES + '/input/hmf/'
            pref = prefix.replace('hmf', 'hgh')
            if self.pf['hgh_Mmax'] is not None:
                pref += '_xM_{:.0f}_{:.2f}'.format(self.pf['hgh_Mmax'],
                    self.pf['hgh_dlogM'])

            fn_hist = path + pref + '.' + suffix
        else:
            # Check to see if parameters match
            if self.pf['verbose']:
                print("Should check that HMF parameters match!")

        # Read output
        if type(fn_hist) is str:
            if fn_hist.endswith('.pkl'):
                f = open(fn_hist, 'rb')
                prefix = fn_hist.split('.pkl')[0]
                hist = pickle.load(f)
                f.close()
                if self.pf['verbose']:
                    print("# Loaded {}.".format(fn_hist.replace(ARES,
                        '$ARES')))

            elif fn_hist.endswith('.hdf5'):
                f = h5py.File(fn_hist, 'r')
                prefix = fn_hist.split('.hdf5')[0]

                if 'mask' in f:
                    mask = np.array(f[('mask')])
                else:
                    mask = np.zeros_like(f[('Mh')])

                hist = {}
                for key in f.keys():

                    if key not in ['cosmology', 't', 'z', 'children']:
                        #hist[key] = np.ma.array(f[(key)], mask=mask,
                        #    fill_value=-np.inf)

                        # Oddly, masking causes a weird issue with a huge
                        # spike at log10(MAR) ~ 1. np.ma operations are
                        # also considerably slower.
                        hist[key] = np.array(f[(key)]) * np.logical_not(mask)

                        #else:
                        #    hist[key] = np.ma.array(f[(key)], mask=mask)
                    else:
                        hist[key] = np.array(f[(key)])

                f.close()
                if self.pf['verbose']:
                    print("# Loaded {}.".format(fn_hist.replace(ARES, '$ARES')))

            else:
                # Assume pickle?
                f = open(fn_hist+'.pkl', 'rb')
                prefix = fn_hist
                hist = pickle.load(f)
                f.close()
                if self.pf['verbose']:
                    name = fn_hist + '.pkl'
                    print("# Loaded {}.".format(name.replace(ARES, '$ARES')))

                if self.pf['verbose']:
                    print("# Read `pop_histories` as dictionary")

            zall = hist['z']
            hist['zform'] = zall
            hist['zobs'] = np.array([zall] * hist['nh'].shape[0])

        elif type(self.pf['pop_histories']) is dict:
            hist = self.pf['pop_histories']
            # Assume you know what you're doing.
        elif type(self.pf['pop_histories']) is tuple:
            func, kw = self.pf['pop_histories']
            hist = func(**kw)
        else:
            hist = None

        return hist

    def save(self, prefix, clobber=False):
        """
        Output model (i.e., galaxy trajectories) to file.
        """

        fn = prefix + '.pkl'

        if os.path.exists(fn) and (not clobber):
            raise IOError('File \'{}\' exists! Set clobber=True to overwrite.'.format(fn))

        hist = self._gen_halo_histories()

        with open(fn, 'wb') as f:
            pickle.dump(hist, f)

        if self.pf['verbose']:
            print("Wrote {}.".format(fn))

        # Also save parameters.
        with open('{}.parameters.pkl'.format(prefix), 'wb') as f:
            pickle.dump(self.pf, f)

        if self.pf['verbose']:
            print("Wrote {}.parameters.pkl.".format(prefix))

    @property
    def dust(self):
        """
        (void) -> DustEmission

        Creates and / or returns an instance of a DustEmission object used to calculate
        dust emissions from all the galaxies at all redshifts for a given frequency band.

        To adjust the frequency band, set the "dust_fmin", "dust_fmax", and "dust_Nfreqs"
        keywords to the desired frequencies (in Hz) and number of frequencies between
        dust_fmin and dust_fmax to be probed.

        To adjust the band of redshifts, set the "dust_zmin", "dust_zmax", and "dust_Nz"
        keywords.
        """
        if not hasattr(self, "_dust"):

            # fetching keywords provided
            if self.pf.get('pop_dust_fmin') is None:
                fmin = 1e14
            else:
                fmin = self.pf['pop_dust_fmin']

            if self.pf.get('pop_dust_fmax') is None:
                fmax = 1e17
            else:
                fmax = self.pf['pop_dust_fmax']

            if self.pf.get('pop_dust_Nfreqs') is None:
                Nfreqs = 500
            else:
                Nfreqs = self.pf['pop_dust_Nfreqs']

            if self.pf.get('pop_dust_zmin') is None:
                zmin = 4
            else:
                zmin = self.pf['pop_dust_zmin']

            if self.pf.get('pop_dust_zmax') is None:
                zmax = 10
            else:
                zmax = self.pf['pop_dust_zmax']

            if self.pf.get('pop_dust_Nz') is None:
                Nz = 7
            else:
                Nz = self.pf['pop_dust_Nz']

            # creating instance
            self._dust = DustEmission(self, fmin, fmax, Nfreqs,
                zmin, zmax, Nz)

        return self._dust

    def dust_sed(self, fmin, fmax, Nfreqs):
        """
        (number, number, integer) -> 1darray, 3darray

        This is a wrapper for DustPopulation.dust_sed.

        -----------

        RETURNS

        frequencies, SED : 1darray, 3darray

        first axis: galaxy index
        second axis: frequency index
        third axis: redshift index

        -----------

        PARAMETERS

        fmin: number
            minimum frequency of the band

        fmax: number
            maximum frequency of the band

        Nfreqs: integer
            number of frequencies between fmin and fmax to be calculated
        """
        return self.dust.dust_sed(fmin, fmax, Nfreqs)
