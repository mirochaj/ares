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
    def synth(self):
        if not hasattr(self, '_synth'):
            self._synth = SpectralSynthesis(**self.pf)
            self._synth.src = self
            #self._synth._src_csfr = self._src_csfr
            self._synth.oversampling_enabled = self.pf['pop_ssp_oversample']
            self._synth.oversampling_below = self.pf['pop_ssp_oversample_age']
            self._synth.careful_cache = self.pf['pop_synth_cache_level']

        return self._synth

    def get_sfr(self, t, **kwargs):
        """
        Return the star formation rate at time (since Big Bang) `t` [in Myr].

        Parameters
        ----------
        t : int, float
        norm :
        """

        if self.pf['source_sfh'] == 'exp_decl':
            norm = kwargs['norm']
            tau = kwargs['tau']
            return norm * np.exp(-t / tau)
        else:
            raise NotImplemented('help')

    @property
    def _tau_guess(self):
        if not hasattr(self, '_tau_guess_'):
            self._t_in = 10**np.arange(0, 4.5, 0.1)
            self._sSFR_in = 10**np.arange(-11, -7, 0.1)
            self._tau_guess_ = np.zeros((self._t_in.size, self._sSFR_in.size))
            for i, _t_ in enumerate(self._t_in):
                sSFR = lambda logtau: 1. \
                    / (10**logtau * (np.exp(_t_ / 10**logtau) - 1.))
                for j, _sSFR_ in enumerate(self._sSFR_in):
                    func = lambda logtau: np.abs(np.log10(sSFR(logtau) / _sSFR_))
                    self._tau_guess[i,j] = fmin(func, 3., disp=False)[0]

        return self._tau_guess_

    def get_kwargs(self, t, mass, sfr, disp=False):
        """
        Determine the free parameters of a model needed to produce stellar mass
        `mass` and star formation rate `sfr` at `t` [since Big Bang / Myr].
        """

        if self.pf['source_sfh'] == 'exp_decl':
            # For this model, the sSFR uniquely determines tau. Just need to
            # solve for it numerically. Put factor of 10^-6 out front to make
            # sure that the units match input [Msun / yr], despite tau being
            # defined in Myr.
            f_sSFR = lambda logtau: 1e-6 / (10**logtau * (np.exp(t / 10**logtau) - 1.))
            func = lambda logtau: np.abs(np.log10(f_sSFR(logtau) / (sfr / mass)))
            tau = 10**fmin(func, 2., disp=disp, full_output=disp,
                ftol=0.01, xtol=0.01)[0]

            # Can analytically solve for normalization once tau in hand.
            norm = sfr / np.exp(-t / tau)

            kw = {'norm': norm, 'tau': tau}
        else:
            raise NotImplemented('help')

        return kw

    def get_sfh(self, t, mass, sfr, **kwargs):
        """
        Return the SFR over all times, provided a boundary condition on the
        stellar mass and SFR at some time `t`.
        """

        if self.pf['source_sfh'] == 'exp_decl':
            kw = self.get_kwargs(t, mass, sfr)
            return self.get_sfr(self.tab_t_pop, **kw)
        else:
            raise NotImplemented('help')

    def get_mass(self, t, **kwargs):
        """
        Return stellar mass for a given SFH model, integrate analytically
        if possible.
        """

        if self.pf['source_sfh'] == 'exp_decl':
            tau = kwargs['tau']
            norm = kwargs['norm']

            # Factor of 1e6 is to convert tau/Myr -> years
            return norm * 1e6 * tau * (1. - np.exp(-t / tau))
        else:
            raise NotImplemented('help')

    def get_spec(self, zobs, t=None, mass=None, sfr=None, waves=None, **kwargs):
        """
        Return the rest-frame spectrum of a galaxy at given t, that has
        stellar mass `mass` and SFR `sfr`.

        Parameters
        ----------
        t : np.ndarray
            Array of times in Myr since Big Bang.
        """

        if waves is None:
            waves = self.tab_waves_c

        if kwargs == {}:
            sfh = self.get_sfh(t, mass, sfr)
        else:
            assert (t is not None) and (mass is not None) and (sfr is not None), \
                "Must provide kwargs or (t, mass, sfr)"
            sfh = self.get_sfr(self.tab_t_pop, **kwargs)

        tasc = self.tab_t_pop[-1::-1]
        sfh_asc = sfh[-1::-1]

        spec = self.synth.get_spec_rest(sfh=sfh_asc, tarr=tasc,
            waves=waves, zobs=zobs, load=False)
                #hist={'SFR': sfh, 't': t})

        return spec

    def get_spec_obs(self):
        pass

    def get_mags(self):
        pass

    def get_lum_per_sfr(self):
        pass

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

        degL = self.pf['pop_sed_degrade']
        waves = self.tab_waves_c[::degL]

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
