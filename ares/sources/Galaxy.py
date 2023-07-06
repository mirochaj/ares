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
from ..util import ProgressBar
from scipy.optimize import fmin
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

    def get_kwargs(self, t, mass, sfr):
        """
        Determine the free parameters of a model needed to produce stellar mass
        `mass` and star formation rate `sfr` at `t` [since Big Bang / Myr].
        """

        if self.pf['source_sfh'] == 'exp_decl':
            # Almost analytic here
            sSFR = lambda logtau: 1. / (10**logtau * (np.exp(t / 10**logtau) - 1.))
            func = lambda tau: np.abs(np.log10(sSFR(tau) / (sfr / mass)))
            tau = 10**fmin(func, 3., disp=False)[0]
            norm = mass / tau / (1 - np.exp(-t / tau)) / 1e6
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

            # Factor of 1e6 is to convert Myr -> years
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

    def generate_sed_tables(self, use_pbar=True):
        """
        Create a lookup table for the SED given this SFH model.
        """

        fn = self.get_tab_fn()

        if not os.path.exists(fn[0:fn.rfind('/')]):
            os.mkdir(fn[0:fn.rfind('/')])

        axes = self.pf['source_sfh_axes']
        axes_names = [ax[0] for ax in axes]
        axes_vals =  [ax[1] for ax in axes]

        ##
        # Look for matching file
        if os.path.exists(fn):
            with h5py.File(fn, 'r') as f:
                sed_tab = np.array(f[('seds')])

                # Check axes
                axes_vals = np.array(f[('axes_vals')])
                axes_names = list(f[('axes_names')])

            print(f"# Loaded {fn}.")
            return sed_tab

        elif rank == 0:
            print(f"# Did not find {fn}. Will generate from scratch.")

        ##
        # If we didn't find one, generate from scratch
        shape = np.array([vals.size for vals in axes_vals])
        axes_ind  =  [range(shape[i]) for i in range(len(axes))]
        ndim = len(axes_names)
        size = np.product(shape)

        combos = itertools.product(*axes_vals)
        coords = itertools.product(*axes_ind)

        kwargs_flat = []
        for combo in combos:
            tmp = {axes_names[i]:combo[i] for i in range(ndim)}
            kwargs_flat.append(tmp)

        # Create an N+2 dimension SED lookup table, where N is the number of
        # free parameters associated with the SFH.
        tasc = self.tab_t_pop[-1::-1]
        zdes = self.tab_z_pop[-1::-1]

        waves = self.tab_waves_c

        ##
        # If we didn't find a matching file, make a new one
        if rank == 0:
            print(f"# Will save SED table to {fn}.")

        pb = ProgressBar(len(kwargs_flat) * tasc.size, name='seds|sfhs',
            use=self.pf['progress_bar'] and use_pbar)
        pb.start()

        data = np.zeros([waves.size, tasc.size] + list(shape))

        for i, ind in enumerate(coords):

            #ind = coords[i]
            kw = kwargs_flat[i]

            for j, _t_ in enumerate(tasc):

                k = i * len(kwargs_flat) + j

                if k % size != rank:
                    continue

                pb.update(k)

                zobs = zdes[j]

                if zobs > 0.5:
                    continue

                if zobs < 0.4:
                    continue

                print('hey', zobs)

                sfh_asc = self.get_sfr(tasc, **kw)
                spec = self.synth.get_spec_rest(waves,
                    sfh=sfh_asc, tarr=tasc, zobs=zobs, load=False, use_pbar=False)

                s = slice(None),j, *list(ind)
                data[s] = spec

        pb.finish()

        if size > 1: # pragma: no cover
            sed_tab = np.zeros_like(data)
            nothing = MPI.COMM_WORLD.Allreduce(data, sed_tab)
        else:
            sed_tab = data

        ##
        # Save results
        if rank > 0:
            sys.exit(0)

        with h5py.File(fn, 'w') as f:
            f.create_dataset('seds', data=sed_tab)
            f.create_dataset('axes_names', data=axes_names)
            f.create_dataset('axes_vals', data=axes_vals)

        print(f"# Wrote {fn}.")

        return sed_tab


    def tab_sed_synth(self):
        pass
