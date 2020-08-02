"""

Global21cm.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 14:55:35 MDT 2014

Description:

"""

from __future__ import print_function

import os
import time
import numpy as np
from types import FunctionType
from ..util.Math import interp1d
from ..util.PrintInfo import print_sim
from ..util.ReadData import _sort_history
from ..util.Pickling import write_pickle_file
from ..util import ParameterFile, ProgressBar, get_hash
from ..analysis.Global21cm import Global21cm as AnalyzeGlobal21cm
from ..physics.Constants import nu_0_mhz, E_LyA, h_p, erg_per_ev, k_B, c

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

class Global21cm(AnalyzeGlobal21cm):
    def __init__(self, **kwargs):
        """
        Set up a two-zone model for the global 21-cm signal.

        ..note :: This is essentially a MultiPhaseMedium calculation, except
            the Lyman alpha background and 21-cm background are calculated,
            and alternative (phenomenological) parameterizations such as a
            tanh for the ionization, thermal, and LW background evolution,
            may be used.

        """

        self.is_complete = False

        # See if this is a tanh model calculation
        is_phenom = self.is_phenom = self._check_if_phenom(**kwargs)

        if 'problem_type' not in kwargs:
            kwargs['problem_type'] = 101

        self.kwargs = kwargs

        # Print info to screen
        if self.pf['verbose']:
            print_sim(self)

    #def __del__(self):
    #    print("Killing it! Processor={}".format(rank))

    @property
    def timer(self):
        if not hasattr(self, '_timer'):
            self._timer = 0.0
        return self._timer

    @timer.setter
    def timer(self, value):
        self._timer = value

    @property
    def count(self):
        if not hasattr(self, '_count'):
            try:
                self._count = self.medium.field.count
            except AttributeError:
                self._count = 1
        return self._count

    @property
    def info(self):
        print_sim(self)

    @property
    def pf(self):
        if not hasattr(self, '_pf'):
            self._pf = ParameterFile(**self.kwargs)
        return self._pf

    @pf.setter
    def pf(self, value):
        self._pf = value

    @property
    def medium(self):
        if not hasattr(self, '_medium'):
            from .MultiPhaseMedium import MultiPhaseMedium
            self._medium = MultiPhaseMedium(cosm=self.cosm, **self.kwargs)

        return self._medium

    @property
    def field(self):
        if not hasattr(self, '_field'):
            self._field = self.medium.field
        return self._field

    @property
    def pops(self):
        return self.medium.field.solver.pops

    @property
    def grid(self):
        return self.medium.field.grid

    def _init_dTb(self):
        """
        Compute differential brightness temperature for initial conditions.
        """
        z = self.all_z

        dTb = []
        for i, data_igm in enumerate(self.all_data_igm):

            n_H = self.medium.parcel_igm.grid.cosm.nH(z[i])

            Ts = \
                self.medium.parcel_igm.grid.hydr.Ts(
                    z[i], data_igm['Tk'], 0.0, data_igm['h_2'],
                    data_igm['e'] * n_H)

            # Compute volume-averaged ionized fraction
            if self.pf['include_cgm']:
                QHII = self.all_data_cgm[i]['h_2']
            else:
                QHII = 0.0

            xavg = QHII + (1. - QHII) * data_igm['h_2']

            # Derive brightness temperature
            Tb = self.medium.parcel_igm.grid.hydr.dTb(z[i], xavg, Ts)
            self.all_data_igm[i]['dTb'] = Tb
            self.all_data_igm[i]['Ts'] = np.array([Ts])
            dTb.append(Tb)

        return dTb

    def _check_if_phenom(self, **kwargs):
        if not kwargs:
            return False

        if ('tanh_model' not in kwargs) and ('gaussian_model' not in kwargs)\
           and ('parametric_model' not in kwargs):
            return False

        self.is_tanh = False
        self.is_gauss = False
        self.is_param = False

        if 'tanh_model' in kwargs:
            if kwargs['tanh_model']:
                from ..phenom.Tanh21cm import Tanh21cm as PhenomModel
                self.is_tanh = True

        elif 'gaussian_model' in kwargs:
            if kwargs['gaussian_model']:
                from ..phenom.Gaussian21cm import Gaussian21cm as PhenomModel
                self.is_gauss = True
        elif 'parametric_model' in kwargs:
            if kwargs['parametric_model']:
                from ..phenom.Parametric21cm import Parametric21cm as PhenomModel
                self.is_param = True

        if (not self.is_tanh) and (not self.is_gauss) and (not self.is_param):
            return False

        model = self._model = PhenomModel(**kwargs)
        self.pf = model.pf

        if self.pf['output_frequencies'] is not None:
            nu = self.pf['output_frequencies']
            z = nu_0_mhz / nu - 1.
        elif self.pf['output_redshifts'] is not None:
            z = self.pf['output_redshifts']
            nu = nu_0_mhz / (1. + z)
        elif self.pf['output_dz'] is not None:
            z = np.arange(self.pf['final_redshift'] + self.pf['output_dz'],
                self.pf['initial_redshift'], self.pf['output_dz'])[-1::-1]
            nu =  nu_0_mhz / (1. + z)
        else:
            nu_min = self.pf['output_freq_min']
            nu_max = self.pf['output_freq_max']
            nu_res = self.pf['output_freq_res']

            nu = np.arange(nu_min, nu_max, nu_res)
            z = nu_0_mhz / nu - 1.

        if self.is_param:
            self.history = model(z)
        elif self.is_gauss:
            self.history = model(nu, **self.pf)
        else:
            self.history = model(z, **self.pf)

        return True

    def run(self):
        """
        Run a 21-cm simulation.

        Returns
        -------
        Nothing: sets `history` attribute.

        """

        # If this was a tanh model or some such thing, we're already done.
        if self.is_phenom:
            return
        if self.is_complete:
            print("Already ran simulation!")
            return

        # Need to generate radiation backgrounds first.
        if self.pf['radiative_transfer']:
            self.medium.field.run()
            self._f_Ja  = self.medium.field._f_Ja
            self._f_Jlw = self.medium.field._f_Jlw
        else:
            self._f_Ja  = lambda z: 0.0
            self._f_Jlw = lambda z: 0.0

        # Start timer
        t1 = time.time()

        tf = self.medium.tf
        self.medium._insert_inits()

        pb = self.pb = ProgressBar(tf, use=self.pf['progress_bar'],
            name='gs-21cm')

        # Lists for data in general
        self.all_t, self.all_z, self.all_data_igm, self.all_data_cgm, \
            self.all_RC_igm, self.all_RC_cgm = \
            self.medium.all_t, self.medium.all_z, self.medium.all_data_igm, \
            self.medium.all_data_cgm, self.medium.all_RCs_igm, self.medium.all_RCs_cgm

        # Add zeros for Ja
        for element in self.all_data_igm:
            element['Ja'] = 0.0
            element['Jlw'] = 0.0

        # List for extrema-finding
        self.all_dTb = self._init_dTb()
        for t, z, data_igm, data_cgm, rc_igm, rc_cgm in self.step():

            # Occasionally the progress bar breaks if we're not careful
            if z < self.pf['final_redshift']:
                break
            if z < self.pf['kill_redshift']:
                break

            # Delaying the initialization prevents progressbar from being
            # interrupted by, e.g., PrintInfo calls
            if not pb.has_pb:
                pb.start()

            pb.update(t)

            # Save data
            self.all_z.append(z)
            self.all_t.append(t)
            self.all_dTb.append(data_igm['dTb'][0])
            self.all_data_igm.append(data_igm.copy())
            self.all_RC_igm.append(rc_igm.copy())

            if self.pf['include_cgm']:
                self.all_data_cgm.append(data_cgm.copy())
                self.all_RC_cgm.append(rc_cgm.copy())

            # Automatically find turning points
            if self.pf['track_extrema']:
                if self.track.is_stopping_point(self.all_z, self.all_dTb):
                    break

        pb.finish()

        self.history_igm = _sort_history(self.all_data_igm, prefix='igm_',
            squeeze=True)

        if self.pf['include_cgm']:
            self.history_cgm = _sort_history(self.all_data_cgm, prefix='cgm_',
                squeeze=True)
        else:
            self.history_cgm = {}

        self.history = self.history_igm.copy()
        self.history.update(self.history_cgm)

        ##
        # In the future, could do this better by only calculating Ja at
        # the end, since it a passive quantity (unless we included its
        # very small heating).
        ##
        #if self.pf['secondary_lya']:
        #    xe = lambda zz: np.interp(zz, self.history['z'][-1::-1],
        #        self.history['igm_e'][-1::-1])
        #    self.medium.field.run(xe=xe)
        #    self._f_Ja = self.medium.field._f_Ja
        #    #self._f_Jlw = self.medium.field._f_Jlw
        #
        #    # Fix Ja in history

        self.history['dTb'] = self.history['igm_dTb']
        #self.history['dTb_bulk'] = self.history['igm_dTb_bulk']

        self.history['Ts'] = self.history['igm_Ts']
        self.history['Ja'] = self.history['igm_Ja']
        self.history['Jlw'] = self.history['igm_Jlw']

        # Save rate coefficients [optional]
        if self.pf['save_rate_coefficients']:
            self.rates_igm = \
                _sort_history(self.all_RC_igm, prefix='igm_', squeeze=True)
            self.rates_cgm = \
                _sort_history(self.all_RC_cgm, prefix='cgm_', squeeze=True)

            self.history.update(self.rates_igm)
            self.history.update(self.rates_cgm)

        self.history['t'] = np.array(self.all_t)
        self.history['z'] = np.array(self.all_z)

        ##
        # Optional extra radio background
        ##
        Tr = np.zeros_like(self.history['z'])
        for popid, pop in enumerate(self.pops):
            if not pop.is_src_radio:
                continue

            z, E, flux = self.field.get_history(popid, flatten=True)

            E21cm = h_p * nu_0_mhz * 1e6 / erg_per_ev
            f21 = interp1d(E, flux, axis=1, bounds_error=False,
                fill_value=0.0, force_scipy=True)
            flux_21cm = f21(E21cm)

            Tr += np.interp(self.history['z'], z, flux_21cm) \
                * E21cm * erg_per_ev * c**2 / k_B / 2. / (nu_0_mhz * 1e6)**2

        if not np.all(Tr == 0):
            assert self.medium.parcel_igm.grid.hydr.Tbg is None
        elif self.medium.parcel_igm.grid.hydr.Tbg is not None:
            Tr = self.medium.parcel_igm.grid.hydr.Tbg(self.history['z'])

        self.history['Tr'] = Tr

        # Correct the brightness temperature if there are non-CMB backgrounds
        if not np.all(Tr == 0):
            zall = self.history['z']
            n_H = self.medium.parcel_igm.grid.cosm.nH(zall)
            Ts = self.medium.parcel_igm.grid.hydr.Ts(zall,
                self.history['igm_Tk'], self.history['Ja'],
                self.history['igm_h_2'], self.history['igm_e'] * n_H, Tr)

            if self.pf['floor_Ts']:
                Ts = max(Ts, self.medium.parcel_igm.grid.hydr.Ts_floor(z=zall))

            # Compute volume-averaged ionized fraction
            xavg = self.history['cgm_h_2'] \
                 + (1. - self.history['cgm_h_2']) * self.history['igm_h_2']

            # Derive brightness temperature
            dTb = self.medium.parcel_igm.grid.hydr.dTb(zall, xavg, Ts, Tr)

            self.history['dTb_no_radio'] = self.history['dTb'].copy()
            self.history['dTb'] = dTb

            #self.history['dTb_bulk'] = \
            #   self.medium.parcel_igm.grid.hydr.dTb(zall, 0.0, Ts, Tr)

        t2 = time.time()

        self.timer = t2 - t1

        self.is_complete = True

    def step(self):
        """
        Generator for the 21-cm signal.

        .. note:: Basically just calling MultiPhaseMedium here, except we
            compute the spin temperature and brightness temperature on
            each step.

        Returns
        -------
        Generator for MultiPhaseMedium object, with notable addition that
        the spin temperature and 21-cm brightness temperature are now
        tracked.

        """

        for t, z, data_igm, data_cgm, RC_igm, RC_cgm in self.medium.step():

            Ja = np.atleast_1d(self._f_Ja(z))
            Jlw = np.atleast_1d(self._f_Jlw(z))

            # Compute spin temperature
            n_H = self.medium.parcel_igm.grid.cosm.nH(z)
            Ts = self.medium.parcel_igm.grid.hydr.Ts(z,
                data_igm['Tk'], Ja, data_igm['h_2'], data_igm['e'] * n_H)

            if self.pf['floor_Ts'] is not None:
                Ts = max(Ts, self.medium.parcel_igm.grid.hydr.Ts_floor(z=z))

            # Compute volume-averaged ionized fraction
            if self.pf['include_cgm']:
                xavg = data_cgm['h_2'] + (1. - data_cgm['h_2']) * data_igm['h_2']
            else:
                xavg = data_igm['h_2']

            # Derive brightness temperature
            dTb = self.medium.parcel_igm.grid.hydr.dTb(z, xavg, Ts)
            dTb_b = self.medium.parcel_igm.grid.hydr.dTb(z, 0.0, Ts)

            # Add derived fields to data
            data_igm.update({'Ts': Ts, 'dTb': dTb, #'dTb_bulk': dTb_b,
                'Ja': Ja, 'Jlw': Jlw})

            # Yield!
            yield t, z, data_igm, data_cgm, RC_igm, RC_cgm

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

        fn = '{0!s}.history.{1!s}'.format(prefix, suffix)

        if os.path.exists(fn):
            if clobber:
                os.remove(fn)
            else:
                raise IOError('{!s} exists! Set clobber=True to overwrite.'.format(fn))

        if suffix == 'pkl':
            write_pickle_file(self.history._data, fn, ndumps=1, open_mode='w',\
                safe_mode=False, verbose=False)

            try:
                write_pickle_file(self.blobs, '{0!s}.blobs.{1!s}'.format(\
                    prefix, suffix), ndumps=1, open_mode='w', safe_mode=False,\
                    verbose=self.pf['verbose'])
            except AttributeError:
                print('Error writing {0!s}.blobs.{1!s}'.format(prefix, suffix))

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
            print("#", end='', file=f)

            for key in self.history:
                if fields is not None:
                    if key not in fields:
                        continue
                print('{0:<18s}'.format(key), end='', file=f)
            print('', file=f)

            # Now, the data
            for i in range(len(self.history[key])):
                s = ''

                for key in self.history:
                    if fields is not None:
                        if key not in fields:
                            continue

                    s += '{:<20.8e}'.format(self.history[key][i])

                if not s.strip():
                    continue

                print(s, file=f)

            f.close()

        if self.pf['verbose']:
            print('Wrote {0!s}.history.{1!s}'.format(prefix, suffix))

        # Save histories for Mmin and SFRD if we're doing iterative stuff
        if self.count > 1 and hasattr(self, '_Mmin_bank'):
            write_pickle_file((self.medium.field._zarr,\
                self.medium.field._Mmin_bank), '{!s}.Mmin.pkl'.format(prefix),\
                ndumps=2, open_mode='w', safe_mode=False,\
                verbose=self.pf['verbose'])

            if self.pf['feedback_LW_sfrd_popid'] is not None:
                pid = self.pf['feedback_LW_sfrd_popid']
                write_pickle_file((self.medium.field.pops[pid].halos.tab_z,\
                    self.medium.field._sfrd_bank), '{!s}.sfrd.pkl'.format(\
                    prefix), ndumps=1, open_mode='w', safe_mode=False,\
                    verbose=self.pf['verbose'])

        write_pf = True
        if os.path.exists('{!s}.parameters.pkl'.format(prefix)):
            if clobber:
                os.remove('{!s}.parameters.pkl'.format(prefix))
            else:
                write_pf = False
                print(('WARNING: {!s}.parameters.pkl exists! Set ' +\
                    'clobber=True to overwrite.').format(prefix))

        if write_pf:

            #pf = {}
            #for key in self.pf:
            #    if key in self.carryover_kwargs():
            #        continue
            #    pf[key] = self.pf[key]

            if 'revision' not in self.pf:
                self.pf['revision'] = get_hash()

            # Save parameter file
            write_pickle_file(self.pf, '{!s}.parameters.pkl'.format(prefix),\
                ndumps=1, open_mode='w', safe_mode=False,\
                verbose=self.pf['verbose'])
