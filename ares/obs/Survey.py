"""

Survey.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Sat 16 Feb 2019 10:40:18 EST

Description:

"""

import re
import os
import copy
import numpy as np
from ..data import ARES
from ..physics.Constants import c
from ..physics.Cosmology import Cosmology

try:
    import ares
except ImportError:
    pass

flux_AB = 3631. * 1e-23 # 3631 * 1e-23 erg / s / cm**2 / Hz
nanoJ = 1e-23 * 1e-9

_path = ARES + '/input'

class Survey(object):
    def __init__(self, cam='nircam', mod='modA', chip=1, force_perfect=False,
        cache={}):
        self.camera = cam
        self.chip = chip
        self.force_perfect = force_perfect
        self.cache = cache
        self.mod = mod

    def PlotFilters(self, ax=None, fig=1, filter_set='W',
        filters=None, annotate=True, annotate_kw={}, skip=None, rotation=90,
        **kwargs): # pragma: no cover
        """
        Plot transmission curves for filters.
        """

        import matplotlib.pyplot as pl

        if ax is None:
            fig = pl.figure(fig, figsize=(6, 6))
            ax = fig.add_subplot(111)
            gotax = False
        else:
            gotax = True

        data = self.read_throughputs(filter_set, filters)

        colors = ['k', 'b', 'g', 'c', 'm', 'y', 'r', 'orange'] * 10
        for i, filt in enumerate(data.keys()):

            if skip is not None:
                if filt in skip:
                    continue

            if filters is not None:
                if filt not in filters:
                    continue

            if kwargs != {}:
                if 'color' in kwargs:
                    c = kwargs['color']
                    del kwargs['color']
            else:
                c = colors[i]

            ax.plot(data[filt][0], data[filt][1], label=filt, color=c,
                **kwargs)

            if annotate:
                if filt.endswith('IR'):
                    _filt = filt[0:-3]
                else:
                    _filt = filt

                ax.annotate(_filt, (data[filt][2], 1), ha='center', va='top',
                    color=c, rotation=rotation, **annotate_kw)

        ax.set_xlabel(r'Observed Wavelength $[\mu \mathrm{m}]$')
        ax.set_ylabel('Transmission')
        ax.set_ylim(-0.05, 1.05)
        #ax.legend(loc='best', frameon=False, fontsize=10, ncol=2)

        return ax

    def read_throughputs(self, filter_set='W', filters=None):
        """
        Assembles a dictionary of throughput curves.

        Each element of the dictionary is a tuple containing:
            (wavelength, throughput, midpoint of filter,
                width of filter (FWHM), transmission averaged over filter)

        Example
        -------

        >>> wave, T, mid, wid, Tavg = self.read_throughputs()


        """
        if ((self.camera, None, 'all') in self.cache) and (filters is not None):
            cached_phot = self.cache[(self.camera, None, 'all')]

            # Just grab the filters that were requested!
            result = {}
            for filt in filters:
                if filt not in cached_phot:
                    continue
                result[filt] = cached_phot[filt]

            return result

        if self.camera == 'nircam':
            return self._read_nircam(filter_set, filters)
        elif self.camera in ['hst', 'hubble']:
            wfc = self._read_wfc(filter_set, filters)
            wfc3 = self._read_wfc3(filter_set, filters)
            hst = wfc.copy()
            hst.update(wfc3)
            return hst
        elif self.camera == 'wfc3':
            return self._read_wfc3(filter_set, filters)
        elif self.camera == 'wfc':
            return self._read_wfc(filter_set, filters)
        elif self.camera == 'irac':
            return self._read_irac(filter_set, filters)
        elif self.camera == 'roman':
            return self._read_roman(filters)
        else:
            raise NotImplemented('help')

    def _read_nircam(self, filter_set='W', filters=None): # pragma: no cover

        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        get_all = False
        if filters is not None:
            if filters == 'all':
                get_all = True
            else:
                assert type(filters) in [list, tuple]
        else:
            filters = []
            if type(filter_set) != list:
                filter_set = [filter_set]

        path = '{}/nircam/nircam_throughputs/{}/filters_only'.format(_path,
            self.mod)

        data = {}
        for fn in os.listdir(path):

            pre = fn.split('_')[0]

            if get_all or (pre in filters):

                if pre in self._filter_cache:
                    data[pre] = self._filter_cache[pre]
                    continue

                if ('W2' in pre):
                    continue

                num = re.findall(r'\d+', pre)[0]
                cent = float('{}.{}'.format(num[0], num[1:]))

                # Wavelength [micron], transmission
                x, y = np.loadtxt('{}/{}'.format(path, fn), unpack=True,
                    skiprows=1)

                data[pre] = self._get_filter_prop(x, y, cent)

                self._filter_cache[pre] = copy.deepcopy(data[pre])

            elif filter_set is not None:

                for _filters in filter_set:

                    if _filters in self._filter_cache:
                        data[pre] = self._filter_cache[_filters]
                        continue

                    if _filters not in pre:
                        continue

                    # Need to distinguish W from W2
                    if (_filters == 'W') and ('W2' in pre):
                        continue

                    # Determine the center wavelength of the filter based its string
                    # identifier.
                    k = pre.rfind(_filters)
                    cent = float('{}.{}'.format(pre[1], pre[2:k]))

                    # Wavelength [micron], transmission
                    x, y = np.loadtxt('{}/{}'.format(path, fn), unpack=True,
                        skiprows=1)

                    data[pre] = self._get_filter_prop(x, y, cent)

                    self._filter_cache[pre] = copy.deepcopy(data[pre])

        return data

    def _read_wfc(self, filter_set='W', filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        get_all = False
        if filters is not None:
            if filters == 'all':
                get_all = True
            else:
                assert type(filters) in [list, tuple]
        else:
            filters = []
            # Grab all 'W' or 'N' etc. filters
            if type(filter_set) != list:
                filter_set = [filter_set]

        path = '{}/wfc'.format(_path)

        data = {}
        for fn in os.listdir(path):

            # Mac OS creates a bunch of ._wfc_* files. Argh.
            if not fn.startswith('wfc_'):
                continue

            if fn.endswith('tar.gz'):
                continue

            pre = fn.split('wfc_')[1].split('.dat')[0]

            if get_all or (pre in filters):

                if pre in self._filter_cache:
                    data[pre] = self._filter_cache[pre]
                    continue

                cent = float('0.{}'.format(pre[1:4]))

                x, y = np.loadtxt('{}/{}'.format(path, fn),
                    unpack=True, skiprows=1)

                # Convert wavelengths from nanometers to microns
                data[pre] = self._get_filter_prop(x / 1e4, y, cent)

                self._filter_cache[pre] = copy.deepcopy(data[pre])

            elif filter_set is not None:
                for _filters in filter_set:

                    if _filters in self._filter_cache:
                        data[pre] = self._filter_cache[_filters]
                        continue

                    if _filters not in pre:
                        continue

                    # Determine the center wavelength of the filter based on its string
                    # identifier.
                    k = pre.rfind(_filters)
                    cent = float('0.{}'.format(pre[1:k]))

                    x, y = np.loadtxt('{}/{}'.format(path, fn),
                        unpack=True, skiprows=1)

                    # Convert wavelengths from nanometers to microns
                    data[pre] = self._get_filter_prop(x / 1e4, y, cent)

                    self._filter_cache[pre] = copy.deepcopy(data[pre])

        return data

    def _read_wfc3(self, filter_set='W', filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        get_all = False
        if filters is not None:
            if filters == 'all':
                get_all = True
            else:
                assert type(filters) in [list, tuple]
        else:
            filters = []
            if type(filter_set) != list:
                filter_set = [filter_set]

        path = path = '{}/wfc3'.format(_path)

        data = {}
        for fn in os.listdir(path):

            if '.txt' not in fn:
                continue

            pre = fn[fn.find('_f')+1:fn.rfind('.')].upper()

            # Read-in no matter what
            if get_all or (pre in filters):

                if pre in self._filter_cache:
                    data[pre] = self._filter_cache[pre]
                    continue

                cent = float('{}.{}'.format(pre[1], pre[2:-1]))

                x, y = np.loadtxt('{}/{}'.format(path, fn),
                    unpack=True, skiprows=1)

                # Convert wavelengths from Angstroms to microns
                data[pre] = self._get_filter_prop(x / 1e4, y, cent)

                self._filter_cache[pre] = copy.deepcopy(data[pre])


            elif filter_set is not None:


                for _filters in filter_set:

                    if _filters in self._filter_cache:
                        data[pre] = self._filter_cache[_filters]
                        continue

                    if _filters not in pre:
                        continue

                    # Determine the center wavelength of the filter based on its
                    # string identifier.
                    cent = float('{}.{}'.format(pre[1], pre[2:-1]))

                    x, y = np.loadtxt('{}/{}'.format(path, fn),
                        unpack=True, skiprows=1)

                    # Convert wavelengths from Angstroms to microns
                    data[pre] = self._get_filter_prop(x / 1e4, y, cent)

                    self._filter_cache[pre] = copy.deepcopy(data[pre])

        return data

    def _read_irac(self, filter_set='W', filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        path = '{}/irac'.format(_path)

        data = {}
        for fn in os.listdir(path):
            x, y = np.loadtxt('{}/{}'.format(path, fn), unpack=True,
                skiprows=1)

            if 'ch1' in fn:
                cent = 3.6
                pre = 'ch1'
            elif 'ch2' in fn:
                cent = 4.5
                pre = 'ch2'
            else:
                raise ValueError('Unrecognized IRAC file: {}'.format(fn))

            data[pre] = self._get_filter_prop(x, y, cent)

            self._filter_cache[pre] = copy.deepcopy(data[pre])

        return data

    def _read_roman(self, filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Need pandas to read Roman ST throughputs.")

        _fn = 'Roman_effarea_20201130.xlsx'

        A = np.pi * (0.5 * 2.4)**2

        path = '{}/roman'.format(_path)

        data = {}
        for fn in os.listdir(path):

            if fn != _fn:
                continue

            df = pd.read_excel(path + '/' + _fn,
                sheet_name='Roman_effarea_20201130',
                header=1)

            _cols = df.columns
            cols = [col.strip() for col in _cols]

            x = df['Wave'].to_numpy()

            for col in cols:
                if col[0] != 'F':
                    continue

                pre = col
                cent = float(col[1] + '.' + col[2:])

                # This is an effective area. Take T = A_eff / (pi * 1.2**2)
                y = df[' '+col].to_numpy() / A
                y[x > 2] = 0 # spurious spike at ~2.6 microns

                data[pre] = self._get_filter_prop(np.array(x), np.array(y), cent)

                self._filter_cache[pre] = copy.deepcopy(data[pre])

            break

        return data

    def _get_filter_prop(self, x, y, cent):
        Tmax = max(y)
        _ok = y > 1e-2 #y > 1e-2 * y.max()

        # Find non-contiguous regions (NIRCAM F090W only?)
        # This is a kludgey fix
        i = np.arange(0, x.size)

        bpts = np.where(np.diff(i[_ok==1]) != 1)[0]
        chunks = np.split(i[_ok==1], bpts+1)
        if len(chunks) == 1:
            ok = _ok
        else:
            # Each chunk is a list of contiguous indices
            for chunk in chunks:
                lo, hi = chunk[0], chunk[-1]
                if not (x[lo] <= cent <= hi):
                    continue
                break

            ok = np.zeros_like(y)
            ok[chunk] = 1

        # Compute width of filter
        hi = max(x[ok == True])
        lo = min(x[ok == True])

        # Average T-weighted wavelength in filter.
        #mi = np.sum(x[ok==True] * y[ok==True]) / np.sum(y[ok==True])

        # Why is this sometimes not very close to `cent`?
        mi = np.mean(x[ok==True])
        dx = np.array([hi - mi, mi - lo])
        Tavg = np.sum(y[ok==1]) / float(y[ok==1].size)

        if self.force_perfect:
            Tavg = 1.
            dx = np.array([0.1] * 2)
            ok = np.logical_and(x >= cent-dx[1], x <= cent+dx[0])
            y[ok==1] = Tavg
            y[~ok] = 0.0

        return x, y, mi, dx, Tavg

    def get_dropout_filter(self, z, filters=None, drop_wave=1216., skip=None):
        """
        Determine where the Lyman break happens and return the corresponding
        filter.
        """

        data = self.read_throughputs(filters='all')

        wave_obs = drop_wave * 1e-4 * (1. + z)

        if filters is not None:
            all_filts = filters
        else:
            all_filts = list(data.keys())

        if skip is not None:
            if type(skip) not in [list, tuple]:
                skip = [skip]

            for element in skip:
                all_filts.remove(element)

        # Make sure filters are in order of ascending wavelength
        x0 = [data[filt][2] for filt in all_filts]
        sorter = np.argsort(x0)
        _all_filts = [all_filts[i] for i in sorter]
        all_filts = _all_filts

        gotit = False
        for j, filt in enumerate(all_filts):

            x0 = data[filt][2]
            p, m = data[filt][3]

            in_filter = (x0 - m <= wave_obs <= x0 + p)

            # Check for exclusivity
            if j >= 1:
                x0b = data[all_filts[j-1]][2]
                pb, mb = data[all_filts[j-1]][3]

                in_blue_neighbor = (x0b - mb <= wave_obs <= x0b + pb)
            else:
                in_blue_neighbor = False

            if j < len(all_filts) - 1:
                x0r = data[all_filts[j+1]][2]
                pr, mr = data[all_filts[j+1]][3]

                in_red_neighbor = (x0r - mr <= wave_obs <= x0r + pr)
            else:
                in_red_neighbor = False

            # Final say
            if in_filter: #and (not in_blue_neighbor) and (not in_red_neighbor):
                gotit = True
                break

        if gotit:
            drop = filt
            if filt != all_filts[-1]:
                drop_redder = all_filts[j+1]
            else:
                drop_redder = None
        else:
            drop = drop_redder = None

        return drop, drop_redder
