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

_path = ARES

class Survey(object):
    def __init__(self, cam='nircam', mod='modA', chip=1, force_perfect=False,
        cache={}):
        self.camera = cam
        self.chip = chip
        self.force_perfect = force_perfect
        self.mod = mod

    def PlotFilters(self, ax=None, fig=1,
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

        data = self.read_throughputs(filters)

        colors = ['k', 'b', 'g', 'c', 'm', 'y', 'r', 'orange'] * 10
        for i, filt in enumerate(data.keys()):

            if skip is not None:
                if filt in skip:
                    continue

            if filters is not None:
                if filt not in filters:
                    continue

            c = colors[i]
            if kwargs != {}:
                if 'color' in kwargs:
                    c = kwargs['color']
                    del kwargs['color']

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

    def read_throughputs(self, filters=None):
        """
        Assembles a dictionary of throughput curves.

        Each element of the dictionary is a tuple containing:
            (wavelength, throughput, midpoint of filter,
                width of filter (FWHM), transmission averaged over filter)

        Example
        -------

        >>> wave, T, mid, wid, Tavg = self.read_throughputs()


        """

        if self.camera in ['nircam', 'jwst']:
            return self._read_nircam(filters)
        elif self.camera in ['hst', 'hubble']:
            wfc = self._read_wfc(filters)
            wfc3 = self._read_wfc3(filters)
            hst = wfc.copy()
            hst.update(wfc3)
            return hst
        elif self.camera == 'wfc3':
            return self._read_wfc3(filters)
        elif self.camera == 'wfc':
            return self._read_wfc(filters)
        elif self.camera in ['irac', 'spitzer']:
            return self._read_irac(filters)
        elif self.camera == 'roman':
            return self._read_roman(filters)
        elif self.camera == 'wise':
            return self._read_wise(filters)
        elif self.camera == '2mass':
            return self._read_2mass(filters)
        elif self.camera == 'euclid':
            return self._read_euclid(filters)
        elif self.camera == 'spherex':
            return self._read_spherex(filters)
        elif self.camera == 'rubin':
            return self._read_rubin(filters)
        elif self.camera == 'panstarrs':
            return self._read_panstarrs(filters)
        elif self.camera == 'sdss':
            return self._read_sdss(filters)
        elif self.camera == 'hsc':
            return self._read_hsc(filters)
        elif self.camera == 'dirbe':
            return self._read_dirbe(filters)
        else:
            raise NotImplemented(f"Unrecognized cam '{cam}'")

    def _read_nircam(self, filters=None): # pragma: no cover

        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        path = os.path.join(
            _path, "nircam", "nircam_throughputs", self.mod, "filters_only"
        )

        data = {}
        for fn in os.listdir(path):

            fname = fn.split('_')[0]

            # Do we care about this filter? If not, move along.
            if filters is not None:
                if type(filters) == str:
                    if filters not in fname:
                        continue

            if fname in self._filter_cache:
                data[fname] = self._filter_cache[fname]
                continue

            if ('W2' in fname):
                continue

            num = re.findall(r'\d+', fname)[0]
            cent = float('{}.{}'.format(num[0], num[1:]))

            # Wavelength [micron], transmission
            full_path = os.path.join(path, fn)
            x, y = np.loadtxt(full_path, unpack=True, skiprows=1)

            data[fname] = self._get_filter_prop(x, y, cent)

            self._filter_cache[fname] = copy.deepcopy(data[fname])

        return data

    def _read_wfc(self, filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        path = os.path.join(_path, "wfc")

        data = {}
        for fn in os.listdir(path):

            # Mac OS creates a bunch of ._wfc_* files. Argh.
            if not fn.startswith('wfc_'):
                continue

            if fn.endswith('tar.gz'):
                continue

            # Full name of the filter, e.g., F606W
            fname = fn.split('wfc_')[1].split('.dat')[0]

            # Do we care about this filter? If not, move along.
            if filters is not None:
                if type(filters) == str:
                    if filters not in fname:
                        continue

            #if get_all or (pre in filters):

            if fname in self._filter_cache:
                data[fname] = self._filter_cache[fname]
                continue

            cent = float('0.{}'.format(fname[1:4]))

            full_path = os.path.join(path, fn)
            x, y = np.loadtxt(full_path, unpack=True, skiprows=1)

            # Convert wavelengths from nanometers to microns
            data[fname] = self._get_filter_prop(x / 1e4, y, cent)

            self._filter_cache[fname] = copy.deepcopy(data[fname])

        return data

    def _read_wfc3(self, filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        path = os.path.join(_path, "wfc3")

        data = {}
        for fn in os.listdir(path):

            if not (fn.startswith('WFC3_IR') or fn.startswith('WFC3_UVIS')):
                continue

            if '_IR' in fn:
                fname = fn[fn.find('_IR')+4:]
            else:
                fname = fn[fn.find('_UVIS1')+7:]

            # Do we care about this filter? If not, move along.
            if filters is not None:
                if type(filters) == str:
                    if filters not in fname:
                        continue

            if fname in self._filter_cache:
                data[fname] = self._filter_cache[fname]
                continue

            if '_IR' in fn:
                cent = float('{}.{}'.format(fname[1], fname[2:-1]))
            else:
                if 'LP' in fname:
                    cent = float('0.{}'.format(fname[1:-2]))
                else:
                    cent = float('0.{}'.format(fname[1:-1]))

            full_path = os.path.join(path, fn)
            x, y = np.loadtxt(full_path, unpack=True, skiprows=1)

            # Convert wavelengths from Angstroms to microns
            data[fname] = self._get_filter_prop(x / 1e4, y, cent)

            self._filter_cache[fname] = copy.deepcopy(data[fname])

        return data

    def _read_irac(self, filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        path = os.path.join(_path, "irac")

        data = {}
        for fn in os.listdir(path):
            full_path = os.path.join(path, fn)
            x, y = np.loadtxt(full_path, unpack=True, skiprows=1)

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

        path = os.path.join(_path, "roman")

        data = {}
        for fn in os.listdir(path):

            if fn != _fn:
                continue

            full_path = os.path.join(path, _fn)
            df = pd.read_excel(
                full_path, sheet_name='Roman_effarea_20201130', header=1
            )

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

    def _read_wise(self, filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        path = os.path.join(_path, "wise")

        data = {}
        cent = 3.368, 4.618
        for i, filt in enumerate(['W1', 'W2']):
            full_path = os.path.join(path, f"RSR-{filt}.txt")
            x, y, z = np.loadtxt(full_path, unpack=True)
            data[filt] = self._get_filter_prop(np.array(x), np.array(y), cent[i])

            self._filter_cache[filt] = copy.deepcopy(data[filt])

        return data

    def _read_2mass(self, filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        path = os.path.join(_path, "2mass")

        data = {}
        cent = 1.235, 1.662, 2.159
        for i, filt in enumerate(['J', 'H', 'Ks']):
            full_path = os.path.join(path, f"2MASS.{filt}")
            x, y = np.loadtxt(full_path, unpack=True)
            x *= 1e-4
            data[filt] = self._get_filter_prop(np.array(x), np.array(y), cent[i])

            self._filter_cache[filt] = copy.deepcopy(data[filt])

        return data

    def _read_euclid(self, filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        path = os.path.join(_path, "euclid")

        data = {}
        cent = 1.0809, 1.3673, 1.7714
        for i, filt in enumerate(['Y', 'J', 'H']):
            full_path = os.path.join(path,
                f"NISP-PHOTO-PASSBANDS-V1-{filt}_throughput.dat")
            x, y = np.loadtxt(full_path, unpack=True, usecols=[0,1])
            x *= 1e-3
            data[filt] = self._get_filter_prop(np.array(x), np.array(y), cent[i])

            self._filter_cache[filt] = copy.deepcopy(data[filt])

        return data

    def _read_panstarrs(self, filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        path = os.path.join(_path, "panstarrs")

        data = {}
        cent = 0.493601, 0.620617, 0.755348, 0.870475, 0.952863
        for i, filt in enumerate(['g', 'r', 'i', 'z', 'y']):
            full_path = os.path.join(path, f"PS1.{filt}")
            x, y = np.loadtxt(full_path, unpack=True, usecols=[0,1])
            x *= 1e-4
            data[filt] = self._get_filter_prop(np.array(x), np.array(y), cent[i])

            self._filter_cache[filt] = copy.deepcopy(data[filt])

        return data

    def _read_spherex(self, filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        path = os.path.join(_path, "spherex", "Public-products-master")
        fn = 'Surface_Brightness_v28_base_cbe.txt'
        full_path = os.path.join(path, fn)
        x, allsky, deep = np.loadtxt(full_path, unpack=True)

        self._filter_cache['all'] = x, allsky, deep

        return x, allsky, deep

    def _read_rubin(self, filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        path = os.path.join(_path, "rubin", "throughputs", "baseline")

        data = {}
        for i, filt in enumerate(list('ugrizy')):
            full_path = os.path.join(path, f"total_{filt}.dat")
            x, y = np.loadtxt(full_path, unpack=True)
            cent = np.mean(x[y > 0])
            data[filt] = self._get_filter_prop(np.array(x), np.array(y), cent)

            self._filter_cache[filt] = copy.deepcopy(data[filt])

        return data

    def _read_sdss(self, filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        path = os.path.join(_path, "sdss")

        from astropy.io import fits
        hdulist = fits.open(f"{path}/filter_curves.fits")

        data = {}
        for i, filt in enumerate(list('ugriz')):
            x = 1e-4 * np.array([element[0] for element in hdulist[i+1].data])
            y = np.array([element[1] for element in hdulist[i+1].data])

            cent = np.mean(x[y > 0])
            data[filt] = self._get_filter_prop(np.array(x), np.array(y), cent)

            self._filter_cache[filt] = copy.deepcopy(data[filt])

        return data

    def _read_hsc(self, filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        path = os.path.join(_path, "hsc")

        data = {}
        cent = 4798.21e-4, 6218.44e-4, 7727.01e-4, 8908.50e-4, 9775.07e-4
        for i, filt in enumerate(list('grizY')):
            full_path = os.path.join(path, f"HSC.{filt}")
            x, y = np.loadtxt(full_path, unpack=True, usecols=[0,1])
            x *= 1e-4
            data[filt] = self._get_filter_prop(np.array(x), np.array(y), cent[i])

            self._filter_cache[filt] = copy.deepcopy(data[filt])

        return data

    def _read_dirbe(self, filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}

        path = os.path.join(_path, "dirbe")

        cent = 1.25, 2.2, 3.5, 4.9, 12, 25, 60, 100, 140, 240
        data = {}
        full_path = os.path.join(path,
            "DIRBE_SYSTEM_SPECTRAL_RESPONSE_TABLE.ASC")

        _data = np.loadtxt(full_path, unpack=True, skiprows=15)

        x = _data[0]
        for _i, band in enumerate(range(1, 11)):
            data[f"band{band}"] = self._get_filter_prop(x, _data[1+_i], cent[_i])
            self._filter_cache[f"band{band}"] = copy.deepcopy(data[f"band{band}"])

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

    def get_filter_info(self, filt):
        """
        Returns the filter center and "full-width-full-max" in microns.
        """

        # Just to make sure we've loaded in data.
        data = self.read_throughputs()

        return self._filter_cache[filt][2], sum(self._filter_cache[filt][3])

    def get_dropout_filter(self, z, filters=None, drop_wave=1216., skip=None):
        """
        Determine where the Lyman break happens and return the corresponding
        filter.
        """

        data = self.read_throughputs()

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
