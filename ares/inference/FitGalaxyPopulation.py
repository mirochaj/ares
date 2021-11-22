"""

FitGLF.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Oct 23 14:34:01 PDT 2015

Description:

"""

import gc, os
import numpy as np
from ..util import read_lit
from ..util.Pickling import write_pickle_file
from ..util.ParameterFile import par_info
from ..util.Stats import symmetrize_errors
from ..populations import GalaxyCohort, GalaxyEnsemble, GalaxyHOD
from .ModelFit import LogLikelihood, FitBase, def_kwargs

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

twopi = 2. * np.pi

class loglikelihood(LogLikelihood):

    @property
    def redshifts(self):
        return self._redshifts
    @redshifts.setter
    def redshifts(self, value):
        self._redshifts = value

    @property
    def metadata(self):
        return self._metadata
    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def units(self):
        return self._units
    @units.setter
    def units(self, value):
        self._units = value

    @property
    def zmap(self):
        if not hasattr(self, '_zmap'):
            self._zmap = {}
        return self._zmap

    @zmap.setter
    def zmap(self, value):
        self._zmap = value

    @property
    def mask(self):
        if not hasattr(self, '_mask'):
            if type(self.xdata) is np.ma.core.MaskedArray:
                self._mask = self.xdata.mask
            else:
                self._mask = np.zeros(len(self.xdata))

        return self._mask

    @property
    def include(self):
        if not hasattr(self, '_include'):

            assert self.metadata is not None

            self._include = []
            for item in self.metadata:
                if item in self._include:
                    continue

                self._include.append(item)

        return self._include

    @property
    def monotonic_beta(self):
        if not hasattr(self, '_monotonic_beta'):
            self._monotonic_beta = False
        return self._monotonic_beta

    @monotonic_beta.setter
    def monotonic_beta(self, value):
        self._monotonic_beta = bool(value)

    def __call__(self, sim):
        """
        Compute log-likelihood for model generated via input parameters.

        Returns
        -------
        Tuple: (log likelihood, blobs)

        """

        # Figure out if `sim` is a population object or not.
        # OK if it's a simulation, will loop over LF-bearing populations.
        if not (isinstance(sim, GalaxyCohort.GalaxyCohort) \
            or isinstance(sim, GalaxyEnsemble.GalaxyEnsemble) or isinstance(sim, GalaxyHOD.GalaxyHOD) ):
            pops = []
            for pop in sim.pops:
                if not hasattr(pop, 'LuminosityFunction'):
                    continue

                pops.append(pop)

        else:
            pops = [sim]

        if len(self.ydata) == 0:
            raise ValueError("Problem: data is empty.")

        if len(pops) > 1:
            raise NotImplemented('careful! need to think about this.')

        # Loop over all data points individually.
        #try:
        phi = np.zeros_like(self.ydata)
        for i, quantity in enumerate(self.metadata):

            if quantity == 'beta':
                _b14 = read_lit('bouwens2014')

            if self.mask[i]:
                #print('masked:', rank, self.redshifts[i], self.xdata[i])
                continue

            xdat = self.xdata[i]
            z = self.redshifts[i]

            if quantity in self.zmap:
                zmod = self.zmap[quantity][z]
            else:
                zmod = z

            for j, pop in enumerate(pops):

                # Generate model LF
                if quantity == 'lf':

                    # New convention: LuminosityFunction always in terms of
                    # observed magnitudes.

                    # Compute LF
                    _x, p = pop.get_lf(z=zmod, bins=xdat, use_mags=True)

                    if not np.isfinite(p):
                        print('WARNING: LF is inf or nan!', zmod, xdat, p)
                        return -np.inf
                elif quantity.startswith('smf'):
                    if quantity in ['smf', 'smf_tot']:
                        M = np.log10(xdat)
                        p = pop.StellarMassFunction(zmod, M)

                    else:
                        M = np.log10(xdat)
                        p = pop.StellarMassFunction(zmod, M, sf_type=quantity)

                elif quantity == 'beta':

                    zstr = int(round(zmod))

                    if zstr >= 7:
                        filt_hst = _b14.filt_deep
                    else:
                        filt_hst = _b14.filt_shallow

                    M = xdat
                    p = pop.Beta(zmod, MUV=M, presets='hst', dlam=20.,
                        return_binned=True, rest_wave=None)

                    if not np.isfinite(p):
                        print('WARNING: beta is inf or nan!', z, M)
                        return -np.inf
                        #raise ValueError('beta is inf or nan!', z, M)

                else:
                    raise ValueError('Unrecognized quantity: {!s}'.format(\
                        quantity))

                # If UVLF or SMF, could do multi-pop in which case we'd
                # increment here.
                phi[i] = p

        ##
        # Apply restrictions to beta
        if self.monotonic_beta:

            if type(self.monotonic_beta) in [int, float, np.float64]:
                Mlim = self.monotonic_beta
            else:

                # Don't let beta turn-over in the range of magnitudes that
                # overlap with UVLF constraints, or 2 extra mags if no UVLF
                # fitting happening (rare).

                xmod = []
                ymod = []
                zmod = []
                xlf = []
                for i, quantity in enumerate(self.metadata):
                    if quantity == 'lf':
                        xlf.append(self.xdata[i])

                    z = self.redshifts[i]

                    if quantity in self.zmap:
                        _zmod = self.zmap[quantity][z]
                    else:
                        _zmod = z

                    xmod.append(self.xdata[i])
                    ymod.append(phi[i])
                    zmod.append(_zmod)

                i_lo = np.argmin(xmod)
                M_lo = xmod[i_lo]
                b_lo = ymod[i_lo]

                if 'lf' in self.metadata:
                    Mlim = np.nanmin(xlf) - 1.
                else:
                    Mlim = M_lo - 2.

            b_hi = {}
            for i, quantity in enumerate(self.metadata):
                if quantity != 'beta':
                    continue

                if zmod[i] not in b_hi:
                    b_hi[zmod[i]] = pop.Beta(zmod[i], MUV=Mlim, presets='hst',
                        dlam=20., return_binned=True, rest_wave=None)

                if not (np.isfinite(Mlim) or np.isfinite(b_hi[zmod[i]])):
                    raise ValueError("Mlim={}, beta_hi={:.2f}".format(Mlim, b_hi))

                # Bit overkill to check every magnitude, but will *really*
                # enforce monotonic behavior.
                if b_hi[zmod[i]] < ymod[i]:
                    print('beta is not monotonic!', zmod[i],
                        Mlim, b_hi[zmod[i]], xmod[i], ymod[i])
                    return -np.inf
                #else:
                #    print("beta monotonic at z={}: beta(MUV={})={}, beta(MUV={})={}".format(zmod[i],
                #        Mlim, b_hi[zmod[i]], xmod[i], ymod[i]))

        #except:
        #    return -np.inf, self.blank_blob

        #phi = np.ma.array(_phi, mask=self.mask)

        #del sim, pops
        lnL = -0.5 * np.ma.sum((phi - self.ydata)**2 / self.error**2)

        return lnL + self.const_term

class FitGalaxyPopulation(FitBase):

    @property
    def loglikelihood(self):
        if not hasattr(self, '_loglikelihood'):
            self._loglikelihood = loglikelihood(self.xdata_flat,
                self.ydata_flat, self.error_flat)

            self._loglikelihood.redshifts = self.redshifts_flat
            self._loglikelihood.metadata = self.metadata_flat
            self._loglikelihood.zmap = self.zmap
            self._loglikelihood.monotonic_beta = self.monotonic_beta

            self.info

        return self._loglikelihood

    @property
    def monotonic_beta(self):
        if not hasattr(self, '_monotonic_beta'):
            self._monotonic_beta = False
        return self._monotonic_beta

    @monotonic_beta.setter
    def monotonic_beta(self, value):
        self._monotonic_beta = bool(value)

    @property
    def zmap(self):
        if not hasattr(self, '_zmap'):
            self._zmap = {}
        return self._zmap

    @zmap.setter
    def zmap(self, value):
        self._zmap = value

    @property
    def redshift_bounds(self):
        if not hasattr(self, '_redshift_bounds'):
            raise ValueError('Set by hand or include in litdata.')

        return self._redshift_bounds

    @redshift_bounds.setter
    def redshift_bounds(self, value):
        assert len(value) == 2

        self._redshift_bounds = tuple(value)

    @property
    def redshifts(self):
        if not hasattr(self, '_redshifts'):
            raise ValueError('Set by hand or include in litdata.')

        return self._redshifts

    @redshifts.setter
    def redshifts(self, value):
        # This can be used to override the redshifts in the dataset and only
        # use some subset of them

        # Need to be ready for 'lf' or 'smf' designation.
        if len(self.include) > 1:
            assert type(value) is dict

        if type(value) in [int, float]:
            value = [value]

        self._redshifts = value

    @property
    def ztol(self):
        if not hasattr(self, '_ztol'):
            self._ztol = 0.
        return self._ztol

    @ztol.setter
    def ztol(self, value):
        self._ztol = value

    @property
    def data(self):
        if not hasattr(self, '_data'):
            raise AttributeError('Must set data by hand!')
        return self._data

    @data.setter
    def data(self, value):
        """
        Set the data (duh).

        The structure is as follows. The highest level division is between
        different quantities (e.g., 'lf' vs. 'smf'). Each of these quantities
        is an element of the returned dictionary. For each, there is a list
        of dictionaries, one per redshift. Each redshift dictionary contains
        the magnitudes (or masses) along with number density measurements
        and error-bars.

        """

        if isinstance(value, basestring):
            value = [value]

        if type(value) in [list, tuple]:
            self._data = {quantity:[] for quantity in self.include}
            self._units = {quantity:[] for quantity in self.include}

            z_by_range = hasattr(self, '_redshift_bounds')
            z_by_hand = hasattr(self, '_redshifts')

            if not z_by_hand:
                self._redshifts = {quantity:[] for quantity in self.include}

            # Loop over data sources
            for src in value:

                # Grab the data
                litdata = read_lit(src)

                # Loop over LF, SMF, etc.
                for quantity in self.include:
                    if quantity not in litdata.data.keys():
                        continue

                    # Short hand
                    data = litdata.data[quantity]
                    redshifts = litdata.redshifts

                    # This is always just a number or str, i.e.,
                    # no need to breakdown by redshift so just do it now
                    self._units[quantity].append(litdata.units[quantity])

                    # Now, be careful about what redshifts to include.
                    if not (z_by_range or z_by_hand):
                        srcdata = data
                        srczarr = redshifts
                        print('not by hand', srczarr)
                    else:
                        srczarr = []
                        srcdata = {}
                        for z in redshifts:

                            if z_by_range:
                                zb = self.redshift_bounds
                                if (zb[0] <= z <= zb[1]):
                                    srczarr.append(z)
                                    srcdata[z] = data[z]
                                    continue

                            # z by hand from here down.
                            # Find closest redshift to those requested,
                            # see if it meets our tolerance.
                            zreq = np.array(self._redshifts[quantity])
                            iz = np.argmin(np.abs(z - zreq))

                            # Does this redshift match any we've requested?
                            if abs(z - zreq[iz]) > self.ztol:
                                continue

                            srczarr.append(z)
                            srcdata[z] = data[z]

                    self._data[quantity].append(srcdata)

                    if not z_by_hand:
                        self._redshifts[quantity].extend(srczarr)

            # Check to make sure we find requested measurements.
            for quantity in self.include:
                zlit = []
                for element in self._data[quantity]:
                    zlit.extend(list(element.keys()))

                zlit = np.array(zlit).ravel()
                zreq = self._redshifts[quantity]

                # Problems straight away if we don't have enough redshifts
                if len(zlit) != len(zreq):
                    s = "Found {} suitable redshifts for {}.".format(len(zlit),
                        quantity)
                    s += " Requested {}.".format(len(zreq))
                    s += "z_requested={}, z_found={}.".format(zreq, zlit)
                    s += " Perhaps rounding issue? Toggle `ztol` attribute"
                    s += " to be more lenient in finding match with measurements."
                    raise ValueError(s)

                # Need to loop over all sources. When we're done, should be
                # able to account for all requested redshifts.
                for j, z in enumerate(zreq):

                    if z != zlit[j]:
                        s = "# Will fit to {} at z={}".format(quantity, zlit[j])
                        s += " as it lies within ztol={} of requested z={}".format(
                            self.ztol, z)
                        if rank == 0:
                            print(s)

        else:
            raise NotImplemented('help!')

    @property
    def include(self):
        if not hasattr(self, '_include'):
            self._include = ['lf']
        return self._include

    @include.setter
    def include(self, value):
        self._include = value

    @property
    def xdata_flat(self):
        if not hasattr(self, '_xdata_flat'):
            self._mask = []
            self._xdata_flat = []; self._ydata_flat = []
            self._error_flat = []; self._redshifts_flat = []
            self._metadata_flat = []

            for quantity in self.include:

                # Sorted by sources
                for i, dataset in enumerate(self.data[quantity]):

                    for j, redshift in enumerate(self.data[quantity][i]):
                        M = self.data[quantity][i][redshift]['M']

                        # These could still be in log10 units
                        if quantity == 'beta':
                            phi = self.data[quantity][i][redshift]['beta']
                        else:
                            phi = self.data[quantity][i][redshift]['phi']

                        err = self.data[quantity][i][redshift]['err']

                        if hasattr(M, 'mask'):
                            self._mask.extend(M.mask)
                            self._xdata_flat.extend(M.data)
                        else:
                            self._mask.extend(np.zeros_like(M))
                            self._xdata_flat.extend(M)

                        if self.units[quantity][i] == 'log10':
                            _phi = 10**phi
                        else:
                            _phi = phi

                        if hasattr(M, 'mask'):
                            self._ydata_flat.extend(_phi.data)
                        else:
                            self._ydata_flat.extend(_phi)

                        # Cludge for asymmetric errors
                        for k, _err in enumerate(err):

                            if self.units[quantity][i] == 'log10':
                                _err_ = symmetrize_errors(phi[k], _err,
                                    operation='min')
                            else:
                                _err_ = _err

                            if type(_err_) in [tuple, list]:
                                self._error_flat.append(np.mean(_err_))
                            else:
                                self._error_flat.append(_err_)

                        zlist = [redshift] * len(M)
                        self._redshifts_flat.extend(zlist)
                        self._metadata_flat.extend([quantity] * len(M))

            self._mask = np.array(self._mask)
            self._xdata_flat = np.ma.array(self._xdata_flat, mask=self._mask)
            self._ydata_flat = np.ma.array(self._ydata_flat, mask=self._mask)
            self._error_flat = np.ma.array(self._error_flat, mask=self._mask)

        return self._xdata_flat

    @property
    def ydata_flat(self):
        if not hasattr(self, '_ydata_flat'):
            xdata_flat = self.xdata_flat

        return self._ydata_flat

    @property
    def error_flat(self):
        if not hasattr(self, '_error_flat'):
            xdata_flat = self.xdata_flat

        return self._error_flat

    @property
    def redshifts_flat(self):
        if not hasattr(self, '_redshifts_flat'):
            xdata_flat = self.xdata_flat

        return self._redshifts_flat

    @property
    def metadata_flat(self):
        if not hasattr(self, '_metadata_flat'):
            xdata_flat = self.xdata_flat

        return self._metadata_flat

    @property
    def units(self):
        if not hasattr(self, '_units'):
            xdata_flat = self.xdata_flat

        return self._units

    @property
    def xdata(self):
        if not hasattr(self, '_xdata'):
            if hasattr(self, '_data'):
                self._xdata = []; self._ydata = []; self._error = []
                #for i, dataset in enumerate(self.redshifts):
                for h, quantity in enumerate(self.include):
                    for i, dataset in enumerate(self.data[quantity]):
                        for j, redshift in enumerate(self.data[i]):
                            self._xdata.append(dataset[redshift]['M'])

                            if quantity == 'beta':
                                self._ydata.append(dataset[redshift]['beta'])
                            else:
                                self._ydata.append(dataset[redshift]['phi'])
                            self._error.append(dataset[redshift]['err'])

        return self._xdata

    @xdata.setter
    def xdata(self, value):
        self._xdata = value

    @property
    def ydata(self):
        if not hasattr(self, '_ydata'):
            if hasattr(self, '_data'):
                xdata = self.xdata

        return self._ydata

    @ydata.setter
    def ydata(self, value):
        self._ydata = value

    @property
    def error(self):
        if not hasattr(self, '_error'):
            if hasattr(self, '_data'):
                xdata = self.xdata
        return self._error

    @error.setter
    def error(self, value):
        self._error = value

    @property
    def guess_override(self):
        if not hasattr(self, '_guess_override_'):
            self._guess_override_ = {}

        return self._guess_override_

    @guess_override.setter
    def guess_override(self, kwargs):
        if not hasattr(self, '_guess_override_'):
            self._guess_override_ = {}

        self._guess_override_.update(kwargs)

    def save_data(self, prefix, clobber=False):
        if rank > 0:
            return

        fn = '{!s}.data.pkl'.format(prefix)

        if os.path.exists(fn) and (not clobber):
            print("{!s} exists! Set clobber=True to overwrite.".format(fn))
            return

        write_pickle_file((self.xdata, self.ydata, self.redshifts,\
            self.error), fn, ndumps=1, open_mode='w', safe_mode=False,\
            verbose=False)
