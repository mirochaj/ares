"""

MetaGalacticBackground.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 16 12:43:06 MST 2015

Description:

"""

import os
import time
import scipy
import numpy as np
from ..static import Grid
from ..util.Pickling import write_pickle_file
from types import FunctionType
from ..util import ParameterFile
from ..obs import Madau1995
from ..util.Misc import split_by_sign
from ..util.Math import interp1d, smooth
from ..solvers import UniformBackground
from ..analysis.MetaGalacticBackground import MetaGalacticBackground \
    as AnalyzeMGB
from ..physics.Constants import E_LyA, E_LL, ev_per_hz, erg_per_ev, \
    sqdeg_per_std, s_per_myr, rhodot_cgs, cm_per_mpc, c, h_p, k_B, \
    cm_per_m, erg_per_s_per_nW
from ..util.ReadData import _sort_history, flatten_energies, flatten_flux
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

_scipy_ver = scipy.__version__.split('.')

# This keyword didn't exist until version 0.14
if float(_scipy_ver[1]) >= 0.14:
    _interp1d_kwargs = {'assume_sorted': True}
else:
    _interp1d_kwargs = {}

_tiny_sfrd = 1e-12

def NH2(z, Mh, fH2=3.5e-4):
    return 1e17 * (fH2 / 3.5e-4) * (Mh / 1e6)**(1. / 3.) * ((1. + z) / 20.)**2.

def f_sh_fl13(zz, Mmin):
    return np.minimum(1, (NH2(zz, Mh=Mmin) / 1e14)**-0.75)

def get_Mmin_func(zarr, Jlw, Mmin_prev, **kwargs):

    # Self-shielding. Function of redshift only!
    if kwargs['feedback_LW_fsh']:
        f_sh = f_sh_fl13
    #elif type(kwargs['feedback_LW_fsh']) is FunctionType:
    #    f_sh = kwargs['feedback_LW_fsh']
    else:
        f_sh = lambda z, Mmin: 1.0

    # Interpolants for Jlw, Mmin, for this iteration
    # This is Eq. 4 from Visbal+ 2014
    Mmin = lambda zz: np.interp(zz, zarr, Mmin_prev)
    f_J = lambda zz: f_sh(zz, Mmin(zz)) * np.interp(zz, zarr, Jlw)

    if kwargs['feedback_LW_Mmin'] == 'visbal2014':
        f_M = lambda zz: 2.5 * 1e5 * pow(((1. + zz) / 26.), -1.5) \
            * (1. + 6.96 * pow(4 * np.pi * f_J(zz), 0.47))
    elif type(kwargs['feedback_LW_Mmin']) is FunctionType:
        f_M = kwargs['feedback_LW_Mmin']
    else:
        raise NotImplementedError('Unrecognized Mmin option: {!s}'.format(\
            kwargs['feedback_LW_Mmin']))

    return f_M

class MetaGalacticBackground(AnalyzeMGB):
    def __init__(self, pf=None, grid=None, **kwargs):
        """
        Initialize a MetaGalacticBackground object.
        """

        self.kwargs = kwargs

        if pf is None:
            self.pf = ParameterFile(**self.kwargs)
        else:
            self.pf = pf

        self._grid = grid
        self._has_fluxes = False
        self._has_coeff = False

        self._run_complete = False

        if not hasattr(self, '_suite'):
            self._suite = []

    @property
    def pf(self):
        if not hasattr(self, '_pf'):
            self._pf = ParameterFile(**self.kwargs)
        return self._pf

    @pf.setter
    def pf(self, val):
        self._pf = val

    @property
    def solver(self):
        if not hasattr(self, '_solver'):
            self._solver = UniformBackground(pf=self.pf, grid=self.grid,
                **self.kwargs)
        return self._solver

    @property
    def pops(self):
        if not hasattr(self, '_pops'):
            self._pops = self.solver.pops
        return self._pops

    @property
    def count(self):
        if not hasattr(self, '_count'):
            self._count = 0
        return self._count

    @property
    def grid(self):
        if self._grid is None:
            self._grid = Grid(
                grid_cells=self.pf['grid_cells'],
                length_units=self.pf['length_units'],
                start_radius=self.pf['start_radius'],
                approx_Salpha=self.pf['approx_Salpha'],
                logarithmic_grid=self.pf['logarithmic_grid'],
                cosmological_ics=self.pf['cosmological_ics'],
                )
            self._grid.set_properties(**self.pf)

        return self._grid

    @property
    def rank(self):
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.rank
        except ImportError:
            rank = 0

        return rank

    def run(self, include_pops=None, xe=None):
        """
        Loop over populations, determine background intensity.

        Parameters
        ----------
        include_pops : list
            For internal use only!

        """

        # In this case, we can evolve all LW sources first, and wait
        # to do other populations until the very end.
        if self.pf['feedback_LW'] and (include_pops is None):
            # This means it's the first iteration
            include_pops = self._lwb_sources
        elif include_pops is None:
            include_pops = range(self.solver.Npops)

        if not hasattr(self, '_data_'):
            self._data_ = {}

        for i, popid in enumerate(include_pops):
            z, fluxes = self.run_pop(popid=popid, xe=xe)
            self._data_[popid] = fluxes

        # Each element of the history is series of lists.
        # The first axis corresponds to redshift, while the second
        # dimension has as many chunks as there are sub-bands for RTE solutions.
        # Also: redshifts are *descending* at this point
        self._history = self._data_.copy()
        self._suite.append(self._data_.copy())

        count = self.count   # Just to make sure attribute exists
        self._count += 1

        is_converged = self._is_Mmin_converged(self._lwb_sources)

        ##
        # Feedback
        ##
        if is_converged:
            self._has_fluxes = True
            self._f_Jc = lambda z: np.interp(z, self._zarr, self._Jc,
                left=0.0, right=0.0)
            self._f_Ji = lambda z: np.interp(z, self._zarr, self._Ji,
                left=0.0, right=0.0)
            self._f_Jlw = lambda z: np.interp(z, self._zarr, self._Jlw,
                left=0.0, right=0.0)

            # Now that feedback is done, evolve all non-LW sources to get
            # final background.
            if include_pops == self._lwb_sources:
                self.reboot(include_pops=self._not_lwb_sources)
                self.run(include_pops=self._not_lwb_sources)

            self._run_complete = True

        else:
            if self.pf['verbose']:
                if hasattr(self, '_sfrd_bank') and self.count >= 2:
                    pid = self.pf['feedback_LW_sfrd_popid']
                    ibad = np.argmax(self._sfrd_rerr[self._ok==1])
                    z_maxerr = \
                        self.pops[pid].halos.tab_z[self._ok==1][ibad]
                    print(("# LWB cycle #{0} complete: mean_err={1:.2e}, " +\
                        "max_err={2:.2e}, z(max_err)={3:.1f}").format(\
                        self.count, np.mean(self._sfrd_rerr[self._ok==1]),\
                        self._sfrd_rerr[self._ok==1][ibad], z_maxerr))
                else:
                    print("# LWB cycle #{} complete.".format(self.count))


            self.reboot()
            self.run(include_pops=self._lwb_sources)

        self._count += 1

    @property
    def today(self):
        """
        Return background intensity at z=zf evolved to z=0 assuming optically
        thin IGM.

        This is just the second term of Eq. 25 in Mirocha (2014).
        """

        return self.flux_today()

    def today_of_E(self, E):
        """
        Grab radiation background at a single energy at z=0.
        """
        nrg, fluxes = self.today

        return np.interp(E, nrg, fluxes)

    def temp_of_E(self, E):
        """
        Convert the z=0 background intensity to a temperature in K.
        """
        flux = self.today_of_E(E)

        freq = E * erg_per_ev / h_p
        return flux * E * erg_per_ev * c**2 / k_B / 2. / freq**2

    def flux_today(self, zf=None, popids=None, units='cgs', xunits='eV'):
        """
        Propage radiation background from `zf` to z=0 assuming optically
        thin universe.

        Parameters
        ----------
        zf : int, float
            Final redshift to include in flux.
        popids : tuple, list
            Can restrict flux to that generated by certain source populations.
            If None, will return total z=0 flux from all populations.
        units : str
            Control output units. By default, result is in erg/s/cm^2/sr for
            units=='cgs'. Other options include:

                units='si' returns nu*Inu in nano-watts / m^2 / sr

        """
        _fluxes_today = []
        _energies_today = []

        if popids is None:
            popids = list(range(len(self.pops)))
        if type(popids) not in [list, tuple, np.ndarray]:
            popids = [popids]

        ct = 0
        _zf = [] # for debugging
        # Loop over pops: assumes energy ranges are non-overlapping!
        for popid, pop in enumerate(self.pops):

            if popid not in popids:
                continue

            if not self.solver.solve_rte[popid]:
                continue

            # Much faster to only read in first redshift element in this
            # case.
            z, E, flux = self.get_history(popid=popid, flatten=True,
                today_only=False)

            if zf is None:
                k = 0
                #_zf.append(z[k])
                _zf.append(pop.zdead)
            else:
                k = np.argmin(np.abs(zf - z))

            Et = E / (1. + z[k])
            ft = flux[k] / (1. + z[k])**2

            _energies_today.append(Et)
            _fluxes_today.append(ft)

            ct += 1

        ##
        # Add the fluxes! Interpolate to common energy grid first.
        ##

        _f = []
        _E = np.unique(np.concatenate(_energies_today))
        for i, flux in enumerate(_fluxes_today):
            _f.append(np.interp(_E, _energies_today[i], flux,
                left=0.0, right=0.0))

        f = np.sum(_f, axis=0)

        # Wavelength in microns (might need it below)
        _lam = h_p * c * 1e4 / erg_per_ev / _E

        # Attenuate by HI absorbers in IGM at z < zf?
        if self.pf['tau_clumpy'] is not None:
            assert self.pf['tau_clumpy'].lower() == 'madau1995'

            m95 = Madau1995(hydr=self.grid.hydr, **self.pf)

            if zf is None:
                assert np.allclose(np.array(_zf) - _zf[0], 0)
                zf = _zf[0]

            f *= np.exp(-m95(zf, _lam))


        if units.lower() == 'cgs':
            pass
        elif units.lower() == 'si':
            nu = _E * erg_per_ev / h_p
            f *= nu * _E * erg_per_ev * cm_per_m**2 / erg_per_s_per_nW
        else:
            raise ValueError('Unrecognized units=`{}`.'.format(units))

        if xunits.lower() == 'ev':
            pass
        elif xunits.lower() in ['angstrom', 'ang', 'a']:
            _E = _lam
        else:
            raise NotImplemented("'don't recognize xunits={}".format(xunits))

        return _E, f

    @property
    def jsxb(self):
        if not hasattr(self, '_jsxb'):
            self._jsxb = self.jxrb(band='soft')
        return self._jsxb

    @property
    def jhxb(self):
        if not hasattr(self, '_jhxb'):
            self._jhxb = self.jxrb(band='hard')
        return self._jhxb

    def jxrb(self, band='soft'):
        """
        Compute soft X-ray background flux at z=0.
        """

        jx = 0.0
        Ef, ff = self.today()
        for popid, pop in enumerate(self.pops):
            if Ef[popid] is None:
                continue

            flux_today = ff[popid] * Ef[popid] \
                * erg_per_ev / sqdeg_per_std / ev_per_hz

            if band == 'soft':
                Eok = np.logical_and(Ef[popid] >= 5e2, Ef[popid] <= 2e3)
            elif band == 'hard':
                Eok = np.logical_and(Ef[popid] >= 2e3, Ef[popid] <= 1e4)
            else:
                raise ValueError('Unrecognized band! Only know \'hard\' and \'soft\'')

            Earr = Ef[popid][Eok]
            # Find integrated 0.5-2 keV flux
            dlogE = np.diff(np.log10(Earr))

            jx += np.trapz(flux_today[Eok] * Earr, dx=dlogE) * np.log(10.)

        return jx

    @property
    def _not_lwb_sources(self):
        if not hasattr(self, '_not_lwb_sources_'):
            self._not_lwb_sources_ = []
            for i, pop in enumerate(self.pops):
                if pop.is_src_lw:
                    continue
                self._not_lwb_sources_.append(i)

        return self._not_lwb_sources_

    @property
    def _lwb_sources(self):
        if not hasattr(self, '_lwb_sources_'):
            self._lwb_sources_ = []
            for i, pop in enumerate(self.pops):
                if pop.is_src_lw:
                    self._lwb_sources_.append(i)

        return self._lwb_sources_

    def run_pop(self, popid=0, xe=None):
        """
        Evolve radiation background in time.

        .. note:: Assumes we're using the generator, otherwise the time
            evolution must be controlled manually.

        Returns
        -------
        Nothing: sets `history` attribute containing the entire evolution
        of the background for each population.

        """

        if self.solver.approx_all_pops:
            return None, None

        if not self.pops[popid].pf['pop_solve_rte']:
            return None, None

        t = 0.0
        z = self.pf['initial_redshift']
        zf = self.pf['final_redshift']

        all_fluxes = []
        for z in self.solver.redshifts[popid][-1::-1]:
            _z, fluxes = self.update_fluxes(popid=popid)
            all_fluxes.append(fluxes)

        return self.solver.redshifts[popid][-1::-1], all_fluxes

    def reboot(self, include_pops=None):
        delattr(self, '_history')
        delattr(self, '_pf')

        if include_pops is None:
            include_pops = range(self.solver.Npops)

        ## All quantities that depend on Mmin.
        #to_del = ['_tab_Mmin_', '_Mmin', '_tab_sfrd_total_',
        #    '_tab_sfrd_at_threshold_', '_tab_fstar_at_Mmin_',
        #    '_tab_nh_at_Mmin_', '_tab_MAR_at_Mmin_']

        # Reset Mmin for feedback-susceptible populations
        for popid in include_pops:
            #pop = self.pops[popid]
            #
            #if not pop.feels_feedback:
            #    continue
            #
            #if self.pf['pop_Tmin{{{}}}'.format(popid)] is not None:
            #    if self.pf['pop_Tmin{{{}}}'.format(popid)] >= 1e4:
            #        continue

            #for key in to_del:
            #    try:
            #        delattr(pop, key)
            #    except AttributeError:
            #        print("Attribute {!s} didn't exist.".format(key))
            #        continue

            # Linked populations will get
            if isinstance(self.pf['pop_Mmin{{{}}}'.format(popid)], basestring):
                if self.pf['pop_Mmin{{{}}}'.format(popid)] not in self.pf['cosmological_Mmin']:
                    continue

            self.kwargs['pop_Mmin{{{}}}'.format(popid)] = \
                np.interp(self.pops[popid].halos.tab_z, self.z_unique, self._Mmin_now)

            # Need to make sure, if any populations are linked to this Mmin,
            # that they get updated too.

            #if not self.pf['feedback_clear_solver']:
            #    pop._tab_Mmin = np.interp(pop.halos.z, self._zarr, self._Mmin_now)
            #    bands = self.solver.bands_by_pop[popid]
            #    z, nrg, tau, ehat = self.solver._set_grid(pop, bands,
            #        compute_emissivities=True)
            #
            #    k = range(self.solver.Npops).index(popid)
            #    self.solver._emissivities[k] = ehat

        if self.pf['feedback_clear_solver']:
            delattr(self, '_solver')
            delattr(self, '_pops')
        else:
            delattr(self.solver, '_generators')

        ##
        # Only read guesses on first iteration. Turn off for all subsequent
        # iterations. Don't like modifying pf in general, but kind of need to
        # here.
        ##
        if self.pf['feedback_LW_guesses'] is not None:
            self.kwargs['feedback_LW_guesses'] = None

        # May not need to do this -- just execute loop just above?
        self.__init__(**self.kwargs)

    @property
    def history(self):
        if hasattr(self, '_history'):
            pass
        elif hasattr(self, 'all_fluxes'):
            self._history = _sort_history(self.all_fluxes)
        else:
            raise NotImplemented('help!')

        return self._history

    @property
    def _subgen(self):
        if not hasattr(self, '_subgen_'):
            self._subgen_ = {}

            for popid, pop in enumerate(self.pops):
                gen = self.solver.generators[popid]

                if gen is None:
                    self._subgen_[popid] = None
                    continue

                # This is kludgey.
                if len(gen) == 1 and (not pop.is_src_lw):
                    self._subgen_[popid] = False
                else:
                    self._subgen_[popid] = True

        return self._subgen_

    def update_fluxes(self, popid=0):
        """
        Loop over flux generators and retrieve the next values.

        ..note:: Populations need not have identical redshift sampling.

        Returns
        -------
        Current redshift and dictionary of fluxes. Each element of the flux
        dictionary corresponds to a single population, and within that, there
        are separate lists for each sub-band over which we solve the RTE.

        """

        pop_generator = self.solver.generators[popid]

        # Skip approximate (or non-contributing) backgrounds
        if pop_generator is None:
            return None, None

        fluxes_by_band = []

        needs_flattening = self._subgen[popid]

        # For each population, the band is broken up into pieces
        for j, generator in enumerate(pop_generator):
            if generator is None:
                fluxes_by_band.append(None)
            else:
                z, f = next(generator)

                if not needs_flattening:
                    fluxes_by_band.append(f)
                    continue

                if z > self.pf['first_light_redshift']:
                    fluxes_by_band.append(np.zeros_like(flatten_flux(f)))
                else:
                    fluxes_by_band.append(flatten_flux(f))

        return z, fluxes_by_band

    def update_rate_coefficients(self, z, **kwargs):
        """
        Compute ionization and heating rate coefficients.

        Parameters
        ----------
        z : float
            Current redshift.

        Returns
        -------
        Dictionary of rate coefficients.

        """

        # Must compute rate coefficients from fluxes
        if self.solver.approx_all_pops:
            kwargs['fluxes'] = [None] * self.solver.Npops
            return self.solver.update_rate_coefficients(z, **kwargs)

        if not self._has_fluxes:
            self.run()

        if not self._has_coeff:

            kw = kwargs.copy()

            kw['return_rc'] = True

            # This chunk only gets executed once
            self._rc_tabs = [{} for i in range(self.solver.Npops)]
            # very similar chunk of code lives in update_fluxes...
            # could structure better, but i'm tired.

            fluxes = {i:None for i in range(self.solver.Npops)}
            for i, pop_generator in enumerate(self.solver.generators):

                kw['zone'] = self.pops[i].zone
                also = {}
                for sp in self.grid.absorbers:
                    also['{0!s}_{1!s}'.format(self.pops[i].zone, sp)] = 1.0
                kw.update(also)

                ##
                # Be careful with toy models
                ##
                if pop_generator is None:
                    zarr = self.z_unique
                else:
                    zarr = self.solver.redshifts[i]

                Nz = len(zarr)

                self._rc_tabs[i]['k_ion'] = np.zeros((Nz,
                        self.grid.N_absorbers))
                self._rc_tabs[i]['k_ion2'] = np.zeros((Nz,
                        self.grid.N_absorbers, self.grid.N_absorbers))
                self._rc_tabs[i]['k_heat'] = np.zeros((Nz,
                        self.grid.N_absorbers))
                self._rc_tabs[i]['k_heat_lya'] = np.zeros(Nz)
                self._rc_tabs[i]['Jc'] = np.zeros(Nz)
                self._rc_tabs[i]['Ji'] = np.zeros(Nz)
                self._rc_tabs[i]['Jlw'] = np.zeros(Nz)

                # Need to cycle through redshift here
                for _iz, redshift in enumerate(zarr):

                    # The use of "continue", rather than "break", is
                    # VERY important because redshifts are in ascending order
                    # at this point.
                    if redshift < self.pf['final_redshift']:
                        # This shouldn't ever happen...
                        continue

                    if redshift < self.pf['kill_redshift']:
                        continue

                    # Get fluxes if need be
                    if pop_generator is None:
                        kw['fluxes'] = None
                    else:
                        # Fluxes are in order of *descending* redshift!
                        fluxes[i] = self.history[i][Nz-_iz-1]
                        kw['fluxes'] = fluxes

                    self._rc_tabs[i]['Jc'][_iz] = self._f_Jc(redshift)
                    self._rc_tabs[i]['Ji'][_iz] = self._f_Ji(redshift)
                    self._rc_tabs[i]['Jlw'][_iz] = self._f_Jlw(redshift)

                    # This routine expects fluxes to be a dictionary, with
                    # the keys being population ID # and the elements lists
                    # of fluxes in each (sub-) band.
                    coeff = self.solver.update_rate_coefficients(redshift,
                        popid=i, **kw)

                    self._rc_tabs[i]['k_ion'][_iz,:] = \
                        coeff['k_ion'].copy()
                    self._rc_tabs[i]['k_ion2'][_iz,:] = \
                        coeff['k_ion2'].copy()
                    self._rc_tabs[i]['k_heat'][_iz,:] = \
                        coeff['k_heat'].copy()

            self._interp = [{} for i in range(self.solver.Npops)]
            for i, pop in enumerate(self.pops):
                if self.solver.redshifts[i] is None:
                    zarr = self.z_unique
                else:
                    zarr = self.solver.redshifts[i]

                # Create functions
                self._interp[i]['k_ion'] = \
                    [None for _i in range(self.grid.N_absorbers)]
                self._interp[i]['k_ion2'] = \
                    [[None,None,None] for _i in range(self.grid.N_absorbers)]
                self._interp[i]['k_heat'] = \
                    [None for _i in range(self.grid.N_absorbers)]

                self._interp[i]['Jc'] = interp1d(zarr,
                    self._rc_tabs[i]['Jc'], kind=self.pf['interp_all'],
                    bounds_error=False, fill_value=0.0)
                self._interp[i]['Ji'] = interp1d(zarr,
                    self._rc_tabs[i]['Ji'], kind=self.pf['interp_all'],
                    bounds_error=False, fill_value=0.0)
                self._interp[i]['Jlw'] = interp1d(zarr,
                    self._rc_tabs[i]['Jlw'], kind=self.pf['interp_all'],
                    bounds_error=False, fill_value=0.0)

                for j in range(self.grid.N_absorbers):
                    self._interp[i]['k_ion'][j] = \
                        interp1d(zarr, self._rc_tabs[i]['k_ion'][:,j],
                            kind=self.pf['interp_all'],
                            bounds_error=False, fill_value=0.0)
                    self._interp[i]['k_heat'][j] = \
                        interp1d(zarr, self._rc_tabs[i]['k_heat'][:,j],
                            kind=self.pf['interp_all'],
                            bounds_error=False, fill_value=0.0)

                    for k in range(self.grid.N_absorbers):
                        self._interp[i]['k_ion2'][j][k] = \
                            interp1d(zarr, self._rc_tabs[i]['k_ion2'][:,j,k],
                                kind=self.pf['interp_all'],
                                bounds_error=False, fill_value=0.0)

            self._has_coeff = True

            return self.update_rate_coefficients(z, **kwargs)

        else:

            to_return = \
            {
             'k_ion': np.zeros((1,self.grid.N_absorbers)),
             'k_ion2': np.zeros((1,self.grid.N_absorbers, self.grid.N_absorbers)),
             'k_heat': np.zeros((1,self.grid.N_absorbers)),
             'k_heat_lya': np.zeros(1),
             'Jc': np.zeros(1),
             'Ji': np.zeros(1),
            }

            for i, pop in enumerate(self.pops):

                if pop.zone != kwargs['zone']:
                    continue

                fset = self._interp[i]

                # Call interpolants, add 'em all up.
                this_pop = \
                {
                 'k_ion':  np.array([[fset['k_ion'][j](z) \
                    for j in range(self.grid.N_absorbers)]]),
                 'k_heat': np.array([[fset['k_heat'][j](z) \
                    for j in range(self.grid.N_absorbers)]]),
                 'k_heat_lya': np.zeros(1),
                 'Jc': fset['Jc'](z),
                 'Ji': fset['Ji'](z),
                 'Jlw': fset['Jlw'](z),
                }

                # Convert to rate coefficient
                for j, absorber in enumerate(self.grid.absorbers):
                    x = kwargs['{0!s}_{1!s}'.format(pop.zone, absorber)]

                    if self.pf['photon_counting']:
                        this_pop['k_ion'][0][j] /= x

                    # No helium for cgm, at least not this carefully
                    if pop.zone == 'cgm':
                        break

                tmp = np.zeros((self.grid.N_absorbers, self.grid.N_absorbers))
                for j in range(self.grid.N_absorbers):
                    for k in range(self.grid.N_absorbers):
                        tmp[j,k] = fset['k_ion2'][j][k](z)

                this_pop['k_ion2'] = np.array([tmp])

                if pop.zone == 'igm' and self.pf['lya_heating']:
                    this_pop['k_heat_lya'] = self.grid.hydr.get_lya_heating(z,
                        kwargs['igm_Tk'], this_pop['Jc'], this_pop['Ji'],
                        1.-kwargs['igm_h_1'])

                for key in to_return:
                    to_return[key] += this_pop[key]

            return to_return

    @property
    def z_unique(self):
        if not hasattr(self, '_z_unique'):
            _allz = []
            for i in range(self.solver.Npops):
                if self.solver.redshifts[i] is None:
                    continue

                _allz.append(self.solver.redshifts[i])

            if sum([item is not None for item in _allz]) == 0:
                dz = self.pf['fallback_dz']
                self._z_unique = np.arange(self.pf['final_redshift'],
                    self.pf['initial_redshift']+dz, dz)
            else:
                self._z_unique = np.unique(np.concatenate(_allz))

        return self._z_unique

    def get_uvb_tot(self, include_pops=None):
        """
        Sum the UV background (i.e., Ja, Jlw) over all populations.
        """

        if include_pops is None:
            include_pops = range(self.solver.Npops)

        #zarr = self.z_unique

        # Compute JLW to get estimate for Mmin^(k+1)
        _allz = []
        _f_Jc = []
        _f_Ji = []
        _f_Jlw = []
        for i, popid in enumerate(include_pops):

            if not (self.pops[popid].is_src_lw or self.pops[popid].is_src_lya):
                _allz.append(self.solver.redshifts[popid])
                _f_Jlw.append(lambda z: 0.0)
                _f_Jc.append(lambda z: 0.0)
                _f_Ji.append(lambda z: 0.0)
                continue

            _z, _Jc, _Ji, _Jlw = self.get_uvb(popid)

            _allz.append(_z)
            _f_Jlw.append(interp1d(_z, _Jlw, kind='linear'))
            _f_Jc.append(interp1d(_z, _Jc, kind='linear'))
            _f_Ji.append(interp1d(_z, _Ji, kind='linear'))

        zarr = self.z_unique

        Jlw = np.zeros_like(zarr)
        Jc = np.zeros_like(zarr)
        Ji = np.zeros_like(zarr)
        for i, popid in enumerate(include_pops):
            Jlw += _f_Jlw[i](zarr)
            Jc += _f_Jc[i](zarr)
            Ji += _f_Ji[i](zarr)

        return zarr, Jc, Ji, Jlw

    def _is_Mmin_converged(self, include_pops):

        # Need better long-term fix: Lya sources aren't necessarily LW
        # sources, if (for example) approx_all_pops = True.
        if not self.pf['feedback_LW']:
            # Will use all then
            include_pops = None
        elif include_pops is None:
            include_pops = range(self.solver.Npops)

        # Otherwise, grab all the fluxes
        zarr, Jc, Ji, Jlw = self.get_uvb_tot(include_pops)
        self._zarr = zarr

        Jc = np.maximum(Jc, 0.)
        Ji = np.maximum(Ji, 0.)
        Jlw = np.maximum(Jlw, 0.)

        self._Jc = Jc
        self._Ji = Ji
        self._Jlw = Jlw

        if not self.pf['feedback_LW']:
            return True

        # Instance of a population that "feels" the feedback.
        # Need for (1) initial _Mmin_pre value, and (2) setting ceiling
        pid = self.pf['feedback_LW_sfrd_popid']
        pop_fb = self.pops[self._lwb_sources.index(pid)]

        # Don't re-load Mmin guesses after first iteration
        if self.pf['feedback_LW_guesses'] is not None and self.count > 1:
            self.pops[pid]._loaded_guesses = True
            print('turning off ModelSet load', self.count, pid, self.pops[pid]._loaded_guesses)


        # Save last iteration's solution for Mmin(z)
        if self.count == 1:
            has_guess = False
            if self.pf['feedback_LW_guesses'] is not None:
                has_guess = True

                #_z_guess, _Mmin_guess = guess
                self._Mmin_pre = self.pops[pid].Mmin(zarr)
                self._Mmax_pre = self.pops[pid]._tab_Mmax

            else:
                self._Mmin_pre = np.min([self.pops[idnum].Mmin(zarr) \
                    for idnum in self._lwb_sources], axis=0)
                self._Mmax_pre = self.pops[pid]._tab_Mmax

            self._Mmin_bank = [self._Mmin_pre.copy()]
            self._Mmax_bank = [self._Mmax_pre.copy()]
            self._Jlw_bank = [Jlw]

            self.pf['feedback_LW_guesses'] = None

            ##
            # Quit right away if you say so. Note: Dangerous!
            ##
            if self.pf['feedback_LW_guesses_perfect'] and has_guess:
                self._Mmin_now = self._Mmin_pre
                self._sfrd_bank = [self.pops[pid].tab_sfrd_total]
                return True

        else:
            self._Mmin_pre = self._Mmin_now.copy()

        if self.pf['feedback_LW_sfrd_popid'] is not None:
            pid = self.pf['feedback_LW_sfrd_popid']
            if self.count == 1:
                self._sfrd_bank = [self.pops[pid].tab_sfrd_total.copy()]
            else:
                self._sfrd_bank.append(self.pops[pid].tab_sfrd_total.copy())
                pre = self._sfrd_bank[-2] * rhodot_cgs
                now = self._sfrd_bank[-1] * rhodot_cgs
                gt0 = np.logical_and(now > _tiny_sfrd, pre > _tiny_sfrd)

                zmin = max(self.pf['final_redshift'], self.pf['kill_redshift'])
                err = np.abs(pre - now) / now

                _ok = np.logical_and(gt0,
                    self.pops[pid].halos.tab_z > zmin)
                zra = self.pf['feedback_LW_tol_zrange']
                in_zra = \
                    np.logical_and(self.pops[pid].halos.tab_z >= zra[0],
                        self.pops[pid].halos.tab_z <= zra[1])

                self._ok = np.logical_and(_ok, in_zra)

                self._sfrd_rerr = err

                if not np.any(self._ok):

                    import matplotlib.pyplot as pl

                    pl.figure(1)
                    pl.semilogy(self.pops[pid].halos.tab_z, pre, ls='-')
                    pl.semilogy(self.pops[pid].halos.tab_z, now, ls='--')

                    #print(self._Mmin_bank[-1])
                    pl.figure(2)
                    pl.semilogy(self.pops[0].halos.tab_z, self.pops[0]._tab_Mmin, ls='-', color='k', alpha=0.5)
                    pl.semilogy(self.pops[0].halos.tab_z, self.pops[0]._tab_Mmax, ls='-', color='b', alpha=0.5)
                    pl.semilogy(self.pops[1].halos.tab_z, self.pops[1]._tab_Mmin, ls='--', color='k', alpha=0.5)
                    pl.semilogy(self.pops[1].halos.tab_z, self.pops[1]._tab_Mmax, ls='--', color='b', alpha=0.5)
                    #pl.semilogy(self.z_unique, self._Mmin_bank[-1], ls='--')

                    neg = now < 0
                    print(pid, now.size, neg.sum(), now)
                    raise ValueError("SFRD < 0!")

        self._Mmin_pre = np.maximum(self._Mmin_pre,
            pop_fb.halos.Mmin_floor(zarr))

        if np.any(np.isnan(Jlw)):
            Jlw[np.argwhere(np.isnan(Jlw))] = 0.0

        # Introduce time delay between Jlw and Mmin?
        # This doesn't really help with stability. In fact, it can make it
        # worse.
        if self.pf['feedback_LW_dt'] > 0: # pragma: no cover
            dt = self.pf['feedback_LW_dt']

            Jlw_dt = []
            for i, z in enumerate(zarr):

                J = 0.0
                for j, z2 in enumerate(zarr[i:]):
                    tlb = self.grid.cosm.LookbackTime(z, z2) / s_per_myr

                    if tlb < dt:
                        continue
                    z2p = zarr[i+j-1]
                    tlb_p = self.grid.cosm.LookbackTime(z, z2p) / s_per_myr
                    zdt = np.interp(dt, [tlb_p, tlb], [z2p, z2])

                    J = np.interp(zdt, zarr, Jlw, left=0.0, right=0.0)

                    break

                Jlw_dt.append(J)

            Jlw_dt = np.array(Jlw_dt)
        else:
            Jlw_dt = Jlw

        # Function to compute the next Mmin(z) curve
        # Use of Mmin here is for shielding prescription (if used),
        # but implicitly used since it set Jlw for this iteration.
        if self.pf['feedback_LW_zstart'] is not None:
            Jlw_dt[zarr > self.pf['feedback_LW_zstart']] = 0

        # Experimental
        if self.pf['feedback_LW_ramp'] > 0: # pragma: no cover
            ramp = self.pf['feedback_LW_ramp']

            nh = 0.0
            for i in self._lwb_sources:
                pop = self.pops[i]
                nh += pop._tab_nh_active

            nh *= cm_per_mpc**3

            # Interpolate to same redshift array as fluxes
            nh = np.interp(zarr, pop.halos.tab_z, nh)

            # Compute typical separation of halos
            ravg = nh**(-1./3.)
            tavg = ramp * ravg / (c / cm_per_mpc)
            tH = self.grid.cosm.HubbleTime(zarr)

            # Basically making the argument that the LW background
            # becomes uniform once the typical spacing between halos
            # is << the light travel time between halos.
            # Could introduce a multiplicative factor to control
            # this more finely...

            # A number from [0, 1] that quantifies how uniform the background is
            f_uni = 1. - np.maximum(np.minimum(tavg / tH, 1.), 0)

            Jlw_dt *= f_uni

        f_M = get_Mmin_func(zarr, Jlw_dt / 1e-21, self._Mmin_pre, **self.pf)

        # Use this on the next iteration, unless the 'mixup' parameters
        # are being used.
        Mnext = f_M(zarr)

        mfreq = self.pf['feedback_LW_mixup_freq']
        mdel = self.pf['feedback_LW_mixup_delay']

        # Set Mmin for the next iteration
        if mfreq > 0 and self.count >= mdel and \
           (self.count - mdel) % mfreq == 0:
            _Mmin_next = np.sqrt(np.product(self._Mmin_bank[-2:], axis=0))
        elif (self.count > 1) and (self.pf['feedback_LW_softening'] is not None):
            if self.pf['feedback_LW_softening'] in ['sqrt', 'gmean']:
                _Mmin_next = np.sqrt(Mnext * self._Mmin_pre)
            elif self.pf['feedback_LW_softening'] == 'mean':
                _Mmin_next = np.mean([Mnext, self._Mmin_pre], axis=0)
            elif self.pf['feedback_LW_softening'] == 'log10_mean':
                _Mmin_next = 10**np.mean([np.log10(Mnext),
                    np.log10(self._Mmin_pre)], axis=0)
            else:
                raise NotImplementedError('help')
        else:
            _Mmin_next = Mnext

        ##
        # Dealing with Mmin fluctuations
        if self.pf['feedback_LW_Mmin_monotonic'] and \
           (self.count % self.pf['feedback_LW_Mmin_afreq'] == 0): # pragma: no cover

            # Detect wiggles: expected behaviour is dMmin/dz < 0.
            # Zero pad at the end to recover correct length.
            dMmindz = np.concatenate((np.diff(_Mmin_next), [0]))

            # Break into positive and negative chunks.
            _x, _y = split_by_sign(zarr, dMmindz)

            # Go in order of ascending time (descending z)
            x = _x[-1::-1]
            y = _y[-1::-1]

            j = 0 # start index of a given chunk within full array of Mmin
            zrev = zarr[-1::-1]
            Mmin_new = _Mmin_next[-1::-1]
            for i, _chunk_ in enumerate(y):
                l = len(_chunk_)

                if (i == 0) or (_chunk_[0] <= 0):
                    j += l
                    continue

                # If we're here, it means Mmin is falling with redshift.

                k = j + l

                # Just replace with Mmin value before it last declined.
                # Guaranteed to be OK since we iterate from high-z to low.
                if self.pf['feedback_LW_Mmin_monotonic'] == 1:
                    Mmin_new[j:k] = Mmin_new[j-1]
                else:
                    raise NotImplemented('help')
                    # Interpolate to next point where Mmin > Mmin_problem
                    dx = zrev[j-1] - zrev[j-2]
                    dy = Mmin_new[j-1] - Mmin_new[j-2]
                    m = 2 * dy / dx

                    print('hey', dx, dy, m)
                    Mmin_guess = m * (zrev[k-1] - zrev[j-1]) + Mmin_new[j-1]
                    #while Mmin_guess > Mmin_new[k]:
                    #    m *= 0.9
                    #    Mmin_guess = m * (zrev[k] - zrev[j-1]) + Mmin_new[j-1]

                    Mmin_new[j:k] = Mmin_guess#m * (zrev[j:k] - zrev[j-1]) + Mmin_new[j-1]

                j += l

            _Mmin_next = Mmin_new[-1::-1]

        # Detect ripples first and only do this if we see some?
        elif (self.pf['feedback_LW_Mmin_smooth'] > 0) and \
           (self.count % self.pf['feedback_LW_Mmin_afreq'] == 0): # pragma: no cover

            s = self.pf['feedback_LW_Mmin_smooth']
            bc = int(s / 0.1)
            if bc % 2 == 0:
                bc += 1

            ztmp = np.arange(zarr.min(), zarr.max(), 0.1)
            Mtmp = np.interp(ztmp, zarr, np.log10(_Mmin_next))
            Ms = smooth(Mtmp, bc, kernel='boxcar')

            _Mmin_next = 10**np.interp(zarr, ztmp, Ms)

        elif (self.pf['feedback_LW_Mmin_fit'] > 0) and \
           (self.count % self.pf['feedback_LW_Mmin_afreq'] == 0): # pragma: no cover
            order = self.pf['feedback_LW_Mmin_fit']
            _Mmin_next = 10**np.polyval(np.polyfit(zarr, np.log10(_Mmin_next), order), zarr)

        # Need to apply Mmin floor
        _Mmin_next = np.maximum(_Mmin_next, pop_fb.halos.Mmin_floor(zarr))

        # Potentially impose ceiling on Mmin
        Tcut = self.pf['feedback_LW_Tcut']

        # Instance of a population that "feels" the feedback.
        # Just need access to a few HMF routines.
        Mmin_ceil = pop_fb.halos.VirialMass(zarr, Tcut)

        # Final answer.
        Mmin = np.minimum(_Mmin_next, Mmin_ceil)

        # Set new solution
        self._Mmin_now = Mmin.copy()
        # Save for our records (read: debugging)

        # Save Mmax too
        self._Mmax_now = self.pops[pid]._tab_Mmax.copy()

        ##
        # Compare Mmin of last two iterations.
        ##

        # Can't be converged after 1 iteration!
        if self.count == 1:
            return False

        if self.count < self.pf['feedback_LW_miniter']:
            return False

        if self.count >= self.pf['feedback_LW_maxiter']:
            return True

        converged = 1
        for quantity in ['Mmin', 'sfrd']:

            rtol = self.pf['feedback_LW_{!s}_rtol'.format(quantity)]
            atol = self.pf['feedback_LW_{!s}_atol'.format(quantity)]

            if rtol == atol == 0:
                continue

            if quantity == 'Mmin':
                pre, post = self._Mmin_pre, self._Mmin_now
            elif quantity == 'sfrd':
                pre, post = np.array(self._sfrd_bank[-2:]) * rhodot_cgs

            # Less stringent requirement, that mean error meet tolerance.
            if self.pf['feedback_LW_mean_err']:
                err_rel = np.abs((pre - post)) / post
                err_abs = np.abs(post - pre)

                ok = np.logical_and(np.isfinite(err_rel), self._ok)
                err_r_mean = np.mean(err_rel[ok==1])

                if rtol > 0:
                    if err_r_mean > rtol:
                        converged *= 0
                    elif err_r_mean < rtol and (atol == 0):
                        converged *= 1

                # Only make it here if rtol is satisfied or irrelevant
                if atol > 0:
                    if err_r_mean < atol:
                        converged *= 1
            # More stringent: that all Mmin values must have converged independently
            else:
                # Be a little careful: zeros will throw this off.
                # This is a little sketchy because some grid points
                # may go from 0 to >0 on back-to-back iterations, but in practice
                # there are so few that this isn't a real concern.
                gt0 = np.logical_and(pre > _tiny_sfrd, post > _tiny_sfrd)
                zmin = max(self.pf['final_redshift'], self.pf['kill_redshift'])

                pid = self.pf['feedback_LW_sfrd_popid']
                zarr = self.pops[pid].halos.tab_z

                if quantity == 'sfrd':
                    ok = np.logical_and(gt0, zarr > zmin)
                else:
                    ok = gt0

                # Check for convergence
                converged = np.allclose(pre[ok], post[ok], rtol=rtol, atol=atol)

                perfect = np.all(np.equal(pre[ok], post[ok]))

            perfect = np.all(np.equal(self._Mmin_pre, self._Mmin_now))

            # Nobody is perfect. Something is weird. Keep on keepin' on.
            # This is happening at iteration 3 for some reason...
            if perfect:
                converged = False

                if np.all(Jlw == 0):
                    raise ValueError('LWB == 0. Did you mean to include LW feedback?')

        if not converged:
            self._Mmin_bank.append(self._Mmin_now.copy())
            self._Mmax_bank.append(self._Mmax_now.copy())
            self._Jlw_bank.append(Jlw)

        return converged

    def get_uvb(self, popid):
        """
        Return Ly-a and LW background flux in units of erg/s/cm^2/Hz/sr.

        ..note:: This adds in line flux.

        """

        # Approximate sources
        if np.any(self.solver.solve_rte[popid]):
            z, E, flux = self.get_history(popid=popid, flatten=True)

            if self.pops[popid].is_src_ion_igm and self.pf['secondary_lya']:
                Ja = np.zeros_like(z) # placeholder
                Ji = np.zeros_like(z)
                Jc = np.zeros_like(z)
            elif self.pops[popid].is_src_lya:
                # Redshift is first dimension!
                l = np.argmin(np.abs(E - E_LyA))
                Jc = flux[:,l+1]
                Ja = flux[:,l]
                Ji = Ja - Jc
            else:
                Ja = np.zeros_like(z)
                Ji = np.zeros_like(z)
                Jc = np.zeros_like(z)

            # Find photons in LW band
            is_LW = np.logical_and(E >= 11.18, E <= E_LL)

            dnu = (E_LL - 11.18) / ev_per_hz

            # Need to do an integral to finish this one.
            Jlw = np.zeros_like(z)
        else:
            # Need a redshift array!
            z = self.z_unique
            Ja = np.zeros_like(z)
            Ji = np.zeros_like(z)
            Jc = np.zeros_like(z)
            Jlw = np.zeros_like(z)

        ##
        # Loop over redshift
        ##
        for i, redshift in enumerate(z):
            if not np.any(self.solver.solve_rte[popid]):
                Jc[i] = self.solver.LymanAlphaFlux(redshift, popid=popid)

                if self.pf['feedback_LW']:
                    Jlw[i] = self.solver.LymanWernerFlux(redshift, popid=popid)

                continue

            elif self.pops[popid].is_src_ion_igm and self.pf['secondary_lya']:
                for k, sp in enumerate(self.grid.absorbers):
                    Ji[i] += self.solver.volume.SecondaryLymanAlphaFlux(redshift,
                        species=k, popid=popid, fluxes={popid:flux[i]})

            # Convert to energy units, and per eV to prep for integral
            LW_flux = flux[i,is_LW] * E[is_LW] * erg_per_ev / ev_per_hz

            Jlw[i] = np.trapz(LW_flux, x=E[is_LW]) / dnu

        return z, Jc, Ji, Jlw

    def get_history(self, popid=0, flatten=False, today_only=False):
        """
        Grab data associated with a single population.

        Parameters
        ----------
        popid : int
            ID number for population of interest.
        flatten : bool
            For sawtooth calculations, the energies are broken apart into
            different bands which have different sizes. Set this to true
            if you just want a single array, rather than having the
            energies and fluxes broken apart by their band.

        Returns
        -------
        Tuple containing the redshifts, energies, and fluxes for the
        given population, in that order.

        if flatten == True:
            The energy array is 1-D.
            The flux array will have shape (z, E)
        else:
            The energies are stored as a list. The number of elements will
            be determined by how many sub-bands there are. Each element will
            be a list or an array, depending on whether or not there is a
            sawtooth component to that particular background.

        """

        hist = self.history[popid]

        # First, get redshifts. If not run "thru run", then they will
        # be in descending order so flip 'em.

        z = self.solver.redshifts[popid]
        zflip = z[-1::-1]
        zmin = min(z)
        #else:
        #    # This may change on the fly due to sub-cycling and such
        #    z = np.array(self.all_z).T[popid][-1::-1]

        if flatten:
            E = np.array(flatten_energies(self.solver.energies[popid]))

            # First loop is over redshift.
            f = np.zeros([len(z), E.size])

            # Looping over redshift, flatten energy, store.
            for i, flux in enumerate(hist):
                if today_only:
                    if zflip[i] != zmin:
                        continue

                fzflat = []
                # This is each 'super-'band, may be broken down into more
                # chunks, e.g., for sawtooth bands (hence the flattening)
                for j in range(len(self.solver.energies[popid])):

                    if not self.solver.solve_rte[popid][j]:
                        fzflat.extend(np.zeros_like(flatten_energies(self.solver.energies[popid][j])))
                    else:
                        fzflat.extend(flux[j])

                f[i] = np.array(fzflat)

            # "tr" = "to return"
            z_tr = z
            E_tr = E
            f_tr = np.array(f)[-1::-1,:]
        else:
            z_tr = z
            E_tr = self.solver.energies[popid]
            f_tr = hist[-1::-1][:]

        # We've flipped the flux array too since they are internally
        # kept in order of descending redshift.
        return z_tr, E_tr, f_tr

    def save(self, prefix, suffix='pkl', clobber=False):
        """
        Save results of calculation.

        Notes
        -----
        1) will save files as prefix.rfield.suffix.
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

        fn_1 = '{0!s}.fluxes.{1!s}'.format(prefix, suffix)
        fn_2 = '{0!s}.emissivities.{1!s}'.format(prefix, suffix)

        all_fn = [fn_1, fn_2]

        f_data = [self.get_history(i, flatten=True) for i in range(self.solver.Npops)]
        z = [f_data[i][0] for i in range(self.solver.Npops)]
        E = [f_data[i][1] for i in range(self.solver.Npops)]
        fl = [f_data[i][2] for i in range(self.solver.Npops)]

        all_data = [(z, E, fl),
            (self.solver.redshifts, self.solver.energies, self.solver.emissivities)]

        for i, data in enumerate(all_data):
            fn = all_fn[i]

            if os.path.exists(fn):
                if clobber:
                    os.remove(fn)
                else:
                    print(('{!s} exists! Set clobber=True to ' +\
                        'overwrite.').format(fn))
                    continue

            if suffix == 'pkl':
                write_pickle_file(data, fn, ndumps=1, open_mode='w',\
                    safe_mode=False, verbose=False)

            elif suffix in ['hdf5', 'h5']:
                raise NotImplementedError('no hdf5 support for this yet.')

            # ASCII format
            else:
                raise NotImplementedError('No ASCII support for this.')

            print('Wrote {!s}.'.format(fn))
