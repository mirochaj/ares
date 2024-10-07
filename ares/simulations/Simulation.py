import os
import copy
import pickle
import numpy as np
from types import FunctionType
from ..util import ParameterFile
from .Global21cm import Global21cm
from .PowerSpectrum21cm import PowerSpectrum21cm
from .MetaGalacticBackground import MetaGalacticBackground
from ..physics.Constants import cm_per_mpc, c, s_per_yr, erg_per_ev, \
    erg_per_s_per_nW, h_p, cm_per_m

class Simulation(object):
    def __init__(self, pf=None, pf_updates=None, **kwargs):
        """ Wrapper class designed to facilitate easy runs of any simulation. """

        if pf is None:
            assert kwargs is not None, \
                "Must provide parameters to initialize a Simulation!"

        self.kwargs = kwargs

        if pf is None:
            self.pf = ParameterFile(**kwargs)
        else:
            self.pf = pf

            ##
            # This is a sneaky way to not have to re-initialize an
            # entire ParameterFile if we're running lots of models.
            # Just need to remember that `self.pf` does not contain
            # population-specific parameters, so much parse parameter
            # names here and inject into correct element of self.pf.pfs,
            # i.e., the individual ParameterFile for each population.
            if pf_updates is not None:
                for par in pf_updates:
                    ib = par.find('{')
                    if ib == -1:
                        if self.pf['verbose']:
                            print(f"# Ignoring par={par} in updates.")
                        continue

                    popid = int(par[ib+1:ib+2])
                    newname = par.strip(f"{{{popid}}}")
                    self.pf.pfs[popid][newname] = pf_updates[par]

    @property
    def sim_gs(self):
        if not hasattr(self, '_sim_gs'):
            self._sim_gs = Global21cm(pf=self.pf, **self.kwargs)
        return self._sim_gs

    @sim_gs.setter
    def sim_gs(self, value):
        """ Set global 21cm instance by hand. """
        self._sim_gs = value

    @property
    def sim_ps(self):
        if not hasattr(self, '_sim_ps'):
            self._sim_ps = PowerSpectrum21cm(pf=self.pf, **self.kwargs)
            self._sim_ps.gs = self.sim_gs
        return self._sim_ps

    #@ps.setter
    #def ps(self, value):
    #    """ Set power spectrum 21cm instance by hand. """
    #    self._ps = value

    @property
    def history(self):
        if not hasattr(self, '_history'):
            self._history = {}
        return self._history

    @property
    def mean_intensity(self):
        if not hasattr(self, '_mean_intensity'):
            #self._mean_intensity = MetaGalacticBackground(**self.pf)
            self._mean_intensity = self.sim_gs.medium.field
        return self._mean_intensity

    @property
    def background_intensity(self):
        return self.mean_intensity

    def _cache_ebl(self, wave_units='mic', flux_units='SI', zlow=None):
        if not hasattr(self, '_cache_ebl_'):
            self._cache_ebl_ = {}

        # Could be clever and convert units here.
        if (wave_units, flux_units, zlow) in self._cache_ebl_:
            _data = self._cache_ebl_[(wave_units, flux_units, zlow)]
            return _data

        return None

    def get_ebl(self, wave_units='mic', flux_units='SI', pops=None,
        zlow=None):
        """
        Return the extragalactic background light (EBL) over all wavelengths.

        Parameters
        ----------
        wave_units : str
            Current options: 'eV', 'microns', 'Ang'
        flux_units : str
            Current options: 'cgs', 'SI'
        pops : list
            If supplied, should be a list of populations to be included, i.e.,
            their (integer) ID numbers (see `self.pops` attribute for list
            of objects).
        zlow : int, float
            If provided, will truncate integral over redshift so that the EBL
            includes only emission from sources at z >= zlow.

        .. note :: 'SI' units means nW / m^2 / sr, 'cgs' means erg/s/Hz/sr.

        Returns
        -------
        Dictionary containing EBL for each source population, with the ID
        number used as a dictionary key. Each element is a tuple containing
        the (observed energies (or wavelengths) in `wave_units`,
        observed fluxes in `flux_units`).

        """

        cached_result = self._cache_ebl(wave_units, flux_units, zlow)
        if cached_result is not None:
            data = cached_result
        else:
            data = {}

        if not self.background_intensity._run_complete:
            self.background_intensity.run()

        for i in range(len(self.pops)):
            if i in data:
                continue

            if pops is not None:
                if i not in pops:
                    continue

            if zlow is not None:
                zf = zlow
            else:
                zf = self.pops[i].zdead

            assert self.pops[i].pf['pop_mask'] is None, \
                "Turn off mask (via `pop_mask`) before computing mean EBL!"

            x, flux = self.mean_intensity.get_spectrum(zf=zf, popids=i,
                units=flux_units, xunits=wave_units)

            #if wave_units.lower() == 'ev':
            #    x = E
            #elif wave_units.lower().startswith('mic'):
            #    x = 1e4 * c / (E * erg_per_ev / h_p)
            #elif wave_units.lower().startswith('ang'):
            #    x = 1e8 * c / (E * erg_per_ev / h_p)
            #else:
            #    raise NotImplemented('Unrecognized `wave_units`={}'.format(
            #        wave_units
            #    ))

            data[i] = x, flux

        # Cache
        self._cache_ebl_[(wave_units, flux_units, zlow)] = data

        return data

    def get_ebl_ps(self, scales, waves, waves2=None, wave_units='mic',
        scale_units='ell', flux_units='SI', dimensionless=False, pops=None,
        include_inter_pop=True, **kwargs):
        """
        Compute power spectrum of EBL at some observed wavelength(s).

        Parameters
        ----------
        scales : int, float, np.ndarray
            Modes (or angular scales) of interest, depending on value of
            `scale_units`.
        waves : int, float, np.ndarray
            Wavelengths at which to compute power spectra in `wave_units`.
            Note that if 2-D, must have shape (number of bins, 2), in which
            case the power spectra will be computed in series of bandpasses.
        pops : list, tuple
            If provided, sets the ID numbers of populations that will be
            included in the model. In other words, any population *not* included
            in this list will be skipped. By default, this is None and all
            source populations defined by the parameters (self.pf) are
            included.
        include_inter_pop : bool
            This flag determines whether "inter-population cross terms" are
            included in the calculation.
        wave_units : str
            Current options: 'eV', 'microns', 'Ang'
        flux_units : str
            Current options: 'cgs', 'SI'
        scale_units : str
            Current options: 'arcmin', 'arcsec', 'degrees', 'ell'

        Optional keyword arguments
        --------------------------
        The `get_ps_obs` methods within ares.populations objects take a
        number of optional arguments that control the output. These include:

        include_1h : bool
            If False, exclude 1-halo term from calculation [Default: True]
        include_2h : bool
            If False, exclude 2-halo term from calculation [Default: True]
        include_shot : bool
            If False, exclude shot noise term from calculation [Default: True]


        Returns
        -------
        Tuple containing (scales, 2 pi / scales or l*l(+1),
            waves, power spectra).

        Note that the power spectra are return as 2-D arrays with shape
        (len(scales), len(waves))

        """

        # Make sure we do mean background first in case LW feedback is on.
        #if not self.mean_intensity._run_complete:
        #    self.mean_intensity.run()

        # Make sure things are arrays
        if type(scales) != np.ndarray:
            scales = np.array([scales])
        if type(waves) != np.ndarray:
            waves = np.array([waves])

        waves_is_2d = False
        if waves.ndim == 2:
            assert waves.shape[1] == 2, \
                "If `waves` is 2-D, must have shape (num waves, 2)."
            waves_is_2d = True

        # Prep scales
        if scale_units.lower() in ['l', 'ell']:
            scales_inv = np.sqrt(scales * (scales + 1))
            # Squared below hence the sqrt here.
        else:
            if scale_units.lower().startswith('deg'):
                scale_rad = scales * (np.pi / 180.)
            elif scale_units.lower() == 'arcmin':
                scale_rad = (scales / 60.) * (np.pi / 180.)
            elif scale_units.lower() == 'arcsec':
                scale_rad = (scales / 3600.) * (np.pi / 180.)
            else:
                raise NotImplemented(f"Don't recognize `scale_units`={scale_units}")

            scales_inv = 2 * np.pi / scale_rad

        if wave_units.lower().startswith('mic'):
            pass
        else:
            raise NotImplemented('help')

        # Do some error-handling if waves is 2-D: means the user provided
        # bandpasses instead of a set of wavelengths.

        if waves2 is None:
            waves2 = waves

        ps = np.zeros((len(self.pops), len(scales), len(waves)))
        px = np.zeros((len(self.pops), len(self.pops), len(scales), len(waves)))
        # Save contributing pieces

        # [optonal] Save redshift chunks
        ps_z = np.zeros((len(self.pops), len(self.pops),
            len(scales), len(waves), self.pops[0].halos.tab_z.size))

        # Loop over source populations and compute power spectrum.
        #
        for i, pop in enumerate(self.pops):

            # Honor user-supplied list of populations to include
            if pops is not None:
                if i not in pops:
                    continue

            for j, popx in enumerate(self.pops):
                # Avoid double counting
                if j > i:
                    break

                # Honor user-supplied list of populations to include
                if pops is not None:
                    if j not in pops:
                        continue

                for k, wave in enumerate(waves):
                    # Will default to 1h + 2h + shot
                    if j == i:
                        ps[i,:,k] = pop.get_ps_obs(scales,
                            wave_obs1=wave, wave_obs2=waves2[k],
                            scale_units=scale_units, **kwargs)
                        ps_z[i,i,:,k,:] = pop._ps_obs_integrand.copy()
                        continue

                    if not include_inter_pop:
                        continue

                    ##
                    # Cross terms only from here on
                    px[i,j,:,k] = pop.get_ps_obs(scales,
                        wave_obs1=wave, wave_obs2=waves2[k],
                        scale_units=scale_units, cross_pop=popx, **kwargs)
                    ps_z[i,j,:,k,:] = pop._ps_obs_integrand.copy()

                ##
                # Clear out some memory -- u(k|M) tabs can be big.
                #if hasattr(pop.halos, '_tab_u_nfw'):
                #    del pop.halos._tab_u_nfw

        ##
        # Increment `ps` with cross terms.
        # Convention is that fluctuations for population `i` includes
        # all crosses with
        ps += px.sum(axis=1)

        ##
        # Modify PS units before return
        if flux_units.lower() == 'si':
            ps *= cm_per_m**4 / erg_per_s_per_nW**2
            px *= cm_per_m**4 / erg_per_s_per_nW**2
            ps_z *= cm_per_m**4 / erg_per_s_per_nW**2
        elif flux_units.lower() == 'mjy':
            ps *= 1e17
            px *= 1e17
            ps_z *= 1e17

        if pops is None:
            hist = self.history # poke
            self._history['ps_nirb'] = scales, scales_inv, waves, ps

        if dimensionless:
            ps *= scales_inv[None,:,None]**2 / 2. / np.pi
            px *= scales_inv[None,:,None]**2 / 2. / np.pi

        self.ps_auto = ps
        self.ps_cross = px
        self.ps_zall = ps_z

        return scales, scales_inv, waves, ps

    @property
    def pops(self):
        return self.sim_gs.medium.field.pops

    #@property
    #def pops(self):
    #    if not hasattr(self, '_pops'):
    #        self._pops = CompositePopulation(pf=self.pf, cosm=self.cosm,
    #            **self._kwargs).pops

    #    return self._pops

    @property
    def grid(self):
        return self.sim_gs.medium.field.grid

    @property
    def hydr(self):
        if not hasattr(self, '_hydr'):
            if hasattr(self.grid, 'hydr'):
                self._hydr = self.grid.hydr
            else:
                self._hydr = Hydrogen(pf=self.pf, cosm=self.cosm, **self.pf)
        return self._hydr

    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            if hasattr(self.grid, 'cosm'):
                self._cosm = self.grid.cosm
            else:
                self._cosm = Cosmology(pf=self.pf, **self.pf)

        return self._cosm

    @property
    def halos(self):
        if not hasattr(self, '_halos'):
            self._halos = self.pops[0].halos
        return self._halos

    def run(self):
        """
        Run everything we can.
        """
        pass

    def run_ebl(self):
        hist = self.mean_intensity.run()
        if not self.mean_intensity._run_complete:
            self.mean_intensity.run()

    def get_21cm_gs(self):
        if '21cm_gs' not in self.history:
            self.sim_gs.run()
            self.history['21cm_gs'] = self.sim_gs.history

        return self.sim_gs

    def get_21cm_ps(self, z=None, k=None):
        if '21cm_ps' not in self.history:
            # Allow user to specify (z, k) if they want
            self.sim_ps.run(z=z, k=k)#(z, k)
            self.history['21cm_ps'] = self.sim_ps.history

        return self.sim_ps

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

        self.sim_gs.save(prefix, clobber=clobber, fields=fields)

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
