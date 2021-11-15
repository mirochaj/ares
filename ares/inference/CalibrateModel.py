"""

CalibrateModel.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Wed 13 Feb 2019 17:11:07 EST

Description:

"""

import os
import numpy as np
from ..util import read_lit
from .ModelFit import ModelFit
from ..simulations import Global21cm
from ..util import ParameterBundle as PB
from .FitGlobal21cm import FitGlobal21cm
from ..populations.GalaxyCohort import GalaxyCohort
from .FitGalaxyPopulation import FitGalaxyPopulation
from ..populations.GalaxyEnsemble import GalaxyEnsemble

try:
    from distpy import DistributionSet
    from distpy import UniformDistribution
except ImportError:
    pass

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

_zcal_lf = [3.8, 4.9, 5.9, 6.9, 7.9, 10.]
_zcal_smf = [3, 4, 5, 6, 7, 8]
_zcal_beta = [4, 5, 6, 7]

acceptable_sfe_params = ['slope-low', 'slope-high', 'norm', 'peak']
acceptable_dust_params = ['norm', 'slope', 'peak', 'fcov', 'yield', 'scatter',
    'kappa', 'slope-high', 'growth']

class CalibrateModel(object): # pragma: no cover
    """
    Convenience class for calibrating galaxy models to UVLFs and/or SMFs.
    """
    def __init__(self, fit_lf=[5.9], fit_smf=False, fit_beta=False,
        fit_gs=None, idnum=0, add_suffix=True, ztol=0.21,
        free_params_sfe=[], zevol_sfe=[],
        include_fshock=False, include_scatter_mar=False, name=None,
        include_dust='var_beta', include_fgrowth=False,
        include_fduty=False, zevol_fduty=False, include_kappa=False,
        zevol_fshock=False, zevol_dust=False, free_params_dust=[],
        save_lf=True, save_smf=False, save_sam=False, include_fdtmr=False,
        save_sfrd=False, save_beta=False, save_dust=False, zmap={},
        monotonic_beta=False):
        """
        Calibrate a galaxy model to available data.

        .. note :: All the `include_*` parameters control what goes into our
            base_kwargs, while the `free_params_*` parameters control what
            we allow to vary in the fit.

        Parameters
        ----------
        fit_lf : bool
            Use available luminosity function measurements?
        fit_beta : bool
            Use available UV colour-magnitude measurements?
        fit_smf : bool
            Use available stellar mass function measurements?
        fit_gs : tuple
            Use constraints on global 21-cm signal?
            If not None, this should be (frequencies / MHz, dTb / mK, err / mK).

        idnum : int
            If model being calibrated has multiple source populations, this is
            the ID number of the one containing luminosity functions etc.

        zevol_sfe_norm : bool
            Allow redshift evolution in the normalization of the SFE?
        zevol_sfe_peak : bool
            Allow redshift evolution in the where the SFE peaks (in mass)?
        zevol_sfe_shape: bool
            Allow redshift evolution in the power-slopes of SFE?

        clobber : bool
            Overwrite existing data outputs?

        """

        self.name = name             # optional additional prefix
        self.add_suffix = add_suffix
        self.fit_lf = fit_lf
        self.fit_smf = fit_smf
        self.fit_gs = fit_gs
        self.fit_beta = fit_beta
        self.idnum = idnum
        self.zmap = zmap
        self.ztol = ztol
        self.monotonic_beta = monotonic_beta

        self.include_fshock = int(include_fshock)
        self.include_scatter_mar = int(include_scatter_mar)

        self.include_dust = include_dust
        self.include_fgrowth = include_fgrowth
        self.include_fduty = include_fduty
        self.include_fdtmr = include_fdtmr
        self.include_kappa = include_kappa

        # Set SFE free parameters
        self.free_params_sfe = free_params_sfe
        for par in self.free_params_sfe:
            if par in acceptable_sfe_params:
                continue

            raise ValueError("Unrecognized SFE param: {}".format(par))

        # What's allowed to vary with redshift?
        if zevol_sfe is None:
            self.zevol_sfe = []
        elif zevol_sfe == 'all':
            self.zevol_sfe = free_params_sfe
        else:
            self.zevol_sfe = zevol_sfe

        # Set SFE free parameters
        self.free_params_dust = free_params_dust
        for par in self.free_params_dust:
            if par in acceptable_dust_params:
                continue

            raise ValueError("Unrecognized dust param: {}".format(par))

        # What's allowed to vary with redshift?
        if zevol_dust is None:
            self.zevol_dust = []
        elif zevol_dust == 'all':
            self.zevol_dust = free_params_dust
        else:
            self.zevol_dust = zevol_dust

        self.zevol_fduty = zevol_fduty

        self.save_lf = int(save_lf)
        self.save_smf = int(save_smf)
        self.save_sam = int(save_sam)
        self.save_sfrd = int(save_sfrd)
        self.save_beta = bool(save_beta) if save_beta in [0, 1, True, False] \
            else int(save_beta)
        self.save_dust = int(save_dust)

    def get_zstr(self, vals, okvals):
        """
        Make a string showing the redshifts we're calibrating to for some
        quantity.
        """
        zcal = []
        for z in okvals:
            if z not in vals:
                continue

            zcal.append(z)

        zs = ''
        for z in zcal:
            zs += '%i_' % round(z)
        zs = zs.rstrip('_')

        return zs

    @property
    def prefix(self):
        """
        Generate output filename.
        """

        s = ''
        if self.fit_lf:
            s += 'lf_' + self.get_zstr(self.fit_lf, _zcal_lf) + '_'
        if self.fit_smf:
            s += 'smf_' + self.get_zstr(self.fit_smf, _zcal_smf) + '_'
        if self.fit_beta:
            s += 'beta_' + self.get_zstr(self.fit_beta, _zcal_beta) + '_'
        if self.fit_gs:
            s += 'gs_{0:.0f}_{0:.0f}_'.format(self.fit_gs[0].min(),
                self.fit_gs[0].max())

        if self.name is not None:
            if self.add_suffix:
                s = self.name + '_' + s
            else:
                s = self.name

        if rank == 0:
            print("# Will save to files with prefix {}.".format(s))

        return s

    @property
    def parameters(self):
        if not hasattr(self, '_parameters'):

            if self.Npops > 1:
                _suff = '{{{}}}'.format(self.idnum)
            else:
                _suff = ''

            free_pars = []
            guesses = {}
            is_log = []
            jitter = []
            ps = DistributionSet()

            # Normalization of SFE
            if 'norm' in self.free_params_sfe:
                free_pars.append('pq_func_par0[0]{}'.format(_suff))
                guesses['pq_func_par0[0]{}'.format(_suff)] = -1.5
                is_log.extend([True])
                jitter.extend([0.1])
                ps.add_distribution(UniformDistribution(-7, 1.),
                    'pq_func_par0[0]{}'.format(_suff))

                if 'norm' in self.zevol_sfe:
                    free_pars.append('pq_func_par6[0]{}'.format(_suff))
                    guesses['pq_func_par6[0]{}'.format(_suff)] = 0.
                    is_log.extend([False])
                    jitter.extend([0.1])
                    ps.add_distribution(UniformDistribution(-3, 3.),
                        'pq_func_par6[0]{}'.format(_suff))

            # Peak mass
            if 'peak' in self.free_params_sfe:
                free_pars.append('pq_func_par1[0]{}'.format(_suff))
                guesses['pq_func_par1[0]{}'.format(_suff)] = 11.5
                is_log.extend([True])
                jitter.extend([0.1])
                ps.add_distribution(UniformDistribution(9., 13.),
                    'pq_func_par1[0]{}'.format(_suff))

                if 'peak' in self.zevol_sfe:
                    free_pars.append('pq_func_par7[0]{}'.format(_suff))
                    guesses['pq_func_par7[0]{}'.format(_suff)] = 0.
                    is_log.extend([False])
                    jitter.extend([2.])
                    ps.add_distribution(UniformDistribution(-6, 6.),
                        'pq_func_par7[0]{}'.format(_suff))

            # Slope at low-mass side of peak
            if 'slope-low' in self.free_params_sfe:
                free_pars.append('pq_func_par2[0]{}'.format(_suff))
                guesses['pq_func_par2[0]{}'.format(_suff)] = 0.66
                is_log.extend([False])
                jitter.extend([0.1])
                ps.add_distribution(UniformDistribution(0.0, 1.5),
                    'pq_func_par2[0]{}'.format(_suff))

                # Allow to evolve with redshift?
                if 'slope-low' in self.zevol_sfe:
                    free_pars.append('pq_func_par8[0]{}'.format(_suff))
                    guesses['pq_func_par8[0]{}'.format(_suff)] = 0.
                    is_log.extend([False])
                    jitter.extend([0.1])
                    ps.add_distribution(UniformDistribution(-3, 3.),
                        'pq_func_par8[0]{}'.format(_suff))

            # Slope at high-mass side of peak
            if 'slope-high' in self.free_params_sfe:
                free_pars.append('pq_func_par3[0]{}'.format(_suff))

                guesses['pq_func_par3[0]{}'.format(_suff)] = -0.3

                is_log.extend([False])
                jitter.extend([0.1])
                ps.add_distribution(UniformDistribution(-3., 0.3),
                    'pq_func_par3[0]{}'.format(_suff))

                # Allow to evolve with redshift?
                if 'slope-high' in self.zevol_sfe:
                    free_pars.append('pq_func_par9[0]{}'.format(_suff))
                    guesses['pq_func_par9[0]{}'.format(_suff)] = 0.
                    is_log.extend([False])
                    jitter.extend([0.1])
                    ps.add_distribution(UniformDistribution(-6, 6.),
                        'pq_func_par9[0]{}'.format(_suff))

            ##
            # fduty
            ##
            if self.include_fduty:
                # Normalization of SFE
                free_pars.extend(['pq_func_par0[40]', 'pq_func_par2[40]'])
                guesses['pq_func_par0[40]'] = 0.5
                guesses['pq_func_par2[40]'] = 0.25
                is_log.extend([False, False])
                jitter.extend([0.2, 0.2])
                ps.add_distribution(UniformDistribution(0., 1.), 'pq_func_par0[40]')
                ps.add_distribution(UniformDistribution(-2., 2.), 'pq_func_par2[40]')

                if self.zevol_fduty:
                    free_pars.append('pq_func_par4[40]')
                    guesses['pq_func_par4[40]'] = 0.
                    is_log.extend([False])
                    jitter.extend([0.1])
                    ps.add_distribution(UniformDistribution(-3, 3.), 'pq_func_par4[40]')

            ##
            # DUST REDDENING
            ##
            if self.include_dust in ['screen', 'screen-dpl']:

                if 'norm' in self.free_params_dust:

                    free_pars.append('pq_func_par0[22]')

                    if 'slope-high' not in self.free_params_dust:
                        guesses['pq_func_par0[22]'] = 2.4
                    else:
                        guesses['pq_func_par0[22]'] = 1.2

                    is_log.extend([False])
                    jitter.extend([0.1])
                    ps.add_distribution(UniformDistribution(0.01, 10.), 'pq_func_par0[22]')

                    if 'norm' in self.zevol_dust:
                        assert self.include_dust == 'screen'
                        # If screen-dpl need to change parameter number!
                        free_pars.append('pq_func_par4[22]')
                        guesses['pq_func_par4[22]'] = 0.
                        is_log.extend([False])
                        jitter.extend([0.5])
                        ps.add_distribution(UniformDistribution(-2., 2.), 'pq_func_par4[22]')

                if 'slope' in self.free_params_dust:
                    free_pars.append('pq_func_par2[22]')
                    guesses['pq_func_par2[22]'] = 0.5
                    is_log.extend([False])
                    jitter.extend([0.05])
                    ps.add_distribution(UniformDistribution(0, 2.), 'pq_func_par2[22]')

                if 'slope-high' in self.free_params_dust:
                    assert self.include_dust == 'screen-dpl'
                    free_pars.append('pq_func_par3[22]')
                    guesses['pq_func_par3[22]'] = 0.5
                    is_log.extend([False])
                    jitter.extend([0.05])
                    ps.add_distribution(UniformDistribution(-1.0, 2.), 'pq_func_par3[22]')

                    if 'slope-high' in self.zevol_dust:
                        raise NotImplemented('help')

                if 'peak' in self.free_params_dust:
                    assert self.include_dust == 'screen-dpl'

                    free_pars.append('pq_func_par1[22]')
                    guesses['pq_func_par1[22]'] = 11.
                    is_log.extend([True])
                    jitter.extend([0.2])
                    ps.add_distribution(UniformDistribution(9., 13.), 'pq_func_par1[22]')

                    if 'peak' in self.zevol_dust:
                        raise NotImplemented('help')
                        free_pars.append('pq_func_par2[24]')
                        guesses['pq_func_par2[24]'] = 0.0
                        is_log.extend([False])
                        jitter.extend([0.5])
                        ps.add_distribution(UniformDistribution(-2., 2.), 'pq_func_par2[24]')

                if 'yield' in self.free_params_dust:

                    assert self.include_fdtmr

                    free_pars.extend(['pq_func_par0[50]', 'pq_func_par2[50]'])
                    guesses['pq_func_par0[50]'] = 0.4
                    guesses['pq_func_par2[50]'] = 0.
                    is_log.extend([False, False])
                    jitter.extend([0.1, 0.2])
                    ps.add_distribution(UniformDistribution(0., 1.0), 'pq_func_par0[50]')
                    ps.add_distribution(UniformDistribution(-2., 2.), 'pq_func_par2[50]')

                    if 'yield' in self.zevol_dust:
                        free_pars.append('pq_func_par4[50]')
                        guesses['pq_func_par4[50]'] = 0.0
                        is_log.extend([False])
                        jitter.extend([0.5])
                        ps.add_distribution(UniformDistribution(-3., 3.), 'pq_func_par4[50]')

                if 'growth' in self.free_params_dust:

                    assert self.include_fgrowth

                    free_pars.extend(['pq_func_par0[60]', 'pq_func_par2[60]'])
                    guesses['pq_func_par0[60]'] = 11.
                    guesses['pq_func_par2[60]'] = 0.
                    is_log.extend([True, False])
                    jitter.extend([0.5, 0.2])
                    ps.add_distribution(UniformDistribution(7., 14.), 'pq_func_par0[60]')
                    ps.add_distribution(UniformDistribution(-2., 2.), 'pq_func_par2[60]')

                    if 'growth' in self.zevol_dust:
                        free_pars.append('pq_func_par4[60]')
                        guesses['pq_func_par4[60]'] = 0.
                        is_log.extend([False])
                        jitter.extend([0.5])
                        ps.add_distribution(UniformDistribution(-4., 4.), 'pq_func_par4[60]')


                if 'scatter' in self.free_params_dust:
                    free_pars.extend(['pq_func_par0[33]'])
                    if 'slope-high' not in self.free_params_dust:
                        guesses['pq_func_par0[33]'] = 0.1
                    else:
                        guesses['pq_func_par0[33]'] = 0.05
                    is_log.extend([False])
                    jitter.extend([0.05])
                    ps.add_distribution(UniformDistribution(0., 0.6), 'pq_func_par0[33]')

                    if 'scatter-slope' in self.free_params_dust:
                        free_pars.extend(['pq_func_par2[33]'])
                        guesses['pq_func_par2[33]'] = 0.
                        is_log.extend([False])
                        jitter.extend([0.1])
                        ps.add_distribution(UniformDistribution(-2., 2.), 'pq_func_par2[33]')

                    if 'scatter' in self.zevol_dust:
                        free_pars.append('pq_func_par4[33]')
                        guesses['pq_func_par4[33]'] = 0.0
                        is_log.extend([False])
                        jitter.extend([0.5])
                        ps.add_distribution(UniformDistribution(-2., 2.), 'pq_func_par4[33]')


                if 'kappa' in self.free_params_dust:
                    free_pars.extend(['pq_func_par4[20]', 'pq_func_par6[20]'])
                    guesses['pq_func_par4[20]'] = 0.0
                    guesses['pq_func_par6[20]'] = 0.0
                    is_log.extend([False, False])
                    jitter.extend([0.2, 0.2])
                    ps.add_distribution(UniformDistribution(-3, 3.), 'pq_func_par4[20]')
                    ps.add_distribution(UniformDistribution(-2, 2.), 'pq_func_par6[20]')

                    if 'kappa' in self.zevol_dust:
                        raise NotImplemented('Cannot do triply nested PQs.')

            # Set the attributes
            self._parameters = free_pars
            self._guesses = guesses
            self._is_log = is_log
            self._jitter = jitter
            self._priors = ps

        return self._parameters

    @property
    def guesses(self):
        if not hasattr(self, '_guesses'):
            tmp = self.parameters
        return self._guesses

    @guesses.setter
    def guesses(self, value):
        if not hasattr(self, '_guesses'):
            tmp = self.parameters

        print("Revising default guessses...")
        self._guesses.update(value)

    @property
    def jitter(self):
        if not hasattr(self, '_jitter'):
            tmp = self.parameters
        return self._jitter

    @jitter.setter
    def jitter(self, value):
        self._jitter = value

    @property
    def is_log(self):
        if not hasattr(self, '_is_log'):
            tmp = self.parameters
        return self._is_log

    @is_log.setter
    def is_log(self, value):
        self._is_log = value

    @property
    def priors(self):
        if not hasattr(self, '_priors'):
            tmp = self.parameters
        return self._priors

    @priors.setter
    def priors(self, value):
        self._priors = value

    @property
    def blobs(self):

        ##
        # First: some generic redshifts, magnitudes, masses.
        redshifts = np.array([4, 6, 8, 10]) # generic

        if self.fit_lf:
            if 'lf' in self.zmap:
                red_lf = np.sort([item for item in self.zmap['lf'].values()])
            else:
                red_lf = np.array(self.fit_lf)
        else:
            red_lf = redshifts

        if self.fit_smf:
            if 'smf' in self.zmap:
                raise NotImplemented('help')
            red_smf = np.array(self.fit_smf)
            # Default to saving LF at same redshifts if not specified otherwise.
            if not self.fit_lf:
                red_lf = red_smf
        else:
            red_smf = red_lf

        if self.fit_beta:
            red_beta = np.array(self.fit_beta)
        else:
            red_beta = red_lf

        MUV = np.arange(-26, 5., 0.5)
        Mh = np.logspace(7, 13, 61)
        Ms = np.arange(7, 13.25, 0.25)

        ##
        # Now, start assembling blobs

        # Account for different location of population instance if
        # fit runs an ares.simulations calculation. Just GS option now.
        if self.fit_gs is not None:
            _pref = 'pops[{}].'.format(self.idnum)
        else:
            _pref = ''

        # For things like SFE, fduty, etc., need to tap into `guide`
        # attribute when using GalaxyEnsemble.
        if self.use_ensemble:
            _pref_g = _pref + 'guide.'
        else:
            _pref_g = _pref

        # Always save the UVLF
        blob_n = ['galaxy_lf']
        blob_i = [('z', red_lf), ('bins', MUV)]
        blob_f = ['{}get_lf'.format(_pref)]

        blob_pars = \
        {
         'blob_names': [blob_n],
         'blob_ivars': [blob_i],
         'blob_funcs': [blob_f],
         'blob_kwargs': [None],
        }

        blob_n = ['fstar']
        blob_i = [('z', redshifts), ('Mh', Mh)]
        blob_f = ['{}fstar'.format(_pref_g)]

        blob_pars['blob_names'].append(blob_n)
        blob_pars['blob_ivars'].append(blob_i)
        blob_pars['blob_funcs'].append(blob_f)
        blob_pars['blob_kwargs'].append(None)

        if self.include_fduty:
            blob_n = ['fduty']
            blob_i = [('z', redshifts), ('Mh', Mh)]
            blob_f = ['{}fduty'.format(_pref_g)]

            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(blob_i)
            blob_pars['blob_funcs'].append(blob_f)
            blob_pars['blob_kwargs'].append(None)

        if self.include_fdtmr:
            blob_n = ['fyield']
            blob_i = [('z', redshifts), ('Mh', Mh)]
            blob_f = ['{}dust_yield'.format(_pref_g)]


            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(blob_i)
            blob_pars['blob_funcs'].append(blob_f)
            blob_pars['blob_kwargs'].append(None)

        # SAM stuff
        if self.save_sam:
            blob_n = ['SFR', 'SMHM']
            blob_i = [('z', redshifts), ('Mh', Mh)]

            if self.use_ensemble:
                blob_f = ['guide.SFR', 'SMHM']
            else:
                blob_f = ['{}SFR'.format(_pref), 'SMHM']

            blob_k = [{}, {'return_mean_only': True}]

            if 'pop_dust_yield' in self.base_kwargs:
                if self.base_kwargs['pop_dust_yield'] != 0:
                    blob_n.append('Md')
                    blob_f.append('XMHM')
                    blob_k.append({'return_mean_only': True, 'field': 'Md'})

            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(blob_i)
            blob_pars['blob_funcs'].append(blob_f)
            blob_pars['blob_kwargs'].append(blob_k)

        # SMF
        if self.save_smf:
            blob_n = ['galaxy_smf']
            blob_i = [('z', red_smf), ('bins', Ms)]

            blob_f = ['StellarMassFunction']

            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(blob_i)
            blob_pars['blob_funcs'].append(blob_f)
            blob_pars['blob_kwargs'].append(None)

        # Covering factor and scale length
        if self.save_dust:
            blob_n = ['dust_scale']
            blob_i = [('z', redshifts), ('Mh', Mh)]
            blob_f = ['guide.dust_scale']

            if type(self.base_kwargs['pop_dust_yield']) == str:
                blob_n.append('dust_yield')
                blob_f.append('guide.dust_yield')

            if 'pop_dust_scatter' in self.base_kwargs:
                if type(self.base_kwargs['pop_dust_scatter'] == str):
                    blob_n.append('sigma_d')
                    blob_f.append('guide.dust_scatter')

            if 'pop_dust_growth' in self.base_kwargs:
                if type(self.base_kwargs['pop_dust_growth'] == str):
                    blob_n.append('fgrowth')
                    blob_f.append('guide.dust_growth')

            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(blob_i)
            blob_pars['blob_funcs'].append(blob_f)
            blob_pars['blob_kwargs'].append(None)

        # MUV-Beta
        if self.save_beta != False:

            Mbins = np.arange(-30, -10, 1.0)

            # This is fast
            blob_n = ['AUV']
            blob_i = [('z', red_beta), ('MUV', MUV)]
            blob_f = ['AUV']

            blob_k = [{'return_binned': True,
                'magbins': Mbins, 'Mwave': 1600.}]

            _b14 = read_lit('bouwens2014')
            filt_hst = {4: _b14.filt_shallow[4], 5: _b14.filt_shallow[5],
                6: _b14.filt_shallow[6], 7: _b14.filt_deep[7]}

            kw_hst = {'cam': ('wfc', 'wfc3'), 'filters': filt_hst,
                'dlam':20., 'rest_wave': None, 'return_binned': True,
                'Mbins': Mbins, 'Mwave': 1600.}

            blob_f.extend(['Beta'])
            blob_n.extend(['beta_hst'])
            blob_k.extend([kw_hst])

            # Save also the geometric mean of photometry as a function
            # of a magnitude at fixed rest wavelength.
            #kw_mag = {'cam': ('wfc', 'wfc3'), 'filters': filt_hst, 'dlam':20.}
            #blob_n.append('MUV_gm')
            #blob_f.append('Magnitude')
            #blob_k.append(kw_mag)

            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(blob_i)
            blob_pars['blob_funcs'].append(blob_f)
            blob_pars['blob_kwargs'].append(blob_k)

        # Cosmic SFRD
        if self.save_sfrd:
            blob_n = ['sfrd']
            blob_i = [('z', np.arange(3.5, 30.1, 0.1))]
            blob_f = ['SFRD']

            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(blob_i)
            blob_pars['blob_funcs'].append(blob_f)
            blob_pars['blob_kwargs'].append(None)

        # Reionization stuff
        if self.fit_gs is not None:
            blob_n = ['tau_e', 'z_B', 'dTb_B', 'z_C', 'dTb_C',
                'z_D', 'dTb_D']
            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(None)
            blob_pars['blob_funcs'].append(None)
            blob_pars['blob_kwargs'].append(None)

            blob_n = ['cgm_h_2', 'igm_Tk', 'dTb']
            blob_i = [('z', np.arange(5.5, 35.1, 0.1))]

            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(blob_i)
            blob_pars['blob_funcs'].append(None)
            blob_pars['blob_kwargs'].append(None)


        return blob_pars

    @property
    def use_ensemble(self):
        return self.base_kwargs['pop_sfr_model'] == 'ensemble'

    @property
    def base_kwargs(self):
        if not hasattr(self, '_base_kwargs'):
            raise AttributeError("Must set `base_kwargs` by hand!")
        return self._base_kwargs

    @base_kwargs.setter
    def base_kwargs(self, value):
        self._base_kwargs = PB(**value)

    def update_kwargs(self, **kwargs):
        bkw = self.base_kwargs
        self._base_kwargs.update(kwargs)
        self.Npops = self._base_kwargs.Npops

    @property
    def Npops(self):
        if not hasattr(self, '_Npops'):
            assert isinstance(self.base_kwargs, PB)
            self._Npops = max(self.base_kwargs.Npops, 1)

        return self._Npops

    @Npops.setter
    def Npops(self, value):
        if hasattr(self, '_Npops'):
            if self.base_kwargs.Npops != self._Npops:
                print("Updated Npops from {} to {}".format(self._Npops,
                    self.base_kwargs.Npops))
                self._Npops = max(self.base_kwargs.Npops, 1)
        else:
            self._Npops = max(self.base_kwargs.Npops, 1)

    def get_initial_walker_position(self):
        guesses = {}
        for i, par in enumerate(self.parameters):
            if self.is_log[i]:
                guesses[par] = 10**self.guesses[par]
            else:
                guesses[par] = self.guesses[par]

        return guesses

    def run(self, steps, burn=0, nwalkers=None, save_freq=10, prefix=None,
        debug=True, restart=False, clobber=False, verbose=True,
        cache_tricks=False, burn_method=0, recenter=False,
        checkpoints=True):
        """
        Create a fitter class and run the fit!
        """

        if prefix is None:
            prefix = self.prefix

        # Setup LF fitter
        fitter_lf = FitGalaxyPopulation()
        fitter_lf.zmap = self.zmap
        fitter_lf.ztol = self.ztol
        fitter_lf.monotonic_beta = self.monotonic_beta

        data = []
        include = []
        fit_galaxies = False
        if self.fit_lf:
            include.append('lf')
            data.extend(['bouwens2015', 'oesch2018'])
            fit_galaxies = True
        if self.fit_smf:
            include.append('smf')
            data.append('song2016')
            fit_galaxies = True
        if self.fit_beta:
            include.append('beta')
            data.extend(['bouwens2014'])
            fit_galaxies = True

        # Must be before data is set
        fitter_lf.redshifts = {'lf': self.fit_lf, 'smf': self.fit_smf,
            'beta': self.fit_beta}
        fitter_lf.include = include

        fitter_lf.data = data

        if self.fit_gs is not None:
            freq, dTb, err = self.fit_gs
            fitter_gs = FitGlobal21cm()
            fitter_gs.frequencies = freq
            fitter_gs.data = dTb
            fitter_gs.error = err

        ##
        # Stitch together parameters
        ##
        pars = self.base_kwargs
        pars.update(self.blobs)

        # Master fitter
        fitter = ModelFit(**pars)

        if fit_galaxies:
            fitter.add_fitter(fitter_lf)

        if self.fit_gs is not None:
            fitter.add_fitter(fitter_gs)

        if self.fit_gs is not None:
            fitter.simulator = Global21cm
        elif self.use_ensemble:
            fitter.simulator = GalaxyEnsemble
        else:
            fitter.simulator = GalaxyCohort

        fitter.parameters = self.parameters
        fitter.is_log = self.is_log
        fitter.debug = debug
        fitter.verbose = verbose

        fitter.checkpoint_append = not checkpoints

        fitter.prior_set = self.priors

        if nwalkers is None:
            nw = 2 * len(self.parameters)
            if rank == 0:
                print("# Running with {} walkers.".format(nw))
        else:
            nw = nwalkers

        fitter.nwalkers = nw

        # Set initial positions of walkers

        # Important the jitter comes first!
        fitter.jitter = self.jitter
        if (not restart):
            fitter.guesses = self.guesses

        if cache_tricks:
            fitter.save_hmf = True
            fitter.save_hist = 'pop_histories' in self.base_kwargs
            fitter.save_src = True    # Ugh can't be pickled...send tables? yes.
        else:
            fitter.save_hmf = False
            fitter.save_hist = False
            fitter.save_src = False

        self.fitter = fitter

        # RUN
        fitter.run(prefix=prefix, burn=burn, steps=steps, save_freq=save_freq,
            clobber=clobber, restart=restart, burn_method=burn_method,
            recenter=recenter)
