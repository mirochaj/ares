"""

CalibrateModel.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Wed 13 Feb 2019 17:11:07 EST

Description: 

"""

import os
import numpy as np
from .ModelFit import ModelFit
from ..util import ParameterBundle as PB
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

_zcal = [3.8, 4.9, 5.9, 6.9, 7.9, 10.]

acceptable_sfe_params = ['slope-low', 'slope-high', 'norm', 'peak']
acceptable_dust_params = ['low', 'high', 'mid', 'width', 'norm', 'mdep', 'zdep']

class CalibrateModel(object):
    """
    Convenience class for calibrating galaxy models to UVLFs and/or SMFs.
    """
    def __init__(self, fit_lf=True, fit_smf=False, fit_gs=False, use_ensemble=True, 
        zcal=[4.], include_sfe=True, free_params_sfe=True, zevol_sfe=[],
        include_fshock=False, include_scatter_mar=False,
        include_dust='var_beta', include_obsc=False, zevol_obsc=False,
        zevol_fshock=False, zevol_dust=False, free_params_dust=[],
        save_lf=True, save_smf=False, 
        save_sfrd=False, save_beta=False, save_dust=False):
        """
        Calibrate a galaxy model to available data.
        
        .. note :: All the `include_*` parameters control what goes into our
            base_kwargs, while the `free_params_*` parameters control what
            we allow to vary in the fit.
        
        Parameters
        ----------
        fit_lf : bool
            Use available luminosity function measurements?
        fit_smf : bool
            Use available stellar mass function measurements?    
        fit_gs : bool
            Use constraints on global 21-cm signal?
        zcal : list
            Calibrate to data at these redshifts.
        
        zevol_sfe_norm : bool
            Allow redshift evolution in the normalization of the SFE?
        zevol_sfe_peak : bool
            Allow redshift evolution in the where the SFE peaks (in mass)?
        zevol_sfe_shape: bool
            Allow redshift evolution in the power-slopes of SFE?
        include_obsc : bool 
            Allow binary obscuration?
        zevol_obsc : bool
            Allow redshift evolution in parameters describing obscuration?
        clobber : bool
            Overwrite existing data outputs?
        """
        
        self.fit_lf = int(fit_lf)
        self.fit_smf = int(fit_smf)
        self.fit_gs = int(fit_gs)
        self.zcal = zcal
        self.include_sfe = include_sfe
        self.include_obsc = int(include_obsc)
        self.include_fshock = int(include_fshock)
        self.include_scatter_mar = int(include_scatter_mar)

        self.include_dust = include_dust

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

        self.zevol_dust = int(zevol_dust)
        self.zevol_obsc = int(zevol_obsc)
        
        self.save_lf = int(save_lf)
        self.save_smf = int(save_smf)
        self.save_sfrd = int(save_sfrd)
        self.save_beta = int(save_beta)
        self.save_dust = int(save_dust)
        self.use_ensemble = int(use_ensemble)
        
        
    @property
    def prefix(self):
        """
        Generate output filename.
        """
        zcal = []
        for z in _zcal:
            if z not in self.zcal:
                continue

            zcal.append(z)
                
        zs = ''
        for z in zcal:
            zs += '%i_' % round(z)
        zs = zs.rstrip('_')
    
        s = ''
        if self.fit_lf:
            s += 'lf_'
        if self.fit_smf:
            s += 'smf_'
        if self.fit_gs:
            s += 'gs_'
    
        if self.include_sfe in [True, 1, 'dpl', 'flex']:
            enorm = 'norm' in self.zevol_sfe
            epeak = 'peak' in self.zevol_sfe
            eslop = 'slope-low' in self.zevol_sfe \
                 or 'slope-high' in self.zevol_sfe
            
            rest = 'sfe-dpl_enorm-{}_epeak-{}_eshape-{}_dust-{}_zcal-{}'.format(
                int(enorm), int(epeak), int(eslop), self.include_dust, zs)
        elif self.include_sfe in ['f17-p', 'f17-E']:
            rest = 'sfe-{}_fshock-{}_dust-{}_zcal-{}'.format(
                self.include_sfe, self.include_fshock, self.include_dust, zs)
        else:
            raise ValueError('Unrecognized option for `include_sfe`.')
            
        s += rest
        
        if rank == 0:
            print("Will save to files with prefix {}.".format(s))

        return s
        
    @property
    def parameters(self):
        if not hasattr(self, '_parameters'):
            
            free_pars = []
            guesses = {}
            is_log = []
            jitter = []
            ps = DistributionSet()
            
            ##
            # MAR scatter
            ##
            if self.include_scatter_mar:
                free_pars.append('pop_scatter_mar')
                guesses['pop_scatter_mar'] = 0.3
                is_log.append(False)
                jitter.append(0.1)                
                ps.add_distribution(UniformDistribution(0, 1.), 'pop_scatter_mar')

            ##
            # Allow redshift evolution in normalization?
            ##
            if self.include_sfe in [True, 1, 'dpl', 'flex']:
                
                # Normalization of SFE
                if 'norm' in self.free_params_sfe:
                    free_pars.append('pq_func_par0[1]')
                    guesses['pq_func_par0[1]'] = -4.
                    is_log.extend([True])
                    jitter.extend([0.5])
                    ps.add_distribution(UniformDistribution(-7, 0.), 'pq_func_par0[1]')
                    
                    if 'norm' in self.zevol_sfe:
                        free_pars.append('pq_func_par2[1]')
                        guesses['pq_func_par2[1]'] = 0.
                        is_log.extend([False])
                        jitter.extend([0.5])
                        ps.add_distribution(UniformDistribution(-3, 3.), 'pq_func_par2[1]')
                        
                # Peak mass
                if 'peak' in self.free_params_sfe:
                    free_pars.append('pq_func_par0[2]')
                    guesses['pq_func_par0[2]'] = 11.5
                    is_log.extend([True])
                    jitter.extend([0.5])
                    ps.add_distribution(UniformDistribution(9., 13.), 'pq_func_par0[2]')
                    
                    if 'peak' in self.zevol_sfe:
                        free_pars.append('pq_func_par2[2]')
                        guesses['pq_func_par2[2]'] = 0.
                        is_log.extend([False])
                        jitter.extend([0.5])
                        ps.add_distribution(UniformDistribution(-3, 3.), 'pq_func_par2[2]')
                        
                # Slope at low-mass side of peak
                if 'slope-low' in self.free_params_sfe:                    
                    free_pars.append('pq_func_par0[3]')
                    guesses['pq_func_par0[3]'] = 0.666
                    is_log.extend([False])
                    jitter.extend([0.333])
                    ps.add_distribution(UniformDistribution(0.0, 1.5), 'pq_func_par0[3]')
                    
                    # Allow to evolve with redshift?
                    if 'slope-low' in self.zevol_sfe:
                        free_pars.append('pq_func_par2[3]')
                        guesses['pq_func_par2[3]'] = 0.
                        is_log.extend([False])
                        jitter.extend([0.5])
                        ps.add_distribution(UniformDistribution(-3, 3.), 'pq_func_par2[3]')
                
                # Slope at high-mass side of peak        
                if 'slope-high' in self.free_params_sfe:
                    free_pars.append('pq_func_par0[4]')
                    guesses['pq_func_par0[4]'] = 0.
                    is_log.extend([False])
                    jitter.extend([0.1])
                    ps.add_distribution(UniformDistribution(-3., 3.), 'pq_func_par0[4]')
                    
                    # Allow to evolve with redshift?
                    if 'slope-high' in self.zevol_sfe:
                        free_pars.append('pq_func_par2[4]')
                        guesses['pq_func_par2[4]'] = 0.
                        is_log.extend([False])
                        jitter.extend([0.5])
                        ps.add_distribution(UniformDistribution(-3, 3.), 'pq_func_par2[4]')
                    
            ##
            # Steve's models
            ##
            elif self.include_sfe in ['f17-p', 'f17-E']:
                # 10 * epsilon_K * omega_49
                free_pars.append('pq_func_par0[1]') 
                guesses['pq_func_par0[1]'] = 0.
                is_log.extend([True])
                jitter.extend([0.5])
                ps.add_distribution(UniformDistribution(-2, 2), 'pq_func_par0[1]')
            
            ##
            # OBSCURATION
            ##
            if self.include_obsc:
                raise NotImplemented('help')
                free_pars.extend(['pq_func_par0[11]', 'pq_func_par0[12]', 'pq_func_par0[13]'])
                guesses['pq_func_par0[11]'] = 0.5
                guesses['pq_func_par0[12]'] = 11.5
                guesses['pq_func_par0[13]'] = 1.0
                is_log.extend([False, False, False])
                jitter.extend([0.2, 0.5, 0.5])
    
                ps.add_distribution(UniformDistribution(0., 1.), 'pq_func_par0[11]')
                ps.add_distribution(UniformDistribution(8, 14), 'pq_func_par0[12]')
                ps.add_distribution(UniformDistribution(0., 8.), 'pq_func_par0[13]')
    
            ##
            # DUST REDDENING
            ##
            if self.include_dust.startswith('phys'):
                
                #if self.zevol_dust:
                #    free_pars.extend(['pq_func_par0[24]', 'pq_func_par2[24]'])
                #    guesses['pq_func_par0[24]'] = 10.5
                #    guesses['pq_func_par2[24]'] = 0.
                #    is_log.extend([False, False])
                #    jitter.extend([0.5, 1.0])
                #    ps.add_distribution(UniformDistribution(8., 13.), 'pq_func_par0[24]')
                #    ps.add_distribution(UniformDistribution(-2, 2.), 'pq_func_par2[24]')
                #else:
                #    free_pars.append('pq_func_par2[21]')    
                #    guesses['pq_func_par2[21]'] = 10.5
                #    is_log.append(False)
                #    jitter.append(0.5)
                #    ps.add_distribution(UniformDistribution(8., 13.), 'pq_func_par2[21]')

                free_pars.extend(['pq_func_par0[21]', 'pq_func_par1[21]',
                    'pq_func_par2[21]', 'pq_func_par3[21]', 'pq_func_par2[22]', 
                    'pq_func_par0[23]', 'pq_func_par2[23]'])

                # Tanh describing covering fraction
                guesses['pq_func_par0[21]'] = 0.05
                guesses['pq_func_par1[21]'] = 0.95
                guesses['pq_func_par2[21]'] = 10.5
                guesses['pq_func_par3[21]'] = 2.
                
                # Mass-dependence of R_d
                guesses['pq_func_par2[22]'] = 0.4
                
                # Normalization and redshift-dependence of R_d
                guesses['pq_func_par0[23]'] = 0.
                guesses['pq_func_par2[23]'] = -1.
                
                is_log.extend([False, False, False, False, False, True, False])
                jitter.extend([0.05, 0.05, 0.5, 0.5, 0.1, 0.5, 0.2])
                
                ps.add_distribution(UniformDistribution(0., 1.), 'pq_func_par0[21]')
                ps.add_distribution(UniformDistribution(0., 1.), 'pq_func_par1[21]')
                ps.add_distribution(UniformDistribution(8, 14), 'pq_func_par2[21]')
                ps.add_distribution(UniformDistribution(0., 5.), 'pq_func_par3[21]')
                ps.add_distribution(UniformDistribution(-1., 2.), 'pq_func_par2[22]')
                ps.add_distribution(UniformDistribution(-3., 2.), 'pq_func_par0[23]')
                ps.add_distribution(UniformDistribution(-2., 1.), 'pq_func_par2[23]')
                                    
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
        self._guesses = value
        
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
        redshifts = np.array([3.8, 4, 4.9, 5, 5.9, 6, 6.9, 7, 7.9, 8, 9, 
            10, 11, 12, 13, 14, 15])
        MUV = np.arange(-25, 5., 0.5)
        MUV_2 = np.arange(-25, -5., 0.5)
        Mh = np.logspace(7, 13, 61)
        Ms = np.arange(7, 13.1, 0.1)

        # Always save the UVLF
        blob_n = ['galaxy_lf']
        blob_i = [('z', redshifts), ('x', MUV)]
        blob_f = ['LuminosityFunction']
        
        blob_pars = \
        {
         'blob_names': [blob_n],
         'blob_ivars': [blob_i],
         'blob_funcs': [blob_f],
         'blob_kwargs': [None],
        }

        # Save the SFE if we're varying its parameters.
        if self.include_sfe in [True, 1, 'dpl', 'flex']:
            blob_n = ['fstar']
            blob_i = [('z', redshifts), ('Mh', Mh)]
            
            if self.use_ensemble:
                blob_f = ['guide.fstar']
            else:
                blob_f = ['fstar']
                
            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(blob_i)
            blob_pars['blob_funcs'].append(blob_f)
            blob_pars['blob_kwargs'].append(None)

        # Binary obscuration
        if self.include_obsc:
            blob_n2.append('fobsc')
            
            if self.use_ensemble:
                blob_f2.append('guide.fobsc')
            else:
                blob_f2.append('fobsc')
                
            raise NotImplemented('must add to pars')
        
        # SMF
        if self.save_smf:    
            blob_n = ['galaxy_smf']
            blob_i = [('z', redshifts), ('bins', Ms)]
            blob_f = ['StellarMassFunction']
            
            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(blob_i)
            blob_pars['blob_funcs'].append(blob_f)
            blob_pars['blob_kwargs'].append(None)
        
        # Covering factor and scale length    
        if self.save_dust:
            blob_n = ['dust_fcov', 'dust_scale']
            blob_i = [('z', redshifts), ('Mh', Mh)]
            blob_f = ['guide.dust_fcov', 'guide.dust_scale']
            
            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(blob_i)
            blob_pars['blob_funcs'].append(blob_f)
            blob_pars['blob_kwargs'].append(None)
        
        # MUV-Beta
        if self.save_beta:
            blob_n = ['beta']
            blob_i = [('z', np.array([4, 6, 8, 10])), ('MUV', MUV_2)]
            blob_f = ['Beta']
            
            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(blob_i)
            blob_pars['blob_funcs'].append(blob_f)
            blob_pars['blob_kwargs'].append([{'wave': 1600.}])
            
        # Cosmic SFRD
        if self.save_sfrd:
            blob_n = ['sfrd']
            blob_i = [('z', np.arange(3.5, 30.1, 0.1))]
            blob_f = ['SFRD']
            
            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(blob_i)
            blob_pars['blob_funcs'].append(blob_f)
            blob_pars['blob_kwargs'].append(None)
        
        return blob_pars
        
    @property
    def base_kwargs(self):
        if not hasattr(self, '_base_kwargs'):
            
            if self.include_sfe in [1, True, 'dpl', 'flex']:
                self._base_kwargs = \
                    PB('mirocha2017:base').pars_by_pop(0, 1) \
                  + PB('mirocha2017:dflex').pars_by_pop(0, 1) \
                  + PB('dust:{}'.format(self.include_dust))
            elif self.include_sfe in ['f17-p', 'f17-E']:
                s = 'energy' if self.include_sfe.split('-')[1] == 'E' \
                    else 'momentum'
                
                self._base_kwargs = \
                    PB('furlanetto2017:{}'.format(s)) \
                  + PB('dust:{}'.format(self.include_dust))
                  
                if self.include_fshock:
                    self._base_kwargs = self._base_kwargs \
                        + PB('furlanetto2017:fshock')
                  
                # Make sure 'pop_L1600_per_sfr' is None?  
                  
            else:
                raise ValueError('Unrecognized option for `include_sfe`.')
        
        return self._base_kwargs
        
    def update_kwargs(self, **kwargs):
        bkw = self.base_kwargs
        self._base_kwargs.update(kwargs)
        
    def run(self, steps, burn=0, nwalkers=None, save_freq=10, prefix=None, 
        debug=True, restart=False, clobber=False, verbose=True):
        """
        Create a fitter class and run the fit!
        """
        
        if prefix is None:
            prefix = self.prefix

        fitter_lf = FitGalaxyPopulation()
        
        data = []
        include = []
        if self.fit_lf:
            include.append('lf')
            data.extend(['bouwens2015', 'oesch2018'])
        if self.fit_smf:
            include.append('smf')
            data.append('song2016')
        if self.fit_gs:    
            raise NotImplemented('sorry folks')

        # Must be before data is set
        fitter_lf.redshifts = {'lf': self.zcal, 'smf': self.zcal}
        fitter_lf.include = include

        fitter_lf.data = data
        

        ##
        # Stitch together parameters
        ##
        pars = self.base_kwargs
        pars.update(self.blobs)
        

        # Master fitter
        fitter = ModelFit(**pars)
        fitter.add_fitter(fitter_lf)
        
        if self.use_ensemble:
            fitter.simulator = GalaxyEnsemble
        else:
            fitter.simulator = GalaxyCohort

        fitter.parameters = self.parameters
        fitter.is_log = self.is_log
        fitter.debug = debug
        fitter.verbose = verbose
        
        fitter.prior_set = self.priors
        
        if nwalkers is None:
            nw = 2 * len(self.parameters)
            if rank == 0:
                print("Running with {} walkers.".format(nw))
        else:
            nw = nwalkers
        
        fitter.nwalkers = nw
        fitter.jitter = self.jitter
        fitter.guesses = self.guesses

        fitter.save_hmf = True
        fitter.save_hist = 'pop_histories' in self.base_kwargs
        fitter.save_src = True    # Ugh can't be pickled...send tables? yes.

        self.fitter = fitter

        # RUN
        fitter.run(prefix=prefix, burn=burn, steps=steps, save_freq=save_freq, 
            clobber=clobber, restart=restart)
        
        
        
        