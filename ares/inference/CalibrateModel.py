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
from .FitGalaxyPopulation import FitGalaxyPopulation
from ..populations.GalaxyCohort import GalaxyCohort
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

class CalibrateModel(object):
    def __init__(self, include_lf=True, include_smf=False, include_gs=False,
        zcal=[4.], zevol_sfe_norm=False, zevol_sfe_peak=False, 
        zevol_sfe_shape=False, include_obsc=False, zevol_obsc=False,
        include_dust='var_beta', save_lf=True, save_smf=False, save_sfrd=False):
        """
        Calibrate a galaxy model to available data.
        
        Parameters
        ----------
        include_lf : bool
            Use available luminosity function measurements?
        include_smf : bool
            Use available stellar mass function measurements?    
        include_gs : bool
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
        
        self.include_lf = int(include_lf)
        self.include_smf = int(include_smf)
        self.include_gs = int(include_gs)
        self.zcal = zcal
        self.zevol_sfe_norm = int(zevol_sfe_norm)
        self.zevol_sfe_peak = int(zevol_sfe_peak)
        self.zevol_sfe_shape = int(zevol_sfe_shape)
        self.include_obsc = int(include_obsc)
        self.zevol_obsc = int(zevol_obsc)
        self.include_dust = int(include_dust)
        self.save_lf = int(save_lf)
        self.save_smf = int(save_smf)
        self.save_sfrd = int(save_sfrd)
        
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
        if self.include_lf:
            s += 'lf_'
        if self.include_smf:
            s += 'smf_'
        if self.include_gs:
            s += 'gs_'
    
        rest = 'enorm-{}_epeak-{}_eshape-{}_obsc-{}_eobsc-{}_dust-{}_zcal-{}'.format(
            self.zevol_sfe_norm, self.zevol_sfe_peak, self.zevol_sfe_shape,
            self.include_obsc, self.zevol_obsc, self.include_dust, zs)
    
        s += rest

        return s
        
    @property
    def parameters(self):
        if not hasattr(self, '_parameters'):
            free_pars = ['pop_scatter_mar']
            guesses = {'pop_scatter_mar': 0.3}
            is_log = [False]
            jitter = [0.1]

            ps = DistributionSet()
            ps.add_distribution(UniformDistribution(0, 1.), 'pop_scatter_mar')

            ##
            # Allow redshift evolution in normalization?
            ##
            if self.zevol_sfe_norm:
                free_pars.extend(['pq_func_par0[1]', 'pq_func_par2[1]'])
                guesses['pq_func_par0[1]'] = -4.
                guesses['pq_func_par2[1]'] = 0.0
                is_log.extend([True, False])
                jitter.extend([0.2, 0.2])
                ps.add_distribution(UniformDistribution(-6, 0.0), 'pq_func_par0[1]')
                ps.add_distribution(UniformDistribution(-3., 3.), 'pq_func_par2[1]')
            else:
                free_pars.append('pq_func_par0[1]')
                guesses['pq_func_par0[1]'] = -4
                is_log.append(True)
                jitter.append(0.2)
                ps.add_distribution(UniformDistribution(-6, 0.), 'pq_func_par0[1]')
    
     
            ##
            # Allow redshift evolution in peak mass?
            ##    
            if self.zevol_sfe_peak:
                free_pars.extend(['pq_func_par0[2]', 'pq_func_par2[2]'])
                guesses['pq_func_par0[2]'] = 11.5
                guesses['pq_func_par2[2]'] = 0.0
                is_log.extend([True, False])
                jitter.extend([0.2, 0.2])
                ps.add_distribution(UniformDistribution(10, 13.), 'pq_func_par0[2]')
                ps.add_distribution(UniformDistribution(-3., 3.), 'pq_func_par2[2]')
            else:
                free_pars.append('pq_func_par0[2]')
                guesses['pq_func_par0[2]'] = 11.5
                is_log.append(True)
                jitter.append(0.2)
                ps.add_distribution(UniformDistribution(10, 13), 'pq_func_par0[2]')
    
            ##    
            # Allow slope in SFE to evolve?
            ##
            if self.zevol_sfe_shape:
                free_pars.extend(['pq_func_par0[3]', 'pq_func_par2[3]'])
                free_pars.extend(['pq_func_par0[4]', 'pq_func_par2[4]'])
                guesses['pq_func_par0[3]'] = 1.0
                guesses['pq_func_par2[3]'] = 0.0
                guesses['pq_func_par0[4]'] = -0.25
                guesses['pq_func_par2[4]'] = 0.0
                is_log.extend([False]*4)
                jitter.extend([0.2, 0.2, 0.2, 0.2])
                ps.add_distribution(UniformDistribution(0., 2.), 'pq_func_par0[3]')
                ps.add_distribution(UniformDistribution(-3., 3.), 'pq_func_par2[3]')
                ps.add_distribution(UniformDistribution(-2, 1.), 'pq_func_par0[4]')
                ps.add_distribution(UniformDistribution(-3., 3.), 'pq_func_par2[4]')
            else:
                free_pars.extend(['pq_func_par0[3]', 'pq_func_par0[4]'])
                guesses['pq_func_par0[3]'] = 1.0
                guesses['pq_func_par0[4]'] = -0.25
                is_log.extend([False]*2)
                jitter.extend([0.2, 0.2])
                ps.add_distribution(UniformDistribution(0., 2.), 'pq_func_par0[3]')
                ps.add_distribution(UniformDistribution(-2, 1.), 'pq_func_par0[4]')

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
        redshifts = np.array([1.75, 2.25, 2.75, 3, 3.8, 4, 4.9, 5, 5.9, 6, 
            6.9, 7, 7.9, 8, 9, 10, 11, 12, 13, 14, 15])
        MUV = np.arange(-28, -5.8, 0.2)
        Mh = np.logspace(7, 13, 61)
        Ms = np.arange(7, 13.1, 0.1)

        # blob 1: the LF. Give it a name, and the function needed to calculate it.
        blob_n1 = ['galaxy_lf']
        blob_i1 = [('z', redshifts), ('x', MUV)]
        blob_f1 = ['LuminosityFunction']

        # blob 2: the SFE. Same deal.
        blob_n2 = ['fstar']
        blob_i2 = [('z', redshifts), ('Mh', Mh)]
        blob_f2 = ['guide.fstar']

        if self.include_obsc:
            blob_n2.append('fobsc')
            blob_f2.append('guide.fobsc')
            raise NotImplemented('must add to pars')
        
        blob_pars = \
        {
         'blob_names': [blob_n1, blob_n2],
         'blob_ivars': [blob_i1, blob_i2],
         'blob_funcs': [blob_f1, blob_f2],
         'blob_kwargs': [None] * 2,
        }
        
        if self.save_smf:    
            blob_n3 = ['galaxy_smf']
            blob_i3 = [('z', redshifts), ('bins', Ms)]
            blob_f3 = ['StellarMassFunction']
            
            blob_pars['blob_names'].append(blob_n3)
            blob_pars['blob_ivars'].append(blob_i3)
            blob_pars['blob_funcs'].append(blob_f3)
            blob_pars['blob_kwargs'].append(None)
            
        if self.save_sfrd:
            raise NotImplemented('help')         
        
        return blob_pars
        
    def augment(self):
        pass
        
    @property
    def base_kwargs(self):
        if not hasattr(self, '_base_kwargs'):
            self._base_kwargs = \
                PB('mirocha2017:base').pars_by_pop(0, 1) \
              + PB('mirocha2017:dflex').pars_by_pop(0, 1) \
              + PB('dust:{}'.format(self.include_dust))
        
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
        if self.include_lf:
            include.append('lf')
            data.extend(['bouwens2015', 'oesch2018'])
        if self.include_smf:
            include.append('smf')
            data.append('song2016')
        if self.include_gs:    
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
        fitter.simulator = GalaxyEnsemble

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
        fitter.save_hist = True
        fitter.save_src = True    # Ugh can't be pickled...send tables? yes.

        self.fitter = fitter

        # RUN
        fitter.run(prefix=prefix, burn=burn, steps=steps, save_freq=save_freq, 
            clobber=clobber, restart=restart)
        
        
        
        