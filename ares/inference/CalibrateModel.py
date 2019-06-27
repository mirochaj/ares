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

_b14 = read_lit('bouwens2014')
filt_hst = {4: _b14.filt_shallow[4], 5: _b14.filt_shallow[5],
    6: _b14.filt_deep[6], 7: _b14.filt_deep[7]}

_zcal_lf = [3.8, 4.9, 5.9, 6.9, 7.9, 10.]
_zcal_smf = [3, 4, 5, 6, 7, 8]
_zcal_beta = [4, 5, 6, 7]

acceptable_sfe_params = ['slope-low', 'slope-high', 'norm', 'peak']
acceptable_dust_params = ['norm', 'slope', 'peak', 'fcov', 'yield']

class CalibrateModel(object):
    """
    Convenience class for calibrating galaxy models to UVLFs and/or SMFs.
    """
    def __init__(self, fit_lf=[5.9], fit_smf=False, fit_gs=False, fit_beta=False,
        use_ensemble=True, add_suffix=True,
        include_sfe=True, free_params_sfe=[], zevol_sfe=[],
        include_fshock=False, include_scatter_mar=False, name=None,
        include_dust='var_beta', include_fduty=False, zevol_fduty=False,
        zevol_fshock=False, zevol_dust=False, free_params_dust=[],
        save_lf=True, save_smf=False, save_sam=False,
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
                
        self.include_sfe = include_sfe
        self.include_fshock = int(include_fshock)
        self.include_scatter_mar = int(include_scatter_mar)
        
        self.include_dust = include_dust
        self.include_fduty = include_fduty

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
        self.use_ensemble = int(use_ensemble)
        
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
            s += 'gs_' # add freq range?
    
        if self.include_sfe in [True, 1, 'dpl', 'flex']:
            enorm = 'norm' in self.zevol_sfe
            epeak = 'peak' in self.zevol_sfe
            eslop = 'slope-low' in self.zevol_sfe \
                 or 'slope-high' in self.zevol_sfe
                 
            enorm_d = 'norm' in self.zevol_dust
            epeak_d = 'norm' in self.zevol_dust
            
            rest = 'sfe-dpl_enorm-{}_epeak-{}_eshape-{}_dust-{}_enorm-{}_epeak-{}_fduty-{}_efduty-{}'.format(
                int(enorm), int(epeak), int(eslop), self.include_dust, int(enorm_d),
                int(epeak_d), int(self.include_fduty), int(self.zevol_fduty))
        #elif self.include_sfe in ['f17-p', 'f17-E']:
        #    rest = 'sfe-{}_fshock-{}_dust-{}_edust-{}_zcal-{}'.format(
        #        self.include_sfe, self.include_fshock, self.include_dust, 
        #        self.zevol_dust, zs)
        else:
            raise ValueError('Unrecognized option for `include_sfe`.')
            
        s += rest
        
        if self.name is not None:
            if self.add_suffix:
                s = self.name + '_' + s
            else:
                s = self.name    
        
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
                    guesses['pq_func_par0[1]'] = -1.4
                    is_log.extend([True])
                    jitter.extend([0.1])
                    ps.add_distribution(UniformDistribution(-7, 0.), 'pq_func_par0[1]')
                    
                    if 'norm' in self.zevol_sfe:
                        free_pars.append('pq_func_par2[1]')
                        guesses['pq_func_par2[1]'] = 0.
                        is_log.extend([False])
                        jitter.extend([0.1])
                        ps.add_distribution(UniformDistribution(-3, 3.), 'pq_func_par2[1]')
                        
                # Peak mass
                if 'peak' in self.free_params_sfe:
                    free_pars.append('pq_func_par0[2]')
                    guesses['pq_func_par0[2]'] = 11.
                    is_log.extend([True])
                    jitter.extend([0.1])
                    ps.add_distribution(UniformDistribution(9., 13.), 'pq_func_par0[2]')
                    
                    if 'peak' in self.zevol_sfe:
                        free_pars.append('pq_func_par2[2]')
                        guesses['pq_func_par2[2]'] = 0.
                        is_log.extend([False])
                        jitter.extend([0.1])
                        ps.add_distribution(UniformDistribution(-3, 3.), 'pq_func_par2[2]')
                        
                # Slope at low-mass side of peak
                if 'slope-low' in self.free_params_sfe:                    
                    free_pars.append('pq_func_par0[3]')
                    guesses['pq_func_par0[3]'] = 0.66
                    is_log.extend([False])
                    jitter.extend([0.1])
                    ps.add_distribution(UniformDistribution(0.0, 1.5), 'pq_func_par0[3]')
                    
                    # Allow to evolve with redshift?
                    if 'slope-low' in self.zevol_sfe:
                        free_pars.append('pq_func_par2[3]')
                        guesses['pq_func_par2[3]'] = 0.
                        is_log.extend([False])
                        jitter.extend([0.1])
                        ps.add_distribution(UniformDistribution(-3, 3.), 'pq_func_par2[3]')
                
                # Slope at high-mass side of peak        
                if 'slope-high' in self.free_params_sfe:
                    free_pars.append('pq_func_par0[4]')
                    guesses['pq_func_par0[4]'] = 0.
                    is_log.extend([False])
                    jitter.extend([0.1])
                    ps.add_distribution(UniformDistribution(-3., 0.1), 'pq_func_par0[4]')
                    
                    # Allow to evolve with redshift?
                    if 'slope-high' in self.zevol_sfe:
                        free_pars.append('pq_func_par2[4]')
                        guesses['pq_func_par2[4]'] = 0.
                        is_log.extend([False])
                        jitter.extend([0.1])
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
            # fduty
            ##
            if self.include_fduty:
                                
                # Normalization of SFE
                free_pars.extend(['pq_func_par0[41]', 'pq_func_par2[40]'])
                guesses['pq_func_par0[41]'] = 0.8
                guesses['pq_func_par2[40]'] = 0.2
                is_log.extend([False, False])
                jitter.extend([0.1, 0.1])
                ps.add_distribution(UniformDistribution(0., 1.), 'pq_func_par0[41]')
                ps.add_distribution(UniformDistribution(0., 2.), 'pq_func_par2[40]')
                
                if self.zevol_fduty:
                    free_pars.append('pq_func_par2[41]')
                    guesses['pq_func_par2[41]'] = 0.
                    is_log.extend([False])
                    jitter.extend([0.1])
                    ps.add_distribution(UniformDistribution(-2, 2.), 'pq_func_par2[41]')
            
            
            ##
            # DUST REDDENING
            ##
            if self.include_dust in ['screen', 'screen-dpl', 'patchy']:
                
                if 'norm' in self.free_params_dust:
                    
                    free_pars.append('pq_func_par0[23]')
                    guesses['pq_func_par0[23]'] = 1.6
                    is_log.extend([False])
                    jitter.extend([0.1])
                    ps.add_distribution(UniformDistribution(0.1, 10.), 'pq_func_par0[23]')
                                        
                    if 'norm' in self.zevol_dust:
                        free_pars.append('pq_func_par2[23]')
                        guesses['pq_func_par2[23]'] = 0.
                        is_log.extend([False])
                        jitter.extend([0.5])
                        ps.add_distribution(UniformDistribution(-2., 2.), 'pq_func_par2[23]')

                if 'slope' in self.free_params_dust:
                    free_pars.append('pq_func_par2[22]')
                    guesses['pq_func_par2[22]'] = 0.45
                    is_log.extend([False])
                    jitter.extend([0.1])
                    ps.add_distribution(UniformDistribution(-0.2, 2.), 'pq_func_par2[22]')

                    if self.include_dust in ['screen-dpl', 'patchy']:
                        free_pars.append('pq_func_par3[22]')
                        guesses['pq_func_par3[22]'] = 0.45
                        is_log.extend([False])
                        jitter.extend([0.1])
                        ps.add_distribution(UniformDistribution(-2., 2.), 'pq_func_par3[22]')
                
                if 'peak' in self.free_params_dust:
                    assert self.include_dust in ['screen-dpl', 'patchy']
                    
                    free_pars.append('pq_func_par0[24]')
                    guesses['pq_func_par0[24]'] = 11.5
                    is_log.extend([True])
                    jitter.extend([0.5])
                    ps.add_distribution(UniformDistribution(9., 13.), 'pq_func_par0[24]')                    

                    if 'peak' in self.zevol_dust:
                        free_pars.append('pq_func_par2[24]')
                        guesses['pq_func_par2[24]'] = 0.0
                        is_log.extend([False])
                        jitter.extend([0.5])
                        ps.add_distribution(UniformDistribution(-2., 2.), 'pq_func_par2[24]')
                    
                if 'fcov' in self.free_params_dust:    

                    # fcov parameters (no zevol)
                    free_pars.extend(['pq_func_par0[25]', #'pq_func_par1[25]',
                        'pq_func_par3[25]', 'pq_func_par0[26]'])

                    # Tanh describing covering fraction
                    guesses['pq_func_par0[25]'] = 0.25
                    #guesses['pq_func_par1[21]'] = 0.98
                    guesses['pq_func_par3[25]'] = 0.2
                    guesses['pq_func_par0[26]'] = 10.8
                
                    is_log.extend([False, False, False])
                    jitter.extend([0.05, 0.1, 0.3])
                    
                    ps.add_distribution(UniformDistribution(0., 1.), 'pq_func_par0[25]')
                    ps.add_distribution(UniformDistribution(0., 5.), 'pq_func_par3[25]')
                    ps.add_distribution(UniformDistribution(8., 14.), 'pq_func_par0[26]')
                                    
                    # Just let transition mass evolve
                    if 'fcov' in self.zevol_dust:
                        free_pars.extend(['pq_func_par2[26]'])
                        guesses['pq_func_par2[26]'] = 0.
                        is_log.extend([False])
                        jitter.extend([0.03])
                        ps.add_distribution(UniformDistribution(-1, 1.), 'pq_func_par2[26]') 
                      
                if 'yield' in self.free_params_dust:
                    
                    free_pars.extend(['pq_func_par0[28]', 'pq_func_par2[27]'])
                    guesses['pq_func_par0[28]'] = 0.3
                    guesses['pq_func_par2[27]'] = 0.
                    is_log.extend([False, False])
                    jitter.extend([0.1, 0.5])
                    ps.add_distribution(UniformDistribution(0., 1.0), 'pq_func_par0[28]')
                    ps.add_distribution(UniformDistribution(-1.5, 1.5), 'pq_func_par2[27]')

                    if 'yield' in self.zevol_dust:
                        free_pars.append('pq_func_par2[28]')
                        guesses['pq_func_par2[28]'] = 0.0
                        is_log.extend([False])
                        jitter.extend([0.5])
                        ps.add_distribution(UniformDistribution(-2., 2.), 'pq_func_par2[28]')
                      
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
        redshifts = np.array([4, 6, 8, 10]) # generic

        if self.fit_lf:
            red_lf = np.array(self.fit_lf)
        else:
            red_lf = redshifts
        
        if self.fit_smf:
            red_smf = np.array(self.fit_smf)
        else:
            red_smf = redshifts    
            
        if self.fit_beta:
            red_beta = np.array(self.fit_beta)
        else:
            red_beta = redshifts    
                    
        MUV = np.arange(-30, 5., 0.5)
        
        Mh = np.logspace(7, 13, 61)
        Ms = np.arange(7, 13.1, 0.1)

        # Always save the UVLF
        blob_n = ['galaxy_lf']
        blob_i = [('z', red_lf), ('x', MUV)]
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

        if self.include_fduty:
            blob_n = ['fduty']
            blob_i = [('z', redshifts), ('Mh', Mh)]

            if self.use_ensemble:
                blob_f = ['guide.fduty']
            else:
                blob_f = ['fduty']

            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(blob_i)
            blob_pars['blob_funcs'].append(blob_f)
            blob_pars['blob_kwargs'].append(None)

        # Binary obscuration
        #if self.include_obsc:
        #    blob_n2.append('fobsc')
        #    
        #    if self.use_ensemble:
        #        blob_f2.append('guide.fobsc')
        #    else:
        #        blob_f2.append('fobsc')
        #        
        #    raise NotImplemented('must add to pars')
        
        # SAM stuff
        if self.save_sam:
            blob_n = ['SFR', 'SMHM']
            blob_i = [('z', redshifts), ('Mh', Mh)]
            
            if self.use_ensemble:
                blob_f = ['guide.SFR', 'SMHM']
            else:
                blob_f = ['SFR', 'SMHM']
                
            blob_k = [{}, {'return_mean_only': True}]    
            
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
            blob_n = ['dust_fcov', 'dust_scale']
            blob_i = [('z', redshifts), ('Mh', Mh)]
            blob_f = ['guide.dust_fcov', 'guide.dust_scale']
            
            if type(self.base_kwargs['pop_dust_yield']) == str:
                blob_n.append('dust_yield')
                blob_f.append('guide.dust_yield')
            
            blob_pars['blob_names'].append(blob_n)
            blob_pars['blob_ivars'].append(blob_i)
            blob_pars['blob_funcs'].append(blob_f)
            blob_pars['blob_kwargs'].append(None)            
        
        # MUV-Beta
        if self.save_beta != False:
            
            Mbins = np.arange(-30, -10, 0.1)
            
            blob_n = ['AUV']
            blob_i = [('z', red_beta), ('MUV', MUV)]
            blob_f = ['AUV']
            # By default, MUV refers to 1600 magnitude
            blob_k = [{'return_binned': True, 'cam': ('wfc', 'wfc3'), 
                'filters': filt_hst, 'dlam': 20.,
                'Mwave': 1600., 'Mbins': Mbins}]
                            
            kw_hst = {'cam': ('wfc', 'wfc3'), 'filters': filt_hst,
                'dlam':20., 'rest_wave': None, 'return_binned': True,
                'Mwave': 1600., 'Mbins': Mbins}

            kw_spec = {'dlam':700., 'rest_wave': (1600., 2300.),
                'return_binned': True, 'Mwave': 1600.}
            
            blob_f.extend(['Beta'] * 2)
            blob_n.extend(['beta_hst', 'beta_spec'])
            blob_k.extend([kw_hst, kw_spec])
            
            # Save also the geometric mean of photometry as a function
            # of a magnitude at fixed rest wavelength.
            kw_mag = {'cam': ('wfc', 'wfc3'), 'filters': filt_hst,
                'dlam': 20.}

            # Save geometric mean magnitudes also
            blob_n.append('MUV_gm')
            blob_f.append('Magnitude')
            blob_k.append(kw_mag)

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
        
            if self.include_fduty:
                self._base_kwargs.update(PB('in_prep:fduty').pars_by_pop(0, 1))
        
        # Initialize with best guesses mostly for debugging purposes
        for i, par in enumerate(self.parameters):
            if self.is_log[i]:
                self._base_kwargs[par] = 10**self.guesses[par]
            else:
                self._base_kwargs[par] = self.guesses[par]
        
        return self._base_kwargs
        
    def update_kwargs(self, **kwargs):
        bkw = self.base_kwargs
        self._base_kwargs.update(kwargs)
        
    def run(self, steps, burn=0, nwalkers=None, save_freq=10, prefix=None, 
        debug=True, restart=False, clobber=False, verbose=True,
        cache_tricks=False):
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
        if self.fit_beta:
            include.append('beta')
            data.extend(['bouwens2014'])
        if self.fit_gs:
            raise NotImplemented('sorry folks')

        # Must be before data is set
        fitter_lf.redshifts = {'lf': self.fit_lf, 'smf': self.fit_smf,
            'beta': self.fit_beta}
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
            clobber=clobber, restart=restart)
        
        
        
        