"""

FitGLF.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Oct 23 14:34:01 PDT 2015

Description: 

"""

import time
import pickle
import numpy as np
from ..util import read_lit
from ..util.PrintInfo import print_fit
import gc, os, sys, copy, types, time, re
from ..util.ParameterFile import par_info
from ..populations import GalaxyPopulation
from ..util.Stats import symmetrize_errors
from .ModelFit import LogLikelihood, ModelFit

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
    def const_term(self):
        if not hasattr(self, '_const_term'):
            self._const_term = -np.log(np.sqrt(twopi)) \
                             -  np.sum(np.log(self.error))
        return self._const_term
        
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

    def __call__(self, pars, blobs=None):
        """
        Compute log-likelihood for model generated via input parameters.
    
        Returns
        -------
        Tuple: (log likelihood, blobs)
    
        """
                
        kwargs = {}
        for i, par in enumerate(self.parameters):
        
            if self.is_log[i]:
                kwargs[par] = 10**pars[i]
            else:
                kwargs[par] = pars[i]

        # Apply prior on model parameters first (dont need to generate model)
        point = {}
        for i in range(len(self.parameters)):
            point[self.parameters[i]] = pars[i]

        lp = self.priors_P.log_prior(point)
        if not np.isfinite(lp):
            return -np.inf, self.blank_blob

        # Update kwargs
        kw = self.base_kwargs.copy()
        kw.update(kwargs)

        # Don't save base_kwargs for each proc! Needlessly expensive I/O-wise.
        self.checkpoint(**kwargs)

        pop = self.pop = GalaxyPopulation(**kw)

        #if self.priors_B.params != []:
        #    lp += self._compute_blob_prior(sim)

        # emcee will crash if this returns NaN. OK if it's inf though.
        if np.isnan(lp):
            return -np.inf, self.blank_blob
        
        t1 = time.time()
                        
        # Loop over all data points individually.
        #try:
        phi = np.zeros_like(self.ydata)
        for i, quantity in enumerate(self.metadata):
            
            if self.mask[i]:
                continue

            xdat = self.xdata[i]
            z = self.redshifts[i]

            # Generate model LF
            if quantity == 'lf':
                # Dust correction for observed galaxies
                AUV = pop.dust.AUV(z, xdat)
        
                # Compare data to model at dust-corrected magnitudes
                M = xdat - AUV
                
                # Compute LF
                p = pop.LuminosityFunction(z=z, x=M, mags=True)
            elif quantity == 'smf':
                M = xdat
                p = pop.StellarMassFunction(z, M)                
            else:
                raise ValueError('Unrecognized quantity: %s' % quantity)

            phi[i] = p
        #except:
        #    return -np.inf, self.blank_blob

        #phi = np.ma.array(_phi, mask=self.mask)

        lnL = 0.5 * np.ma.sum((phi - self.ydata)**2 / self.error**2)
            
        # Final posterior calculation
        PofD = lp + self.const_term - lnL

        if np.isnan(PofD) or (type(phi) == np.ma.core.MaskedConstant):
            return -np.inf, self.blank_blob
            
        try:
            blobs = pop.blobs
        except:
            blobs = self.blank_blob
            
        del pop, kw
        gc.collect()
                
        return PofD, blobs
    
class FitGalaxyPopulation(ModelFit):
    @property 
    def save_hmf(self):
        if not hasattr(self, '_save_hmf'):
            self._save_hmf = True
        return self._save_hmf
    
    @save_hmf.setter
    def save_hmf(self, value):
        self._save_hmf = value
    
    @property 
    def save_psm(self):
        if not hasattr(self, '_save_psm'):
            self._save_psm = True
        return self._save_psm
    
    @save_psm.setter
    def save_psm(self, value):
        self._save_psm = value    
    
    @property
    def loglikelihood(self):
        if not hasattr(self, '_loglikelihood'):

            if (self.save_hmf or self.save_psm):        
                pop = GalaxyPopulation(**self.base_kwargs)

                if self.save_hmf:
                    hmf = pop.halos
                    assert 'hmf_instance' not in self.base_kwargs
                    self.base_kwargs['hmf_instance'] = hmf    
                if self.save_psm:
                    #raise NotImplemented('help')
                    psm = pop.src
                    assert 'pop_psm_instance' not in self.base_kwargs
                    self.base_kwargs['pop_psm_instance'] = psm

            self._loglikelihood = loglikelihood(self.xdata_flat, 
                self.ydata_flat, self.error_flat, 
                self.parameters, self.is_log, self.base_kwargs, 
                self.prior_set_P, self.prior_set_B, 
                self.prefix, self.blob_info, self.checkpoint_by_proc)
            
            self._loglikelihood.redshifts = self.redshifts_flat
            self._loglikelihood.metadata = self.metadata_flat

            self.info

        return self._loglikelihood

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

    #@redshifts.setter
    #def redshifts(self, value):
    #    # This can be used to override the redshifts in the dataset and only
    #    # use some subset of them
    #
    #    # Need to be ready for 'lf' or 'smf' designation.
    #    if len(self.include) > 1:
    #        assert type(value) is dict
    #
    #    if type(value) in [int, float]:
    #        value = [value]
    #        
    #    self._redshifts = value

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
        
        if type(value) == str:
            value = [value]

        if type(value) in [list, tuple]:
            self._data = {quantity:[] for quantity in self.include}
            self._units = {quantity:[] for quantity in self.include}

            z_by_range = hasattr(self, '_redshift_bounds')
            z_by_hand = hasattr(self, '_redshifts')

            self._redshifts = {quantity:[] for quantity in self.include}

            for src in value:
                litdata = read_lit(src)

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
                            if z not in self._redshifts:
                                continue
                                
                            srczarr.append(z)
                            srcdata[z] = data[z]   
                                                    
                    self._data[quantity].append(srcdata)
                    self._redshifts[quantity].append(srczarr)
                            

        else:
            raise NotImplemented('help!')

    @property
    def include(self):
        if not hasattr(self, '_include'):
            self._include = ['lf', 'smf']
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
            
                for i, dataset in enumerate(self.redshifts[quantity]):
                    for j, redshift in enumerate(self.data[quantity][i]):
                        M = self.data[quantity][i][redshift]['M']
                        
                        # These could still be in log10 units
                        phi = self.data[quantity][i][redshift]['phi']
                        err = self.data[quantity][i][redshift]['err']
                        
                        if hasattr(M, 'mask'):
                            self._mask.extend(M.mask)
                        else:
                            self._mask.extend(np.zeros_like(M))
                            
                        self._xdata_flat.extend(M)
                        
                        if self.units[quantity][i] == 'log10':
                            self._ydata_flat.extend(10**phi)
                        else:    
                            self._ydata_flat.extend(phi)
                
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
                for i, dataset in enumerate(self.redshifts):
                    for j, redshift in enumerate(self.data[i]):
                        self._xdata.append(self.data[i][redshift]['M'])
                        self._ydata.append(self.data[i][redshift]['phi'])
                        self._error.append(self.data[i][redshift]['err'])
                    
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
            
        fn = '%s.data.pkl' % prefix
        
        if os.path.exists(fn) and (not clobber):
            print "%s exists! Set clobber=True to overwrite." % fn
            return
                
        f = open(fn, 'wb')
        pickle.dump((self.xdata, self.ydata, self.redshifts, self.error), f)
        f.close()
     
    
