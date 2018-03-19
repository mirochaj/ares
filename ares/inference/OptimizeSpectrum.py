"""

OptimizeSpectrum.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 25 09:00:27 2013

Description: 

"""

import os
import copy
import pickle
import numpy as np
from ..static import Grid
from ..sources import BlackHole, Star
from ..physics.Constants import erg_per_ev 
from ..util import ParameterFile, ProgressBar
from ..physics.CrossSections import PhotoIonizationCrossSection as sigma_E

#try:
#    import h5py
#except ImportError:
#    pass    
    
try:
    from scipy.optimize import basinhopping
except ImportError:
    pass
    
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
    
def halve_list(li):
    N = len(li)
    return li[0:N//2], li[N//2:]    
    
class SpectrumOptimization(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
        self.base_kwargs = self.pf

        # Use Grid class to carry info about absorbing species
        self.grid = Grid(**kwargs)
        self.grid.set_properties(**self.pf)
        
        # Shortcut
        self.Z = self.grid.Z
        
        self.mcmc = False
        
        # Needs to be made more general!
        self.rs = Star(**kwargs)
        
        # Forget why this has to be set by hand...
        self.rs.grid = self.grid        

    @property
    def thinlimit(self):
        if not hasattr(self, '_thinlimit'):
            self._thinlimit = False
        return self._thinlimit
    
    @thinlimit.setter
    def thinlimit(self, value):
        self._thinlimit = value

    @property
    def logN(self):
        return self.rs.tab.logN
        
    @property
    def integrals(self):
        if not hasattr(self, '_integrals'):
            self._integrals = copy.deepcopy(self.rs.tab.IntegralList)
            self._integrals.pop(self._integrals.index('Tau'))
        
        return self._integrals
        
    @integrals.setter
    def integrals(self, value):
        self._integrals = value
    
    @property
    def tau_dims(self):
        if not hasattr(self, '_tau_dims'):    
        # Figure out size of optical depth array [np.prod(logN.shape) x nfreq]
            self._tau_dims = list(self.rs.tab.dimsN.copy())
            self._tau_dims.append(len(self.Z))
            self._tau_dims.append(self.nfreq)
            
            self._tab_dims = list(self.rs.tab.dimsN.copy())
        return self._tau_dims
        #self.tab_dims.insert(0, len(self.Z))
        #self.tab_dims.insert(0, len(self.integrals))
    
    @property
    def tab_dims(self):
        return self.tau_dims
    
    @property
    def nfreq(self):
        return self._nfreq
    @nfreq.setter
    def nfreq(self, value):
        self._nfreq = int(value)
        
    @property
    def step_size(self):
        if not hasattr(self, '_step_size'):
            self._step_size = 0.5
        return self._step_size
    @step_size.setter
    def step_size(self, value):
        self._step_size = value
    
    @property
    def limits(self):
        if not hasattr(self, '_limits'):
            self._limits = [(self.rs.Emin, self.rs.Emax)] * self.nfreq \
                + [(0, 1)] * self.nfreq
        return self._limits
        
    @limits.setter
    def limits(self, value):
        if len(value) == 2:
            Elim, Llim = value
            self._limits = [Elim] * self.nfreq + [Llim] * self.nfreq
        else:
            assert len(value) == (2 * self.nfreq)
            self._limits = value
                
    @property
    def guess(self):
        if not hasattr(self, '_guess'):
            self._guess = []
            for i in range(self.nfreq * 2):
                if i < self.nfreq:
                    self._guess.append(np.random.rand() * \
                    (self.limits[i][1] - self.limits[i][0]) + self.limits[i][0])
                else:
                    self._guess.append(self.rs.Spectrum(self._guess[i - self.nfreq]))
    
            tot = np.sum(self._guess[self.nfreq:])
            for i in range(self.nfreq, self.nfreq * 2):
                self._guess[i] /= tot
            
            self._guess = np.array(self._guess)
    
        return self._guess
    @guess.setter
    def guess(self, value):
        assert len(value) == 2 * self.nfreq, "len(guess) must be = 2*nfreq!"
        self._guess = np.array(value)
    
    def _accept_test(self, **kwargs):
                                       
        # If cost is not finite
        if not np.isfinite(kwargs['f_new']):
            return False             
                               
        # Force normalization to be in interval [0, 1]
        # Force energies to be within provided limits        
        for i, x in enumerate(kwargs['x_new']):
            if not (self.limits[i][0] <= x <= self.limits[i][1]):
                return False

        # Force sum of normalizations to be less than 1
        if np.sum(kwargs['x_new'][self.nfreq:]) > 1:
            return False
        
        return True
    
    @property
    def basinhopper_kwargs(self):
        """
        Possibilities
        -------------
        niter: int
            Number of basin hopping iterations.
        T : int, float
            Temperature parameter. Should be comparable to separation between
            local minima.
        stepsize : int, float
            Initial displacement.
        interval : int
            How often to update stepsize.
        niter_success : int
            Stop if global minimum doesn't change for this many steps.
        disp : bool
            Display status messages?
        
        """
        if not hasattr(self, '_basinhopper_kwargs'):
            self._basinhopper_kwargs = {}
        return self._basinhopper_kwargs
        
    @basinhopper_kwargs.setter
    def basinhopper_kwargs(self, value):
        if not hasattr(self, '_basinhopper_kwargs'):
            self._basinhopper_kwargs = {}
            
        if type(value) is dict:
            self._basinhopper_kwargs.update(value)
        elif type(value) is tuple:
            self._basinhopper_kwargs.update(dict(value))
        else:
            raise TypeError('Must supply dict or tuple for basinhopper_kwargs')
    
    def run(self, prefix, repeat=1, clobber=False, restart=False, save_freq=10):
        
        if rank == 0:
            print('Finding optimal {0}-bin discrete SED...'.format(self.nfreq))
        
        prefix_by_proc = '{0!s}.{1!s}'.format(prefix, str(rank).zfill(3))
                
        actual_restart = True        
        if restart:
            if not os.path.exists("{0}.chain.pkl".format(prefix_by_proc)):
                print("Can't be a restart, {0}*.pkl not found!".format(prefix))
                print("Continuing on with new run.")
                actual_restart = False
        else:
            actual_restart = False        
                            
        if not actual_restart:
            if os.path.exists("{0}.chain.pkl".format(prefix_by_proc)):
                if not clobber:
                    raise IOError("{0}*.pkl exists! Set clobber=True to overwrite".format(prefix))
                else:
                    os.system("rm -f {!s}.*.pkl".format(prefix_by_proc))    
                
            # Save metadata
            with open("{0}.binfo.pkl".format(prefix), 'wb') as f:
                pickle.dump(self.base_kwargs, f)
            
            # Save metadata
            with open("{0}.pinfo.pkl".format(prefix), 'wb') as f:
                pinfo = ['E{0}'.format(i) for i in range(self.nfreq)] \
                      + ['L{0}'.format(i) for i in range(self.nfreq)]
                pickle.dump((pinfo, [False] * len(pinfo)), f)
                 
        _logL = np.zeros(repeat)
        _results = np.zeros((repeat, 2 * self.nfreq))         
                         
        pb = self.pb = ProgressBar(repeat, use=repeat > 1, name='sedop')
        
        ct = 0
        for i in range(repeat):
            
            if i % size != rank: 
                continue
            
            if not pb.has_pb:
                pb.start()
                        
            self.__call__()
            
            _logL[i] = self.logL * 1
            _results[i] = self.pars.copy()
                        
            ct += 1
            pb.update(ct * size)
            
            # Checkpoints
            #if (ct % save_freq != 0):
            #    continue
                        
            with open('{0}.chain.pkl'.format(prefix_by_proc), 'ab') as f:
                pickle.dump([np.atleast_2d(self.pars)], f)
                
            with open('{0}.logL.pkl'.format(prefix_by_proc), 'ab') as f:
                pickle.dump([self.logL], f)
            
        pb.finish()
        
        if size > 1:
            self.logL = np.zeros_like(_logL)
            MPI.COMM_WORLD.Allreduce(_logL, self.logL)
            
            self.chain = np.zeros_like(_results)
            MPI.COMM_WORLD.Allreduce(_results, self.chain)
        else:
            self.logL = _logL
            self.chain = _results
        
        if rank == 0:
            print('Optimization complete.')
            
        #self.dump(prefix)    
    
    def __call__(self, **kwargs):
        """
        Construct optimal discrete spectrum for a given radiation source.
        
        Need: Column density range for all species.
            logN = list of arrays, each element should be logN for corresponding
                entry in Z.
        """
        
        self.sampler = basinhopping(self.cost, x0=self.guess, 
            accept_test=self._accept_test, **self.basinhopper_kwargs)
         
        E, L = self.sampler.x[0:self.nfreq], self.sampler.x[self.nfreq:] 
        loc = np.argsort(E)
         
        self.logL = self.sampler.fun
        self.pos = self.pars = self.results = np.concatenate((E[loc], L[loc]))
         
    def cost(self, pars):
        E, LE = pars[0:self.nfreq], pars[self.nfreq:]
        #bfx = sigma_E(E)        
                
        # Compute optical depth for all combinations of column densities
        if self.thinlimit:
            tau = np.zeros(self.tau_dims)
        else:
            tau = self.tau(E)
                        
        # Compute discrete versions of phi & psi
        # NOTE: if ionization thresholds all below smallest emission energy,
        # integrals for all species are identical.
        discrete_tables = self.discrete_tabs(E, LE, tau)
                
        # Keep only one element if we're doing optically thin optimization
        if self.thinlimit:
            mask = 0
        else:
            mask = Ellipsis
            
        # Compute cost
        cost = 0.0
        for i, integral in enumerate(self.integrals):
            for j, absorber in enumerate(self.grid.absorbers):
                name = self.tab_name(integral, absorber)

                cont = self.rs.tabs[name]
                disc = discrete_tables[name]
                
                # Difference array (finite elements only)
                diffarr = np.atleast_1d(np.abs(disc - cont)[mask])

                if not np.all(np.isfinite(diffarr)):
                    return np.inf
                
                if self.thinlimit:
                    cost = np.max(diffarr)
                else:
                    cost = np.mean(diffarr)
                                        
        return cost
    
    def tab_name(self, integral, absorber):
        return 'log{0!s}_{1!s}'.format(integral, absorber)

    def discrete_tabs(self, E, LE, tau=None):
        """
        Compute values of integral quantities assuming discrete emission.
        """
                
        if tau is None:
            tau = self.tau(E)
        
        discrete_tables = {}
        for i, integral in enumerate(self.integrals):
            for j, absorber in enumerate(self.grid.absorbers):
                Eth = self.grid.ioniz_thresholds[absorber]
                                
                tmp = np.zeros(self.tab_dims)
                if integral == 'Phi':
                    tmp = self.DiscretePhi(E, LE, tau[:,j,:], Eth)
                if integral == 'Psi':
                    tmp = self.DiscretePsi(E, LE, tau[:,j,:], Eth)
                        
                discrete_tables[self.tab_name(integral, absorber)] = tmp.copy()        
                            
        return discrete_tables   
        
    def DiscretePhi(self, E, LE, tau_E, Eth):
        to_sum = LE * np.exp(-tau_E) / E
        to_sum[:,E < Eth] = 0.0
        summed = np.sum(to_sum, axis=-1)
        return np.log10(summed / erg_per_ev)

    def DiscretePsi(self, E, LE, tau_E, Eth):
        to_sum = LE * np.exp(-tau_E)
        to_sum[:,E < Eth] = 0.0
        summed = np.sum(to_sum, axis = -1)
        return np.log10(summed)
    
    def tau(self, E):
        """
        Compute total optical depth (over all species) as a function of 
        discrete emission energy E (eV).
        
        Parameters
        ----------
        E : int, float
            Photon energy in eV.
        
        Returns
        -------
        """
        
        E = np.atleast_1d(E)
        tau_E = np.zeros(self.tau_dims)
        
        for i in range(self.rs.tab.elements_per_table):
            for j, absorber in enumerate(self.grid.absorbers):
                
                loc = list(self.rs.tab.indices_N[i])
                loc.append(j)
                    
                tmp = np.array([self.rs.tab.PartialOpticalDepth(EE, float(self.rs.tab.Nall[i]), absorber) for EE in E])
                                
                tau_E[tuple(loc)] = tmp
                    
        return tau_E
        
    #def dump(self, prefix, clobber):
    #    """
    #    Write optimization result to HDF5 file.
    #    """    
    #    
    #    if rank > 0:
    #        return
    #        
    #    f = open(fn, 'wb')
    #    pickle.dump(self.chain)
    #        
    #    f = h5py.File(fn, 'w')
    #    f.create_dataset('chain', data=self.chain)
    #    f.create_dataset('cost', data=self.logL)
    #    f.close()
    #    
    #    print('Wrote chain to {!s}.'.format(fn))    
        
    

