"""

OptimizeSpectrum.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 25 09:00:27 2013

Description: 

"""

import copy
import numpy as np
from ..static import Grid
from ..sources import BlackHole, Star
from ..physics.Constants import erg_per_ev 
from ..util.ParameterFile import ParameterFile
from ..physics.CrossSections import PhotoIonizationCrossSection as sigma_E

try:
    import h5py
except ImportError:
    pass    
    
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
    return li[0:N/2], li[N/2:]    
    
class SpectrumOptimization(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)

        # Use Grid class to carry info about absorbing species
        self.grid = Grid(**kwargs)
        self.grid.set_properties(**self.pf)
        
        # Shortcut
        self.Z = self.grid.Z
        
        # 
        self.mcmc = False
        
        # Should make property/setter
        self.thinlimit = True
        
        # Needs to be made more general!
        self.rs = Star(grid=self.grid, init_tabs=True, **kwargs)
        
        # Forget why this has to be set by hand...
        self.rs.grid = self.grid        

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
            for i in xrange(self.nfreq * 2):
                if i < self.nfreq:
                    self._guess.append(np.random.rand() * \
                    (self.limits[i][1] - self.limits[i][0]) + self.limits[i][0])
                else:
                    self._guess.append(self.rs.Spectrum(self._guess[i - self.nfreq]))
    
            tot = np.sum(self._guess[self.nfreq:])
            for i in xrange(self.nfreq, self.nfreq * 2):
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
    
    def run(self, steps=1, **kwargs):
        
        if rank == 0:
            print('Finding optimal {0}-bin discrete SED...'.format(self.nfreq))
        
        logL = []
        results = []
        for i in range(steps):
            self.__call__(**kwargs)
            logL.append(self.logL)
            results.append(self.pars.copy())
    
        self.all_logL = logL
        self.all_results = results
        
        if rank == 0:
            print('Optimization complete.')
    
    def __call__(self, **kwargs):
        """
        Construct optimal discrete spectrum for a given radiation source.
        
        Need: Column density range for all species.
            logN = list of arrays, each element should be logN for corresponding
                entry in Z.
        """
        
        
        self.sampler = basinhopping(self.cost, x0=self.guess, 
            accept_test=self._accept_test)
         
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
        to_sum[E < Eth] = 0.0
        summed = np.sum(to_sum, axis = -1)
        return np.log10(summed / erg_per_ev)

    def DiscretePsi(self, E, LE, tau_E, Eth):
        to_sum = LE * np.exp(-tau_E)
        to_sum[E < Eth] = 0.0
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
        
        for i in xrange(self.rs.tab.elements_per_table):
            for j, absorber in enumerate(self.grid.absorbers):
                
                loc = list(self.rs.tab.indices_N[i])
                loc.append(j)
                    
                tmp = np.array(map(lambda EE: self.rs.tab.PartialOpticalDepth(EE, 
                    float(self.rs.tab.Nall[i]), absorber), E))
                                
                tau_E[tuple(loc)] = tmp
                    
        return tau_E
        
    def dump(self, fn):
        """
        Write optimization result to HDF5 file.
        """    
        
        if rank > 0:
            return
            
        f = h5py.File(fn, 'w')
        f.create_dataset('chain', data=self.sampler.chain)
        
        if self.mcmc:
            f.create_dataset('post', data=self.sampler.post)
        else:
            f.create_dataset('cost', data=self.sampler.cost)
        
        print('Wrote chain to {!s}.'.format(fn))    
        f.close()
        
    

