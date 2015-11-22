"""

SFE.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Nov 22 12:01:50 PST 2015

Description: 

"""

import numpy as np
from .ParameterFile import ParameterFile

z0 = 4. # just needs to be lower than all redshifts we consider

class SFE(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
        
    def __call__(self, z, M, *coeff):
        return 10**self._log_fstar(z, M, *coeff)
        
    def _log_fstar(self, z, M, *coeff):    
        if self.Mfunc == self.zfunc == 'logpoly':            
            logf = coeff[0] + coeff[1] * np.log10(M / 1e10) \
                 + coeff[2] * ((1. + z) / 8.) \
                 + coeff[3] * ((1. + z) / 8.) * np.log10(M / 1e10) \
                 + coeff[4] * (np.log10(M / 1e10))**2. \
                 + coeff[5] * (np.log10(M / 1e10))**3.
        elif (self.Mfunc == 'lognormal') and (self.zfunc == 'linear'):
            logM = np.log10(M)
            
            fstar_of_z = lambda zz: coeff[0] + coeff[1] * (zz - z0) / z0 
            Mpeak_of_z = lambda zz: coeff[2] + coeff[3] * (zz - z0) / z0  # log
            sigma_of_z = lambda zz: coeff[4] + coeff[5] * (zz - z0) / z0 
            
            f = fstar_of_z(z) * np.exp(-(logM - Mpeak_of_z(z))**2 / 2. / 
                sigma_of_z(z)**2)
            logf = np.log10(f)
        elif (self.Mfunc == 'lognormal') and (self.zfunc == 'const'):
            logM = np.log10(M)
             
            f = coeff[0] * np.exp(-(logM - coeff[1])**2 / 2. / 
                coeff[2]**2)
            logf = np.log10(f)
        
        return logf    
        
    @property
    def Npops(self):
        return self.pf.Npops
        
    @property
    def pop_id(self):
        # Pop ID number for HAM population
        if not hasattr(self, '_pop_id'):
            for i, pf in enumerate(self.pf.pfs):
                if pf['pop_model'] == 'ham':
                    break
            
            self._pop_id = i
        
        return self._pop_id        
        
    @property
    def irrelevant(self):
        if not hasattr(self, '_irrelevant'):
            if self.pf.pfs[self.pop_id]['pop_model'] != 'ham':
                self._irrelevant = True
            else:
                self._irrelevant = False
        
        return self._irrelevant
            
        
    @property
    def Mfunc(self):
        return self.pf.pfs[self.pop_id]['pop_fstar_M_func']
    
    @property
    def zfunc(self):
        return self.pf.pfs[self.pop_id]['pop_fstar_z_func']    
    
    @property
    def Mext(self):
        return self.pf.pfs[self.pop_id]['pop_fstar_M_extrap']
        
    @property
    def zext(self):
        return self.pf.pfs[self.pop_id]['pop_fstar_z_extrap']        
        
    @property
    def Ncoeff(self):
        if not hasattr(self, '_Ncoeff'):  
            if self.Mfunc == self.zfunc == 'logpoly':            
                self._Ncoeff = 6
            elif (self.Mfunc == 'lognormal') and (self.zfunc == 'linear'):
                self._Ncoeff = 6
            elif (self.Mfunc == 'lognormal') and (self.zfunc == 'const'):
                self._Ncoeff = 3
            
        return self._Ncoeff
        
    @property
    def guesses(self):
        if not hasattr(self, '_guesses'):  
            if self.Mfunc == self.zfunc == 'logpoly':            
                self._guesses = -1. * np.ones(6)
            elif (self.Mfunc == 'lognormal') and (self.zfunc == 'linear'):
                self._guesses = np.array([0.25, 0.05, 11., 0.05, 0.5, 0.05]) 
            elif (self.Mfunc == 'lognormal') and (self.zfunc == 'const'):
                self._guesses = np.array([0.25, 11., 0.5])
    
        return self._guesses    
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 