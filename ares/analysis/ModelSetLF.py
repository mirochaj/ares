"""

GLFSet.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Oct 24 10:42:45 PDT 2015

Description: 

"""

import re
import ares
import numpy as np
import matplotlib.pyplot as pl
from .ModelSet import ModelSet
from ..populations.Galaxy import param_redshift
from ..util.SetDefaultParameterValues import SetAllDefaults

class LuminosityFunctionSet(ModelSet):
    """
    Basically a ModelSet instance with routines specific to the high-z
    galaxy luminosity function.
    """
    
    def get_data(self, z):
        i = self.data['z'].index(z)
        
        return self.data['x'][i], self.data['y'][i], self.data['err'][i]
    
    def ReconstructedLF(self, z, ax=None, fig=1, N=100, 
        resample=True, **kwargs):
        """
        Plot constraints on the luminosity function at given z.
        """

        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True

        # Find relevant elements in chain
        samples = {}
        for i, key in enumerate(self.parameters):
            if not re.search('_lf_', key):
                continue
                
            prefix, redshift = param_redshift(key)
            if z != redshift:
                continue
                
            samples[prefix[0:prefix.find('{')]] = self.chain[:,i]
        
        i = self.data['z'].index(z)
        
        if resample:
            mi, ma = self.data['x'][i][0], self.data['x'][i][-1]
            M = np.linspace(mi-2, ma+2, 200)
        else:
            M = np.array(self.data['x'][i])
        
        pst = 10**samples['pop_lf_pstar']
        Mst = samples['pop_lf_Mstar']            
        a = samples['pop_lf_alpha']
        
        for i in range(N):
            phi = 0.4 * np.log(10.) * pst[i] \
                * (10**(0.4 * (Mst[i] - M)))**(1. + a[i]) \
                * np.exp(-10**(0.4 * (Mst[i] - M)))
            
            ax.semilogy(M, phi, **kwargs)
        
        pl.draw()
        return ax
            
    def PlotData(self, z, ax=None, fig=1, **kwargs):
        M, phi, err = self.get_data(z)
        
        ax, gotax = self.get_ax(ax, fig)
            
        ax.errorbar(M, phi, yerr=err, fmt='none', **kwargs)
        ax.set_xlabel(r'$M$')
        ax.set_ylabel(r'$\phi \ (\mathrm{cMpc}^{-3} \ \mathrm{mag}^{-1})$')
        ax.set_yscale('log')
        pl.draw()
        
        return ax
            
    def StellarSED(self, ax=None, fig=1, N=500, src='leitherer1999', **kwargs):
        """
        Make a triangle plot of the stellar SED parameters.
        """
        
        pars = []
        for par in self.parameters:
            if re.search('pop_Z', par):
                pars.append(par)
                break
                
        data = self.ExtractData(pars)
        
        if N > self.chain.shape[0]:
            N = self.chain.shape[0]
            
        src = read_lit(src)
        i_Z = self.parameters.index(pars[0])
        
        Z = []; kappa_UV = []; Nion = []; Nlw = []
        for i in range(500):
            
            j = np.random.randint(0, N)
            
            kw = {par: data[par][j] for par in data}
            
            pop = src.StellarPopulation(**kw)
            
            Z.append(self.chain[j,i_Z])
            kappa_UV.append(pop.kappa_UV)
            Nion.append(pop.Nion)
            Nlw.append(pop.Nlw)
            
        
        parnames = ['pop_Z', 'pop_kappa_UV', 'pop_Nion', 'pop_Nlw']
            
        
        
                
            
    def alpha_hmf(self, z, M=1e9):
        """
        Compare constraints on the slope of the LF vs. the slope of the HMF.
        """
        
        if not hasattr(self, '_pop'):
            self._pop = ares.populations.GalaxyPopulation(verbose=False)
        
        i_z = np.argmin(np.abs(z - self._pop.halos.z))
        
        alpha_of_M = np.diff(np.log(self._pop.halos.dndm[i_z,:])) \
            / np.diff(self._pop.halos.lnM)
        
        return np.interp(M, self._pop.halos.M[0:-1], alpha_of_M)
            
    def SlopeEvolution(self):
        pass
        
    def MstarEvolution(self):
        pass
    
    def LstarEvolution(self):
        pass
        
    def PredictLF(self, z, name='galaxy_lf', **kwargs):
        """
        Plot luminosity function at redshift z, as predicted by fits.
        """
        pass
        #lf = self.ExtractData(name)
            
        
            
        
        