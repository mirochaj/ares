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
from ..util import read_lit
import matplotlib.pyplot as pl
from .ModelSet import ModelSet
from ..populations.Galaxy import param_redshift
from ..util.SetDefaultParameterValues import SetAllDefaults

ln10 = np.log(10.)

phi_of_M = lambda M, pstar, Mstar, alpha: 0.4 * ln10 * pstar \
    * (10**(0.4 * (Mstar - M)*(1. + alpha))) \
    * np.exp(-10**(0.4 * (Mstar - M)))

class ModelSetLF(ModelSet):
    """
    Basically a ModelSet instance with routines specific to the high-z
    galaxy luminosity function.
    """
    
    def get_data(self, z):
        i = self.data['z'].index(z)
        
        return self.data['x'][i], self.data['y'][i], self.data['err'][i]
    
    def SFE(self, z, ax=None, fig=1, N=1, **kwargs):
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
            
        if N == 1:
            kwargs = [self.max_likelihood_parameters]
        else:
            raise NotImplemented('')    
        
    def LuminosityFunction(self, z, ax=None, fig=1, N=100, 
        compare_to=None, popid=0, name='galaxy_lf', best_only=True, 
        assume_schecter=True, scatter_kwargs={}, Mlim=(-24, -10), **kwargs):
        """
        Plot the luminosity function used to train the SFE.
        
        """
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
        
        # Plot fits compared to observational data
        
        M = np.arange(Mlim[0], Mlim[1], 0.05)
        lit = read_lit(compare_to)
    
        if (compare_to is not None) and (z in lit.redshifts):
            ax.errorbar(lit.data['lf'][z]['M'], lit.data['lf'][z]['phi'],
                yerr=lit.data['lf'][z]['err'], fmt='o', zorder=10,
                **scatter_kwargs)
                
            # Plot B15 constraints too?
            
        if assume_schecter:
            pars = self.max_likelihood_parameters
            
            Mstar = pars['pop_lf_Mstar[%.2g]{%i}' % (z, popid)]
            pstar = pars['pop_lf_pstar[%.2g]{%i}' % (z, popid)]
            alpha = pars['pop_lf_alpha[%.2g]{%i}' % (z, popid)]

            phi = phi_of_M(M, pstar, Mstar, alpha)
            ax.semilogy(M, phi, ls='--', **kwargs)
        else:
            info = self.blob_info(name)
            ivars = self.blob_ivars[info[0]]
            
            i = list(ivars[0]).index(z)
            M = ivars[1]
            
            loc = np.argmax(self.logL)
            
            phi = []
            for i, mag in enumerate(M):
                data, is_log = self.ExtractData(name, ivar=[z, M[i]])
                
                if best_only:
                    phi.append(data[name][loc])
                else:
                    phi.append(data[name])    
                
            phi = np.array(phi)
                
            if best_only:    
                ax.semilogy(M, phi, **kwargs)
            else:
                for i in range(N):
                    ax.semilogy(M, phi[:,i], **kwargs)

        ax.set_xlabel(r'$M_{\mathrm{UV}}$')
        ax.set_ylabel(r'$\phi(M)$')
        ax.set_ylim(1e-7, 1e-1)
        ax.set_xlim(-25, -15)
        
        return ax
        
    def FaintEndSlope(self, z, mag=None, ax=None, fig=1, N=100, 
        name='alpha_lf', best_only=False, **kwargs):
        """
        
        """
        
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
            
        info = self.blob_info(name)
        ivars = self.blob_ivars[info[0]]
        
        i = list(ivars[0]).index(z)
        M = ivars[1]
        
        loc = np.argmax(self.logL)
        
        alpha = []
        for i, mag in enumerate(M):
            data, is_log = self.ExtractData(name, ivar=[z, M[i]])
            
            if best_only:
                alpha.append(data[name][loc])
            else:
                alpha.append(data[name])
         
        alpha = np.array(alpha) 
                
        if best_only:    
            ax.plot(M, alpha, **kwargs)
        else:
            for i in range(N):
                ax.plot(M, alpha[:,i], **kwargs)   
                
        ax.set_xlabel(r'$M_{\mathrm{UV}}$')
        ax.set_ylabel(r'$\alpha$')
        #ax.set_ylim(1e-7, 1e-1)
        #ax.set_xlim(-25, Mlim[1])
        
        return ax         
            
    def AddConstraints(self, z, compare_to, ax):
        """
        
        """
        lit = read_lit(compare_to)
        
        if (compare_to is not None) and (z in lit.redshifts):
            ax.errorbar(lit.data['lf'][z]['M'], lit.data['lf'][z]['phi'],
                yerr=lit.data['lf'][z]['err'], fmt='o', zorder=10,
                **scatter_kwargs)
    
    def ReconstructedLF(self, z, ax=None, fig=1, N=1, resample=True, **kwargs):
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
        
        
        
        lf = self.ExtractData(name)
            
        
            
        
        