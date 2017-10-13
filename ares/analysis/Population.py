"""

Population.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri May 29 18:30:49 MDT 2015

Description: 

"""

import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import cumtrapz
from ..util.ReadData import read_lit
from ..util.Aesthetics import labels
from ..physics.Constants import s_per_yr

class Population(object):
    def __init__(self, pop):
        assert pop.is_ham_model, "These routines only apply for HAM models!"
        self.pop = pop
        
    def LuminosityFunction(self, z, ax=None, fig=1, cumulative=False, **kwargs):

        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
            
        Mh, Lh = self.pop.ham.Lh_of_M(z)
                
        k = np.argmin(np.abs(z - self.pop.ham.halos.z))
        dndlnm = self.pop.ham.halos.dndlnm[k]
        
        integrand = Lh * dndlnm
        Lhc = cumtrapz(integrand, x=self.pop.ham.halos.lnM, initial=0.0)
        
        ax.loglog(Mh, Lhc / Lhc[-1], **kwargs)
        ax.set_ylim(1e-3, 2)
        
        return ax
        
    def ObservedLF(self, source, z, ax=None, fig=1):
        
        data = read_lit(source)
        
        assert z in data.redshifts, "Requested redshift not in source {!s}".format(source)
        
        uplims = np.array(data.data['lf'][z]['err']) < 0
        
        err_lo = []; err_hi = []
        for hh, err1 in enumerate(o13.data['lf'][z]['err']):
            if uplims[hh]:
                err_hi.append(0.0)
                err_lo.append(0.8 * o13.data['lf'][z]['phi'][hh])
            else:
                err_hi.append(err1)
                err_lo.append(err1)

        mp.grid[i].errorbar(o13.data['lf'][z]['M'], o13.data['lf'][z]['phi'],
            yerr=(err_lo, err_hi), uplims=list(uplims), fmt='o', 
            color='g', zorder=10, mec='g', ms=3)    
        
    def MassToLight(self, z, ax=None, fig=1, scatkw={}, **kwargs):
        """
        Plot the halo mass to luminosity relationship yielded by AM.
        """        
                
                
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
            
        Mh, Lh = self.pop.ham.Lh_of_M(z)
        
        
        ax.loglog(Mh, Lh, **kwargs)
        ax.set_ylim(1e23, 1e33)
    
        if z in self.pop.ham.redshifts:
            k = self.pop.ham.redshifts.index(z)
            ax.scatter(self.pop.ham.MofL_tab[k], self.pop.ham.LofM_tab[k], 
                **scatkw)
                
        ax.set_xlim(1e6, 1e15)        
        ax.set_xlabel(labels['Mh'])
        ax.set_ylabel(labels['Lh'])

        pl.draw()
    
        return ax
        
    def SFE(self, z, ax=None, fig=1, scatkw={}, **kwargs):
        
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
    
        Marr = np.logspace(8, 14)    
        
        fast = self.pop.ham.SFE(z=z, M=Marr)
        ax.loglog(Marr, fast, **kwargs)
    
        if z in self.pop.ham.redshifts:
            k = self.pop.ham.redshifts.index(z)
            ax.scatter(self.pop.ham.MofL_tab[k], self.pop.ham.fstar_tab[k],
                **scatkw)
        
        ax.set_xlabel(labels['Mh'])
        ax.set_ylabel(labels['fstar'])
        pl.draw()
        
        return ax
        
    def HMF_vs_LF(self, z, ax=None, fig=1, mags=False, data=None, **kwargs):
        """
        Plot the halo mass function vs. the stellar mass function.
    
        Plot SFR function instead?
    
        Parameters
        ----------
        z : int, float
            Redshift of interest.
        mags : bool
            If True, luminosity function will be plotted in AB magnitude,
            otherwise, in rest-frame 1600 Angstrom luminosity
    
        """
    
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
    
        # HMF
        Mh = self.pop.ham.halos.M
        i_z = np.argmin(np.abs(z - self.pop.ham.halos.z))
        nofm = self.pop.ham.halos.dndm[i_z]
        above_Mmin = self.pop.halos.M >= self.pop.ham.Mmin[i_z]

        phi_hmf = nofm[0:-1] * np.diff(Mh) * above_Mmin[0:-1]

        ax.loglog(Mh[0:-1], phi_hmf)        

        # Now, LF
        xLF, phi = self.pop.ham.LuminosityFunction(z, mags=mags)

        phi_lf = phi[0:-1] * np.diff(xLF)

        ax2 = ax.twiny()
        ax2.semilogy(xLF[0:-1], phi_lf, 'r')
        
        # Change tick colors
        ax2.spines['top'].set_color('red')
        ax2.xaxis.label.set_color('red')
        ax2.tick_params(axis='x', colors='red', which='both')

        if mags:
            ax2.set_xlabel(r'Galaxy Luminosity $(M_{\mathrm{UV}})$', color='r')
            ax2.set_xlim(-6, -25)
        else:
            ax2.set_xlabel(r'Galaxy Luminosity $(L_{\mathrm{UV}} / \mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{Hz}^{-1})$', 
                color='r')
            ax2.set_xscale('log')                
            ax2.set_xlim(1e25, 1e33)
            
        ax.set_xlabel(r'Halo Mass $(M_h / M_{\odot})$')
        ax.set_xlim(1e6, 1e13)
        ax.set_ylim(1e-9, 2)
        
        ax.set_ylabel(r'Number Density $(\phi / \mathrm{cMpc}^{-3})$')

        pl.draw()        
    
        return ax, ax2
        
    def HMF_vs_SMF(self, z, ax=None, fig=1, **kwargs):
        """
        Plot the halo mass function vs. the stellar mass function.
        
        Plot SFR function instead?
        
        Parameters
        ----------
        z : int, float
            Redshift of interest.
        mags : bool
            If True, luminosity function will be plotted in AB magnitude,
            otherwise, in rest-frame 1600 Angstrom luminosity
            
        """
        
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
        
        # HMF
        Mh = self.pop.ham.halos.M
        i_z = np.argmin(np.abs(z - self.pop.ham.halos.z))
        nofm = self.pop.ham.halos.dndm[i_z]
        above_Mmin = self.pop.halos.M >= self.pop.ham.Mmin[i_z]
        
        ax.loglog(Mh, nofm * Mh * above_Mmin)        
                
        # Now, need SFR
        Mh_, Ms_ = self.SMHM(z)
        nofm_ = np.exp(np.interp(np.log(Mh_), np.log(self.pop.halos.M), 
            np.log(self.pop.halos.dndm[i_z])))
                
        above_Mmin = Mh_ >= self.pop.ham.Mmin[i_z]
                
        ax2 = ax.twiny()
        ax2.loglog(Ms_, nofm_ * Mh_ * above_Mmin, 'r')
        ax2.set_xlabel(r'$M_{\ast} / M_{\odot}$', color='r')
        for tl in ax2.get_xticklabels():
            tl.set_color('r')
        for tl in ax2.get_xticks():
            tl.set_color('r')    
                
        ax.set_xlabel(r'$M_h / M_{\odot}$')
        ax.set_xlim(0.8 * self.pop.ham.Mmin[i_z], 1e14)
        ax.set_ylim(1e-13, 1e2)
        ax.set_ylabel('Number Density')
                
        pl.draw()        
        
        return ax
        
    def SMHM(self, z, ratio=False, Nz=100, zmax=40):
        """
        Compute the stellar-mass halo-mass (SMHM) relation.
        """
        
        # Array of formation redshifts from high to low
        zarr = np.linspace(z, zmax, Nz)[-1::-1]

        Mh_all = []
        Mstar_all = []
        for i in range(Nz):
            
            # Obtain halo mass for all times since zmax = zarr[i]
            zz, Mh = self.pop.ham.Mh_of_z(zarr[i:])
            
            Mh_all.append(Mh[-1])
            
            # Compute stellar mass
            Macc_of_z = self.pop.ham.Macc(zz, Mh)
            fstar_of_z = np.array([self.pop.ham.fstar(zz[j], Mh[j]) \
                for j in range(len(zz))])

            dtdz = -self.pop.cosm.dtdz(zz)
            
            eta = np.interp(zz, self.pop.ham.halos.z, self.pop.ham.eta)
            
            sfr_of_z = fstar_of_z * self.pop.cosm.fbaryon * Macc_of_z * eta
            integrand = sfr_of_z * dtdz / s_per_yr
            
            Mstar = np.trapz(integrand, x=zz)
            Mstar_all.append(Mstar)
            
        return np.array(Mh_all)[-1::-1], np.array(Mstar_all)[-1::-1]
        
    def SamplePosterior(self, x, func, pars, errors, Ns=1e3):
        """
        Draw random samples from posterior distributions.
        
        Parameters
        ----------
        x : np.ndarray
            Independent variable of input function, `func`.
        func : function 
            Function used to generate samples. Currently, support for single
            independent variable (`x`) and an arbitrary number of keyword arguments.        
        pars : dict
            Dictionary of best-fit parameter values
        errors : dict
            Dictionary of 1-sigma errors on the best-fit parameters
        Ns : int
            Number of samples to draw

        Examples
        --------
        >>> import ares
        >>> import numpy as np
        >>> import matplotlib.pyplot as pl
        >>>
        >>> r15 = ares.util.read_lit('robertson2015')
        >>> z = np.arange(0, 8, 0.05)
        >>> pop = ares.analysis.Population(r15)
        >>> models = pop.SamplePosterior(z, r15.SFRD, r15.sfrd_pars, r15.sfrd_err)
        >>>
        >>> for i in range(int(models.shape[1])):
        >>>     pl.plot(z, models[:,i], color='b', alpha=0.05)

        Returns
        -------
        Array with dimensions `(len(x), Ns)`.

        """
        
        # Generate arrays of random values. Keep in dictionary
        kw = {key:np.random.normal(pars[key], errors[key], Ns) \
            for key in errors}

        # Handle non-vectorized case
        try:
            return np.array(list(map(lambda xx: func(xx, **kw), x)))
        except ValueError:
            arr = np.zeros((len(x), Ns))
            for i in range(int(Ns)):
                new_kw = {key:kw[key][i] for key in kw}
                arr[:,i] = list(map(lambda xx: func(xx, **new_kw), x))

            return arr

    def PlotLF(self, z):
        """
        Plot the luminosity function.
        """
        pass
    
    def PlotLD(self, z):
        """
        Plot the luminosity density.
        """
        pass    
        
    def PlotSFRD(self, z):
        """
        Plot the star formation rate density.
        """
        pass        
    
    
