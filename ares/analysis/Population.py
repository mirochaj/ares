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
from ..physics.Constants import s_per_yr

class Population(object):
    def __init__(self, pop):
        assert pop.is_ham_model, "These routines only apply for HAM models!"
        self.pop = pop
        
    def HMF_vs_LF(self, z, mags=True, ax=None, fig=1, **kwargs):
        """
        Plot the halo mass function vs. the galaxy luminosity function.
        
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
        i_z = np.argmin(np.abs(z - self.pop.ham.halos.z))
        nofm = self.pop.ham.halos.dndm[i_z]
        
        ax.loglog(self.pop.ham.halos.M, nofm)
                
        ##
        # LF first
        ##
        
        if mags:
            Lunits = 'mab'
        else:
            Lunits = 'erg/s/Hz'
            
        
        phi_xax, phi = self.pop.ham.LuminosityFunction(z, Lunits=Lunits)
        
        #ax.semilogy(phi_xax, phi, **kwargs)
        
        
        # Now, need mass-to-light
        Mh, Lh = self.pop.ham.Lh_of_M(z)

        if mags:
            dndm_xax = self.pop.ham.magsys.L_to_mAB(Lh, z=z)[-1::-1]
            phi = phi[-1::-1]
        else:
            dndm_xax = Lh
        
        #t = np.arange(0.01, 10.0, 0.01)
        #s1 = np.exp(t)
        #ax1.plot(t, s1, 'b-')
        #ax1.set_xlabel('time (s)')
        ## Make the y-axis label and tick labels match the line color.
        #ax1.set_ylabel('exp', color='b')
        #for tl in ax1.get_yticklabels():
        #    tl.set_color('b')
        #
        #
        
        
        #s2 = np.sin(2*np.pi*t)
        ax2 = ax.twinx()
        ax2.plot(dndm_xax, nofm, 'r')
        #ax2.set_ylabel('sin', color='r')
        #for tl in ax2.get_yticklabels():
        #    tl.set_color('r')
        #plt.show()    
        
        #if mags:
        #    #ax.set_xscale('linear')
        #    #ax2.set_xlabel(r'$M_{\mathrm{UV}}$')
        #else:
        #    
        #    ax.set_xlabel(r'$L_h \ (\mathrm{erg} \ \mathrm{s} \ \mathrm{Hz}^{-1})$')
        
        ax.set_xscale('log')
        ax.set_xlabel(r'$M_h / M_{\odot}$')
        ax.set_xlim(self.pop.ham.Mmin[i_z], 1e14)
        ax.set_ylim(1e-12, 1e6)
        ax.set_ylabel('Number Density')
        
        pl.draw()        
        
        return ax
        
    def SMHM(self, z, ratio=False, Nz=100, zmax=40):
        """
        Plot the stellar-mass halo-mass relation.
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
            
            print i
            
        return np.array(Mh_all), np.array(Mstar_all)
        
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
            return np.array(map(lambda xx: func(xx, **kw), x))
        except ValueError:
            arr = np.zeros((len(x), Ns))
            for i in range(int(Ns)):
                new_kw = {key:kw[key][i] for key in kw}
                arr[:,i] = map(lambda xx: func(xx, **new_kw), x)

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
    
    