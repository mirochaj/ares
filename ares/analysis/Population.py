"""

Population.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri May 29 18:30:49 MDT 2015

Description: 

"""

import numpy as np
import matplotlib.pyplot as pl

class Population(object):
    def __init__(self, pop):
        pass
            
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
    
    