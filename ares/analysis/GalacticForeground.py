"""

GalacticForeground.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Dec 15 20:21:50 MST 2014

Description: Most of this was stolen from Geraint Harker!

"""

import os
import numpy as np
import matplotlib.pyplot as pl
from ..util.SetDefaultParameterValues import ForegroundParameters

try:
    import healpy as hp
except ImportError:
    pass

prefix = '%s/input/gsm' % os.getenv('ARES')

def radiometer(band=0.01, exptime=1e3, Tsys=1e2):
    """
    Compute noise using radiometer equation.

    Parameters
    ----------
    band : int, float
        Band in MHz.
    exptime : int, float
        Exposure time in hours.
    Tsys : int, float
        System temperature in Kelvin.
        
    Returns
    -------
    RMS noise expected in Kelvin.
    
    """

    band *= 1e6
    exptime *= 3600.

    return Tsys / np.sqrt(band * exptime)

class GSM:
    def __init__(self, **kwargs):
        self.pf = ForegroundParameters().copy()
        self.pf.update(kwargs)
        
    def plot(self, freq, **kwargs):
        """
        Plot galactic emission at input frequency (in MHz).
        
        References
        ----------
        de Oliveira-Costa et al. (2008)

        """

        m = self.get_map(freq).squeeze()
        
        hp.mollview(m, title=r'GSM @ $\nu=%g$ MHz' % freq, norm='log', **kwargs)

    def logpoly(self, freq, coeff):
        """
        Compute polynomial in log-log space.
    
        Parameters
        ----------
        nu : int, float
            Frequency in MHz.
        coeff : np.ndarray
            Parameters describing the polynomial.
            [normalization, alpha0, alpha1,...]

        Example
        -------

        Returns
        -------
        Power at frequency nu.    

        """

        return np.exp(np.polyval(coeff, np.log(freq / self.freq_pivot)))

    @property
    def freq_pivot(self):
        return self.pf['fg_pivot']

    @property
    def polyorder(self):
        return self.pf['fg_order']
    
    @property
    def polycoeff(self):
        if not hasattr(self, '_polycoeff'):
            self._polycoeff = self.mean_coeff(np.linspace(35.25,119.75,170))

        return self._polycoeff
        
    def foreground_dOC(self, freq):
        """
        Compute galactic foreground spectrum using de Oliviera-Costa GSM.
        """        
        return np.exp(np.polyval(self.polycoeff, np.log(freq / self.freq_pivot)))

    def get_map(self, freq, nside=512, **kwargs):
        """
        
        Returns a numpy array of shape [12*nside^2 len(freq)] containing 
        maps at the frequencies given in freq, in HEALPix 'ring' format 
        (unless otherwise specified)."""
        
        freq = np.atleast_1d(freq)
        nsideread = 512
        npixread = 12 * nsideread**2
        ncomp = 3 # Number of components
        npix = 12 * nside**2 # Number of pixels
        
        mapfile = '%s/%s' % (prefix, 'component_maps_408locked.dat')
        
        x, y, ypp, n = self.load_components(ncomp, **kwargs)
        
        f = self.compute_components(ncomp,freq,x,y,ypp,n)
        assert( f.shape[0] == ncomp+1 and f.shape[1] == freq.size )
        norm = f[-1,:]
        A = np.loadtxt(mapfile)
        assert(A.shape[0] == npixread and A.shape[1] == ncomp)
        maps = np.zeros([npix,freq.size])
        for i in xrange(freq.size):
          tmp = np.dot(A,f[:-1,i])
          if nside != nsideread:
            maps[:,i] = hp.pixelfunc.ud_grade(tmp,nside)
          else:
            maps[:,i] = tmp
          maps[:,i] = maps[:,i]*norm[i]
        return maps

    def load_components(self, ncomp,**kwargs):
        """Load the principal components from a file and spline them for later use."""
        
        compfile = '%s/%s' % (prefix, 'components.dat')
        tmp = np.loadtxt(compfile)
        if tmp.shape[1] != ncomp + 2:
          raise ValueError('No. of components in compfile does not match ncomp.')
        
        n = tmp.shape[0] # No. of spline points
        y = np.zeros([n,ncomp+1])
        ypp = np.zeros([n,ncomp+1])
        
        x = np.log(tmp[:,0])
        y[:,:ncomp] = tmp[:,2:]
        y[:,ncomp] = np.log(tmp[:,1]) # This column gives an overall scaling
        
        yp0 = 1.e30 # Imposes y'' = 0 at starting point
        yp1 = 1.e30 # Imposes y'' = 0 at endpoint
        
        for i in xrange(ncomp+1):
          ypp[:,i] = self.myspline_r8(x,y[:,i],n,yp0,yp1);
        
        return x,y,ypp,n


    def myspline_r8(self, x,y,n,yp1,ypn):
      """Traces its heritage to some Numerical Recipes routine."""
      assert(x.size==y.size)
      assert(x[-1]>=x[0])
      u = np.zeros(n)
      y2 = np.zeros(n)
      
      if yp1 > 9.9e29:
        y2[0] = 0
        u[0] = 0
      else:
        y2[0] = -0.5
        u[0] = (3/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1);
      
      sig = (x[1:-1]-x[:-2])/(x[2:]-x[:-2])
      tmp = (6*((y[2:n]-y[1:n-1])/(x[2:n]-x[1:n-1])-(y[1:n-1]-y[:n-2])/
                (x[1:n-1]-x[:n-2]))/(x[2:n]-x[:n-2]))
      
      for i in xrange(1,n-1):
        p = sig[i-1]*y2[i-1]+2
        y2[i] = (sig[i-1]-1)/p
        u[i] = (tmp[i-1]-sig[i-1]*u[i-1])/p
      
      if ypn > 9.9e29:
        qn = 0
        un = 0
      else:
        qn = 0.5
        un = (3/(x[n-1]-x[n-2]))*(ypn-(y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
    
      y2[n-1] = (un-qn*u[n-2])/(qn*y2[n-2]+1)
    
      for k in xrange(n-2,-1,-1):
        y2[k] = y2[k]*y2[k+1]+u[k]
    
      return y2
    
    
    def compute_components(self, ncomp, nu, x, y, ypp, n):
      """Compute principal components at frequencies nu."""
      a = np.zeros([ncomp+1,nu.size])
      lnnu = np.log(nu)
      for i in xrange(ncomp+1):
        for j in xrange(nu.size):
          a[i,j] = self.mysplint_r8(x,y[:,i],ypp[:,i],n,lnnu[j])
      a[ncomp,:] = np.exp(a[ncomp,:])
      return a
    
    
    def mysplint_r8(self, xa, ya, y2a, n, x):
      """Spline interpolation."""
      assert(xa.size==n and ya.size==n and y2a.size==n)
      ind = np.searchsorted(xa,x)
      if ind == xa.size: # x>xa[-1], so do linear extrapolation
        a = (ya[n-1]-ya[n-2])/(xa[n-1]-xa[n-2])
        y = ya[n-2] + a*(x-xa[n-2])
      elif ind==0: # x<xa[0], so do linear extrapolation
        a = (ya[1]-ya[0])/(xa[1]-xa[0])
        y = ya[0] + a*(x-xa[0])
      else: # Do cubic interpolation
        khi = ind
        klo = ind-1
        h = xa[khi]-xa[klo]
        a = (xa[khi]-x)/h
        b = (x-xa[klo])/h
        y = a*ya[klo]+b*ya[khi]+((a**3-a)*y2a[klo]+(b**3-b)*y2a[khi])*(h**2)/6
      return y
      
    def calc_coeffs(self, freq,  nside=256):
      """Calculate polynomial coefficients."""
      
      gsmallnu = self.get_map(freq,nside)
      
      x = np.log(freq / self.freq_pivot)
      y = np.log(gsmallnu.T)
      
      coeffmaps = np.polyfit(x,y,self.polyorder)
      
      return coeffmaps

    def mean_coeff(self, freq, nside=256):
        coeffmap = self.calc_coeffs(freq, nside)

        meancoeff = []
        for i in range(coeffmap.shape[0]):
            meancoeff.append(np.mean(coeffmap[i]))

        return meancoeff

    #def main(self):
    #  freq_pivot = 80
    #  nside = 256
    #  polyorder = 3
    #  freqs_in = np.linspace(35.25,119.75,170)
    #  fname = r'gsm_coeffs_poly3_fp80_ns256_hr170'
    #  coeffmaps = calc_coeffs(freqs_in,freq_pivot,polyorder,nside)
    #  write_coeffs(fname,coeffmaps,freq_pivot,polyorder,nside)

