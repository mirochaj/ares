"""

ObservedLF.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jan 28 12:38:11 PST 2016

Description: 

"""

import numpy as np
from ..util import read_lit
import matplotlib.pyplot as pl

datasets = ('bouwens2015', 'atek2015', 'oesch2013')

class ObservedLF(object):
    def __init__(self):
        pass
        
    def compile_data(self, z, sources='all'):
        """
        Create a master dictionary containing the MUV points, phi points,
        and (possibly asymmetric) errorbars for all (or some) data available.
        
        Parameters
        ----------
        z : int, float
            Redshift, dummy!
            
        """
        
        data = {}
        
        if sources == 'all':
            sources = datasets
            
        for source in datasets:
            src = read_lit(source)
            
            if z not in src.redshifts:
                print "No z=%g data in %s" % (z, source)
                continue
            
            data[source] = {}
            
            uplims = np.array(src.data['lf'][z]['err']) < 0    
            
            data[source]['M'] = src.data['lf'][z]['M']
            
            if src.units['phi'] == 'log10':
                data[source]['phi'] = 10**src.data['lf'][z]['phi']
                
                logphi_lo_tmp = logphi_ML - err2   # log10 phi
                logphi_hi_tmp = logphi_ML + err2   # log10 phi

                phi_lo = 10**logphi_lo_tmp
                phi_hi = 10**logphi_hi_tmp


                err11 = 10**logphi_ML - phi_lo
                err12 = phi_hi - 10**logphi_ML
            
            else:
                
                data[source]['M'] = src.data['lf'][z]['M']
                data[source]['phi'] = src.data['lf'][z]['phi']
                data[source]['err'] = src.data['lf'][z]['err']
        
        
            
        
    def Plot(self, source='all'):
        pass