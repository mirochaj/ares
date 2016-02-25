"""

DustCorrection.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue Jan 19 17:55:27 PST 2016

Description: 

"""

import numpy as np
from .ParameterFile import ParameterFile

class DustCorrection(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
        
    @property
    def method(self):
        return self.pf['dustcorr_Afun']
	
	#   ==========   Parametrization of Auv   ==========   #
	
    def AUV(self, z, mag):
		''' Return non-negative mean correction <Auv> averaged over Luv assuming a normally distributed Auv '''
		
		# The scatter in <Auv> can be attributed to 1. scatter in beta(Muv) 2. intrinsic scatter in Auv(beta)
		if self.pf['dustcorr_Afun'] is None:
			return 0.0
		elif self.pf['dustcorr_Afun'].lower() == 'meurer1999':
			sigma = np.sqrt((self.pf['s_beta']*self.MeurerDC(z, mag)[1][1])**2 + self.pf['s_AUV']**2)
			temp = self.MeurerDC(z, mag)[1][0] + self.MeurerDC(z, mag)[1][1] * self.Beta(z, mag) + 0.2*np.log(10)*sigma**2
			return np.maximum(temp, 0.0)
		elif self.pf['dustcorr_Afun'].lower() == 'pettini1998':
			sigma = np.sqrt((self.pf['s_beta']*self.PettiniDC(z, mag)[1][1])**2 + self.pf['s_AUV']**2)
			temp = self.PettiniDC(z, mag)[1][0] + self.PettiniDC(z, mag)[1][1] * self.Beta(z, mag) + 0.2*np.log(10)*sigma**2
			return np.maximum(temp, 0.0)
		elif self.pf['dustcorr_Afun'].lower() == 'capakhighz':
			sigma = np.sqrt((self.pf['s_beta']*self.CapakDC(z, mag)[1][1])**2 + self.pf['s_AUV']**2)
			temp = self.CapakDC(z, mag)[1][0] + self.CapakDC(z, mag)[1][1] * self.Beta(z, mag) + 0.2*np.log(10)*sigma**2
			return np.maximum(temp, 0.0)
		elif self.pf['dustcorr_Afun'].lower() == 'evolving':
			sigma = np.sqrt((self.pf['s_beta']*self.EvolvDC(z, mag)[1][1])**2 + self.pf['s_AUV']**2)
			temp = self.EvolvDC(z, mag)[1][0] + self.EvolvDC(z, mag)[1][1] * self.Beta(z, mag) + 0.2*np.log(10)*sigma**2
			return np.maximum(temp, 0.0)					
		else:
			raise NotImplemented('sorry!')
	
	
    def MeurerDC(self, z, mag):
    	coeff = [4.43, 1.99]    
        val = coeff[0] + coeff[1] * self.Beta(z, mag)
        val = np.maximum(val, 0.0)
        return [val, coeff]
	
	
    def PettiniDC(self, z, mag):
    	coeff = [1.49, 0.68]
    	val =  coeff[0] + coeff[1] * self.Beta(z, mag)
    	val = np.maximum(val, 0.0)
        return [val, coeff]
    
    def CapakDC(self, z, mag):
    	# Fit to Capak upper limits
    	coeff = [0.312, 0.176]
    	val =  coeff[0] + coeff[1] * self.Beta(z, mag)
    	val = np.maximum(val, 0.0)
        return [val, coeff]
        
    def EvolvDC(self, z, mag, z1=5.0, z2=6.0):
    	# Stepwise correction for the given thresholds z1 & z2
    	if z < z1:
    		return self.MeurerDC(z, mag)
    	elif (z >= z1) & (z < z2):
    		return self.PettiniDC(z, mag)
    	else:
    		return self.CapakDC(z, mag) 
    	
    
    #   ==========   Parametrization of Beta   ==========   #
    def Beta(self, z, mag):
        
        if self.pf['dustcorr_Bfun'] == 'constant':
            return self.pf['dustcorr_Bfun_par0']
        elif self.pf['dustcorr_Bfun'] == 'FitMason':
        	return self.BetaFit(z, mag)
        else:
            raise NotImplemented('sorry!')
    
    
    def beta0(self, z):
    	''' Get the measured UV continuum slope from Bouwens+2014 '''
    	_z = np.round(z,0)
    	
    	if _z < 4.0:
        	raise ValueError('z is out of bounds')
    	elif _z == 4.0:
        	val = -1.85; err_rand = 0.01; err_sys = 0.06
    	elif _z == 5.0:
        	val = -1.91; err_rand = 0.02; err_sys = 0.06
    	elif _z == 6.0:
        	val = -2.00; err_rand = 0.05; err_sys = 0.08
    	elif _z == 7.0:
        	val = -2.05; err_rand = 0.09; err_sys = 0.13
    	elif _z == 8.0:
        	val = -2.13; err_rand = 0.44; err_sys = 0.27
    	else:
    		# Assume constant at z>8
        	val = -2.00; err_rand = 0.00; err_sys = 0.00

    	return [val, err_rand, err_sys]
    
    
    def dbeta0_dM0(self, z):
    	''' Get the measured slope of the UV continuum slope from Bouwens+2014 '''
    	_z = np.round(z,0)
    	
    	if _z < 4.0:
        	raise ValueError('z is out of bounds')
    	elif _z == 4.0:
        	val = -0.11; err = 0.01
    	elif _z == 5.0:
        	val = -0.14; err = 0.01
    	elif _z == 6.0:
        	val = -0.20; err = 0.02
    	elif _z == 7.0:
        	val = -0.20; err = 0.04
    	elif _z == 8.0:
        	val = -0.20; err = 0.07
    	else:
    		# Assume constant at z>8
        	val = -0.15; err = 0.00
        
        return [val, err]
    
    
    def BetaFit(self, z, mag):
    	''' An linear + exponential fit to Bouwens+14 data adopted from Mason+2015 '''
    	_M0 = -19.5; _c = -2.33
    	if mag < _M0:
    		return self.dbeta0_dM0(z)[0]*(mag - _M0) + self.beta0(z)[0]
    	else:
    		return (self.beta0(z)[0] - _c) * np.exp(self.dbeta0_dM0(z)[0]*(mag - _M0)/(self.beta0(z)[0] - _c)) + _c
    
        

        
        
        