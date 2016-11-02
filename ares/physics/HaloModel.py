import numpy as np
from scipy.integrate import quad
from .HaloMassFunction import HaloMassFunction

class HaloModel(HaloMassFunction):
    
    @property
    def field(self):
        pass
        
    def PS_OneHalo(self, z, k, profile_FT):

        iz = np.argmin(np.abs(z - self.z))
        logMmin = np.log10(self.VirialMass(1e4, z))
        iM = np.argmin(np.abs(logMmin - self.logM))
        fcoll = float(self.fcoll(z, logMmin))

        prof = np.abs(map(lambda logM: profile_FT(z, k, logM), self.logM))
                
        dndlnm = self.dndm[iz,:] * self.M
                
        integrand = dndlnm * \
            (self.M / fcoll / self.cosm.mean_density0)**2 * \
            prof**2
         
        return np.trapz(integrand[iM:], x=self.lnM[iM:])
        
        
        
        #_integrand = lambda logm: self.dndlog10m(z, logm) * (10.**logm / self.fcoll(z, logMmin) / (self.MF.mean_density0*hubble_0**2))**2 * \
        #                          abs(profile_FT(z, k, logm))**2 / f_duty
        #return quad(_integrand, logMmin, self.logMmax)[0]
    
    def PS_TwoHalo(self, z, k, profile_FT):
        iz = np.argmin(np.abs(z - self.z))
        logMmin = np.log10(self.VirialMass(1e4, z))
        iM = np.argmin(np.abs(logMmin - self.M))
        fcoll = self.fcoll(z, logMmin)

        dndlnm = self.dndm[iz,:] * self.M
        prof = np.abs(map(lambda logM: profile_FT(z, k, logM), self.logM))

        integrand = dndlnm * \
            (self.M / fcoll / self.cosm.mean_density0) * \
            prof * self.bias(z, self.logM)
            
        return np.trapz(integrand[iM:], x=self.lnM[iM:])**2 * self.psCDM(z, k)

    def PowerSpectrum(self, z, k, profile_FT):
        return self.PS_OneHalo(z, k, profile_FT) #\
             #+ self.PS_TwoHalo(z, k, profile_FT)
    


        
        