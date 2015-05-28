"""
Ueda, Y., Akiyama, M., Ohta, K., & Miyaji, T. 2003, ApJ, 598, 886
"""

_logLa = 44.6
_La = 10**_logLa

def _dpl(L, phi0, L0, g1, g2):
    return (phi0 / L0) / ((L / L0)**g1 + (L / L0)**g2)

def lf(L):
    pass
    
def _evolution_factor(z, e1=4.23, e2=-1.5):
    pass
    
def _zc(L, zcst=1.9, La=_La):
    pass

def Spectrum():
    pass
    
info = \
{
 'Lmin': 10**41.5,
 'Lmax': 10**46.5,
}    

