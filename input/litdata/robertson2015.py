"""
Robertson, B. E., Ellis, R. S., Furlanetto, S. R., & Dunlop, J. S. 2015, ApJ,
802, L19

Notes
-----

"""

pars_ml = \
{
 'a': 0.01306,
 'b': 3.66,
 'c': 2.28,
 'd': 5.29,
}

pars_err = \
{
 'a': 0.001,
 'b': 0.21,
 'c': 0.14,
 'd': 0.19,
}
    
def _sfrd(z, a=None, b=None, c=None, d=None):
    return a * (1. + z)**b / (1 + ((1 + z) / c)**d)    
    
def sfrd(z):
    return _sfrd(z, **pars_ml)
    
def _sample(z, Ns=1e4):
   
    import numpy as np
    
    # sfrd = (N, z)
    kw = {key:np.random.normal(pars_ml[key], pars_err[key], Ns) \
        for key in pars_ml}
    
    return np.array(map(lambda zz: _sfrd(zz, **kw), z))
    
def _plot(z, Ns=1e4, **kwargs):
    arr = _sample(z, Ns)
    
    import matplotlib.pyplot as pl

    ax = pl.subplot(111)

    for i in range(int(Ns)):
        ax.plot(z, arr[:,i], **kwargs)
        
    return ax 