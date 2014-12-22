"""

test_foreground.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Oct 15 16:35:25 MDT 2014

Description: 

"""

import time, ares
import numpy as np
import matplotlib.pyplot as pl
from ares.util.TanhModel import *
from ares.inference import ModelFit, default_blobs

nu = np.arange(40, 201, 0.1)

# Foreground from de Oliviera-Costa
gsm = ares.analysis.GSM()

coeff = np.array([0.01868, 0.03059, -0.1204, -2.445, 7.4257])
Tfg = gsm.logpoly(nu, coeff)

# Thermal noise [mK]
rms_noise = 1.0
Tnoise = np.random.randn(Tfg.size) * rms_noise

# Signal itself
sim = ares.simulations.Global21cm(tanh_model=True, tanh_nu=nu)

T21 = sim.history['dTb'] / 1e3

data = Tfg + T21 + Tnoise / 1e3

# Plot positive as solid, negative as dashed
#pos = sim.history['dTb'] > 0.
#neg = sim.history['dTb'] <= 0.
#pl.semilogy(sim.history['nu'][pos], T21[pos], color='b', ls='-')
#pl.semilogy(sim.history['nu'][neg], np.abs(T21[neg]), color='b', ls='--')    
#
## Plot noise
#pl.semilogy([nu.min(), nu.max()], [rms_noise]*2, color='r', ls=':')
#
## Plot foreground
#pl.semilogy(nu, Tfg, color='k')
    
# These go to every calculation
base_pars = \
{
 'tanh_model': True,
 'tanh_nu': nu,
 'inline_analysis': default_blobs,
}
    
# Initialize fitter
fit = ModelFit(**base_pars)

# Fit a "raw" signal
fit.data = data * 1e3 # convert to mK
fit.set_error(noise=rms_noise)

"""
MUST FIT FOR FOREGROUND PARAMETERS!
"""

axes = ['tanh_J0', 'tanh_Jz0', 'tanh_Jdz',
        'tanh_xz0', 'tanh_xdz', 'tanh_Tz0', 'tanh_Tdz']
axes.extend(['fg_%i' % i for i in range(len(coeff))])

is_log = [True] + [False] * 6 + [False] * len(coeff)

# Set axes of parameter space
fit.set_axes(axes, is_log=is_log)

priors = \
{
 'tanh_J0': ['uniform', -2., 3.],
 'tanh_Jz0': ['uniform', 15., 35.],
 'tanh_Jdz': ['uniform', 0.1, 10],
 'tanh_xz0': ['uniform', 5., 20.],
 'tanh_xdz': ['uniform', 0.1, 10],
 'tanh_Tz0': ['uniform', 5., 20.],
 'tanh_Tdz': ['uniform', 0.1, 10],
 'tau_e': ['gaussian', 0.08, 0.01, 30.],
}

# Set priors on foreground parameters - uniform distribution factor of 4 wide
for i in range(len(coeff)):
    val = coeff[i]
    if val < 0:
        mi, ma = 1.1 * val, 0.9 * val
    else:
        mi, ma = 0.9 * val, 1.1 * val
        
    priors.update({'fg_%i' % i: ['uniform', mi, ma]})

fit.priors = priors

fit.nwalkers = 32

# Run it!
fit.run(prefix='test_foreground', steps=1e3, clobber=True, 
    save_freq=10, fit_turning_points=False)


