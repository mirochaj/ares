"""

TanhModel.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Sep  9 16:25:41 MDT 2014

Description: 

"""

import time
import numpy as np
from ..physics import Hydrogen
from ..analysis import Global21cm
from ..physics.Constants import k_B, J21_num
from ..physics.RateCoefficients import RateCoefficients
from ..util.SetDefaultParameterValues import TanhParameters

# Default parameters
tanh_kwargs = TanhParameters()

tanh_pars = ['tanh_J0', 'tanh_Jz0', 'tanh_Jdz', 
    'tanh_T0', 'tanh_Tz0', 'tanh_Tdz', 'tanh_x0', 'tanh_xz0', 'tanh_xdz']

# Create instance of Hydrogen class
hydr = Hydrogen()

rc = RateCoefficients(recombination='A')
alpha_A = rc.RadiativeRecombinationRate(0, 1e4)

def tanh_generic(z, zref, dz):
    return 0.5 * (np.tanh((zref - z) / dz) + 1.)
    
def temperature(z, Tref, zref, dz):
    return Tref * tanh_generic(z, zref=zref, dz=dz) + hydr.cosm.Tgas(z)
    
def ionized_fraction(z, xref, zref, dz):
    return xref * tanh_generic(z, zref=zref, dz=dz)

def heating_rate(z, Tref, zref, dz, Lambda=None):

    Tk = temperature(z, Tref, zref, dz)

    dtdz = hydr.cosm.dtdz(z)

    dTkdz = 0.5 * Tref * (1. - np.tanh((zref - z) / dz)**2) / dz
    dTkdt = dTkdz / dtdz

    n = hydr.cosm.nH(z)

    if Lambda is None:
        cool = 0.0
    else:
        cool = Lambda(z)

    return 1.5 * n * k_B * (dTkdt + 2. * hydr.cosm.HubbleParameter(z) * Tk) \
        + cool / dtdz
    
def ionization_rate(z, xref, zref, dz, C=1.):
    xi = ionized_fraction(z, xref, zref, dz)
    
    dtdz = hydr.cosm.dtdz(z)
    
    dxdz = 0.5 * xref * (1. - np.tanh((zref - z) / dz)**2) / dz
    dxdt = dxdz / dtdz
    
    n = hydr.cosm.nH(z)
    
    return dxdt + alpha_A * C * n * xi
    
def tanh_model(z, **kwargs):
    """
    Check "tanh_pars" for list of acceptable parameters.
    
    Returns
    -------
    glorb.analysis.-21cm instance, which contains the entire signal
    and the turning points conveniently in the "turning_points" 
    attribute.
    
    """
    kw = tanh_kwargs.copy()
    
    for element in kwargs:
        kw.update({element: kwargs[element]})
        
    theta = [kw[par] for par in tanh_pars]
    
    return tanh_model_for_emcee(z, theta)
    
def tanh_model_for_emcee(z, theta):
    """
    Compute 21-cm signal given tanh model parameters.
    
    Input parameters assumed to be in the following order:
    
    Jref, zref_J, dz_J, 
    Tref, zref_T, dz_T, 
    xref, xref_J, dz_x
    
    where Jref, Tref, and xref are the step heights, zref_? are the step
    locations, and dz_? are the step-widths.
    
    Note that Jref is in units of J21.
    
    Returns
    -------
    ares.analysis.Global21cm instance, which contains the entire signal
    and the turning points conveniently in the "turning_points" 
    attribute.
    
    """
        
    # Unpack parameters
    Jref, zref_J, dz_J, Tref, zref_T, dz_T, xref, zref_x, dz_x = theta
    
    Jref *= J21_num
    
    try:
        Tgas = hydr.cosm.Tgas(z)
    except ValueError:
        Tgas = np.array(map(hydr.cosm.Tgas, z))

    Ja = Jref * tanh_generic(z, zref=zref_J, dz=dz_J)
    Tk = Tref * tanh_generic(z, zref=zref_T, dz=dz_T) + Tgas
    xi = xref * tanh_generic(z, zref=zref_x, dz=dz_x)
    
    # Compute (proper) electron density assuming xHII = xHeII, xHeIII = 0.
    # Needed for collisional coupling.
    
    # Spin temperature
    Ts = hydr.SpinTemperature(z, Tk, Ja, xi, 0.0)

    # Brightness temperature
    dTb = hydr.DifferentialBrightnessTemperature(z, xi, Ts)

    # Save some stuff
    hist = \
    {
     'z': z, 
     'dTb': dTb,
     'igm_Tk': Tk,
     'Ts': Ts,
     'Ja': Ja,
     'cgm_h_2': xi,
     'igm_heat': np.array(map(lambda z: heating_rate(z, Tref, zref_T, dz_T), z)),
     'cgm_Gamma': np.array(map(lambda z: ionization_rate(z, xref, zref_x, dz_x), z))
    }
    
    # Add heating rates, etc.

    tmp = Global21cm(history=hist)    

    return tmp
