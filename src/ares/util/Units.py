"""

Units.py

Author: Jordan Mirocha
Affiliation: Caltech
Created on: Wed Jul  8 12:49:00 2026

Description:

"""

import numbers
import numpy as np
from ..physics.Constants import h_p, c, erg_per_ev
from .Misc import numeric_types

def get_ev_from_x(x, units='eV'):
    """
    Convert input `x` from `units` to electron volts.

    .. note :: Will always return energies in ascending order! This is
        because we're usually doing this to find some bounding range over
        which to integrate.
    .. note :: Currently understands the following units: eV, Angstroms,
        microns, and Hz.

    """

    type_in = type(x)
    if type_in in [list, tuple]:
        x = np.array(x)
    elif type_in in numeric_types:
        x = np.array([x])

    if units.lower() == 'ev':
        xout = x.copy()
    elif units.lower().startswith('ang'):
        xout = h_p * c / erg_per_ev / x / 1e-8
    elif (units.lower() == 'um') or units.lower().startswith('mic'):
        xout = h_p * c / erg_per_ev / x / 1e-4
    elif units.lower().startswith('hz'):
        xout = h_p * x / erg_per_ev
    else:
        raise NotImplemented('help')

    # Re-order if necessary
    if x.size > 1:
        if xout[0] > xout[1]:
            xout = np.flip(xout)
    if type_in == tuple:
        return tuple(xout)
    elif type_in == list:
        return list(xout)
    elif type_in in numeric_types:
        return float(xout)
    else:
        return xout
    
def get_ang_from_x(x, units='eV'):
    """
    Convert input `x` from `units` to Angstroms.
    """

    # If supplied units are already Angstroms, we're done.
    if units.lower().startswith('ang'):
        return x
        
    # This routine always returns in order of ascending photon energy,
    # so it's possible that `x` has been flipped.
    # There's a check below to make sure
    xout = get_ev_from_x(x, units=units)
    type_in = type(x)

    if isinstance(x, numbers.Number):
        x_is_band = False
        out = h_p * c / erg_per_ev / xout / 1e-8
    else:
        x_is_band = True
        out = h_p * c / erg_per_ev / np.array(xout) / 1e-8

    # Check for order change, since get_ev_from_x aways returns in
    # ascending energy. Want to match input order of `x`.
    # In other words, match order of input `x` unless we're converting
    # from wavelength to energy.
    if units.lower() not in ['ev', 'hz'] and x_is_band and (out[0] > out[1]):
        # Maybe this is only microns right now?
        out = np.flip(out)
    if type_in == tuple:
        return tuple(out)
    elif type_in == list:
        return list(out)
    elif type_in in numeric_types:
        return float(out)
    else:
        return out

