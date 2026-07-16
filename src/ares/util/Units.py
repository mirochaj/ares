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

def get_wave_or_equivalent(x_in, units, units_out):
    """
    Convert between photon wavelength, energy, and frequency.

    Parameters
    ----------
    x_in : int, float, np.ndarray
        Array of values that we'd like convert to different units.
    units : str
        Units of `x_in`, e.g., 'cm', 'ang', 'mic', 'hz', 'ghz', 'ev', 'keV'.
    units_out : str
        Units we'd like to convert `x_in` to.

    Returns
    -------
    Input array `x_in` converted to output units `units_out`.

    """
    if type(x_in) in [list, tuple]:
        x_in = np.array(x_in)

    if units.lower() == units_out.lower():
        return x_in
    
    if type(x_in) in [tuple, list]:
        x_in = np.array(x_in)
    
    ##
    # Start by convert input unit to cm
    if units.lower() == 'cm':
        x_cm = x_in
    elif units.lower().startswith('ang'):
        x_cm = x_in * 1e-8
    elif (units.lower() == 'um') or units.lower().startswith('mic'):
        x_cm = x_in * 1e-4
    elif units.lower() == 'hz':
        x_cm = c / x_in
    elif units.lower() == 'mhz':
        x_cm = c / (x_in * 1e6)
    elif units.lower() == 'ghz':
        x_cm = c / (x_in * 1e9)
    elif units.lower() == 'ev':
        x_cm = h_p * c / (x_in * erg_per_ev)
    elif units.lower() == 'kev':
        x_cm = h_p * c / (x_in * 1e3 * erg_per_ev)
    else:
        raise NotImplemented(f'Unrecognized input unit={units}')
    
    if units_out.lower() == 'cm':
        return x_cm
    elif units_out.lower().startswith('ang'):
        return x_cm * 1e8
    elif (units.lower() == 'um') or units_out.lower().startswith('mic'):
        return x_cm * 1e4
    elif units_out.lower() == 'hz':
        return c / x_cm
    elif units_out.lower() == 'mhz':
        return c / x_cm / 1e6
    elif units_out.lower() == 'ghz':
        return c / x_cm / 1e9
    elif units_out.lower() == 'ev':
        return h_p * c / x_cm / erg_per_ev
    elif units_out.lower() == 'kev':
        return h_p * c / x_cm / erg_per_ev / 1e3
    else:
        raise NotImplemented(f'Unrecognized input unit={units}')
    
def get_dwave_or_equivalent(x_in, units, units_out):
    # Potentially put per-Hz or per-Ang back in
    if units_out.lower().endswith('/hz'):
        x_out = get_wave_or_equivalent(x_in, units=units, 
            units_out='hz')
    elif units_out.lower().endswith('/ang'):
        x_out = get_wave_or_equivalent(x_in, units=units, 
            units_out='ang')
    else:
        return 1.
    
    return np.abs(np.diff(x_out))

def get_ev_from_x(x, units):
    """
    Convert input `x` from `units` to eV.
    """
    return get_wave_or_equivalent(x, units, 'eV')

def get_ang_from_x(x, units):
    """
    Convert input `x` from `units` to Angstroms.
    """
    return get_wave_or_equivalent(x, units, 'ang')

def get_ev_from_x_OLD(x, units='eV'):
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
        return xout
    else:
        return xout
    
def get_ang_from_x_OLD(x, units='eV'):
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
        return out
    else:
        return out

