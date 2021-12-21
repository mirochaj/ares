"""

Photometry.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Mon 27 Jan 2020 09:59:10 EST

Description:

"""

def get_filters_from_waves(z, fset, wave_lo=1300., wave_hi=2600., picky=True):
    """
    Given a redshift and a full filter set, return the filters that probe
    a given wavelength range (rest-UV continuum by default).

    Parameters
    ----------
    z : int, float
        Redshift of interest.
    fset : dict
        A filter set, i.e., the kind of dictionary created by
        ares.obs.Survey.get_throughputs.
    wave_lo, wave_hi: int, float
        Rest wavelengths bounding range of interest [Angstrom].
    picky : bool
        If True, will only return filters that lie entirely in the specified
        wavelength range. If False, returned list of filters will contain those
        that straddle the boundaries (if such cases exist).

    Returns
    -------
    List of filters that cover the specified rest-wavelength range.

    """

    # Compute observed wavelengths in microns
    l1 = wave_lo * (1. + z) * 1e-4
    l2 = wave_hi * (1. + z) * 1e-4

    out = []
    for filt in fset.keys():
        # Hack out numbers
        _x, _y, mid, dx, Tbar = fset[filt]

        fhi = mid + dx[0]
        flo = mid - dx[1]

        entirely_in = (flo >= l1) and (fhi <= l2)
        partially_in = (flo <= l1 <= fhi) or (flo <= l2 <= fhi)

        if picky and (partially_in and not partially_in):
            continue
        elif picky and entirely_in:
            pass
        elif not (partially_in or entirely_in):
            continue
        elif (partially_in or entirely_in):
            pass

        out.append(filt)

    return out
