"""

Photometry.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Mon 27 Jan 2020 09:59:10 EST

Description: 

"""

def what_filters(z, fset, wave_lo=1300., wave_hi=2600., picky=True):
    """
    Given a redshift and a full filter set, return the filters that probe
    the rest UV continuum only.
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
        
        if picky:
            if not ((flo >= l1) and (fhi <= l2)):
                continue
        else:
            if not ((flo <= l1 <= fhi) or (flo <= l2 <= fhi)):
                continue
        
        out.append(filt)
        
    return out
    
    