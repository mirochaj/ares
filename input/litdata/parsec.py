import os
import numpy as np

_input = os.getenv('ARES') + '/input/parsec/'

cols = ['Zini', 'Age', 'Mini', 'Mass', 'logL', 'logTe', 'logg', 'label', 
    'McoreTP', 'C_O', 'period0', 'period1', 'pmode', 'Mloss', 'tau1m', 'X',
    'Y', 'Xc', 'Xn', 'Xo', 'Cexcess', 'Z', 'mbolmag', 'Umag', 'Bmag', 'Vmag',
    'Rmag', 'Imag', 'Jmag', 'Hmag', 'Kmag']

def _load(fn, Marr):
    """
    Load file `fn` and assume mass grid `Marr`.
    """
    
    raw = np.loadtxt(_input+fn, unpack=True)
    
    t = np.unique(raw[1])
    M_ZAMS = raw[2,raw[1]==t.min()]
    
    Nt = t.size
    Nm = M_ZAMS.size
    
    # Ugh...file doesn't have placeholders for stars that no longer exist...
    # Need to reshape data!
    
    icol = [1, 2, 3, 4, 5, 7]
    keys = ['Age', 'Mini', 'Mass', 'logL', 'logTe', 'label']
    
    shape = (Marr.size, Nt)
    
    # Might be easier to just make up a 2-D interpolant in mass and age...
    
    f = open(_input+fn, 'r')
    i_m = None
    i_t = -1
    lnum = 0
    all_data = {key: -np.inf * np.ones(shape) for key in keys}
    while True:
        
        line = f.readline()
        
        if not line.strip():
            break
        
        is_hdr = line.startswith('# Zini')
        
        if (line[0] == '#') and (not is_hdr):
            continue
        
        # Reached a new timestep    
        if is_hdr:
            if i_m is not None:
                for key in keys:
                    all_data[key][:,i_t] = np.interp(Marr, data['Mini'], data[key],
                        left=-np.inf, right=-np.inf)
            
            i_m = 0
            i_t += 1
            data = {key:[] for key in keys}
            continue
            
        ele = line.split()
            
        # Store info at this mass and this time.
        for j, key in enumerate(keys):
            data[key].append(float(ele[icol[j]]))
                    
        i_m += 1
            
    f.close()
    
    # Convert to Myr
    all_data['Age'] /= 1e6
            
    return all_data
    