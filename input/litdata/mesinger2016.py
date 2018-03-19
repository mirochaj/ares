import os
import numpy as np

_input = os.getenv('ARES') + '/input/eos'

def load(model='faint_galaxies'):

    k = []
    z = []
    ps = []
    dTb = []
    for fn in os.listdir('%s/EoS_%s' % (_input, model)):
        if not fn.startswith('ps_no_halos'):
            continue
            
        _z = float(fn[13:19])    
        _dTb = float(fn[fn.index('aveTb')+5:fn.index('aveTb')+11])
        
        z.append(_z)
        dTb.append(_dTb)
    
        x, y, err = np.loadtxt('%s/EoS_%s/%s' % (_input,model,fn), unpack=True)
        
        k.append(x)
        ps.append(y)
        
    z = np.array(z)    
    dTb = np.array(dTb)
    k = np.array(k)
    ps = np.array(ps)
    
    return {'z': z, 'k': k[0], 'ps_21_dl': ps, 'dTb': dTb}

    

