import os
import numpy as np

info = \
{
 'reference': 'Ferland 1980',
 'data': 'Table 1'
}

def _load():
    ARES = os.environ.get('ARES')
    E, T10, T20 = np.loadtxt('{}/input/litdata/ferland1980.txt'.format(ARES),
        delimiter=',')
        
    return E, T10, T20
    
    