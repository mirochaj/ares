"""

degrade_bpass_seds.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Fri 12 Apr 2019 15:51:48 EDT

Description: 

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as pl
from ares.util.Math import smooth

try:
    degrade_to = int(sys.argv[1])
except IndexError:
    degrade_to = 10

for fn in os.listdir('SEDS'):
    
    if fn.split('.')[-1].startswith('deg'):
        continue
        
    if 'readme' in fn:
        continue
    
    full_fn = 'SEDS/{}'.format(fn)
    out_fn = full_fn+'.deg{}'.format(degrade_to)
    
    if os.path.exists(out_fn):
        print("File {} exists! Moving on...".format(out_fn))
        continue
    
    print("Loading {}...".format(full_fn))
    data = np.loadtxt(full_fn)
    wave = data[:,0]
    
    ok = wave % degrade_to == 0
    new_dims = data.shape[0] // degrade_to
    
    if new_dims == ok.sum() - 1:
        new_dims += 1
    
    new_wave = wave[ok==1]
    new_data = np.zeros((new_dims, data.shape[1]))
    new_data[:,0] = new_wave
    
    for i in range(data.shape[1]):
        if i == 0:
            continue
            
        ys = smooth(data[:,i], degrade_to+1)[ok==1]
        
        new_data[:,i] = ys
        
    np.savetxt(out_fn, new_data)
    print("Wrote {}".format(out_fn)) 
    
    del data, wave


