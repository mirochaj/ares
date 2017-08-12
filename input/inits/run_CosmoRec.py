"""

run_CosmoRec.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Mar  6 14:11:53 MST 2015

Description: Supply path to CosmoRec executable via command-line.

"""

import numpy as np
import ares, sys, os, re

try:
    to_CR = sys.argv[1]
except IndexError:
    print "Supply path to CosmoRec executable via command-line!"
    sys.exit(1)

pf = ares.util.SetDefaultParameterValues.CosmologyParameters()

CR_pars = \
 [
  1000,
  3000,
  0,           # final redshift
  0.24,        # helium
  2.725,
  0.26,
  0.044,
  0.0,         # O_l
  0.0,         # O_k
  0.71,        
  3.04,
  1.14,
  3,
  500,
  0,
  3,
  2,
  1,
  0,
  1,
  2,
  3,
  2,
  './outputs/',
  '.dat'
 ]
 
mapping = \
{
"omega_m_0": 5,
"omega_b_0": 6,
"hubble_0": 9,
"helium_by_mass": 3,
"cmb_temp_0": 4,
}
 
for par in pf:
    if par not in mapping:
        continue
        
    i = mapping[par]
    val = pf[par]
    
    # Update
    CR_pars[i] = val
    
# Create parameter file for reference
f = open('CosmoRec.parameters.dat', 'w')
for element in CR_pars:
    print >> f, element
f.close()

if not os.path.exists('outputs'):
    os.mkdir('outputs')

# Run the thing
os.system('%s CosmoRec.parameters.dat' % to_CR)
    
for fn in os.listdir('outputs'):
    if re.search('final', fn):
        break
    
# Convert it to ares format   
data = np.loadtxt('outputs/%s' % fn) 

new_data = \
 {'z': data[:,0][-1::-1], 
  'xe': data[:,1][-1::-1], 
  'Tk': data[:,2][-1::-1],
 }

np.savez('initial_conditions.npz', **new_data)
print "Wrote initial_conditions.npz."
