"""

CheckForParameterConflicts.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 21 16:05:07 2011

Description: Check to make sure there are no conflicts between parameters.

"""

import numpy as np

known_conflicts = [ \
    (['infinite_c', 0], ['parallelization', 1])]
    #(['isothermal', 0], ['restricted_timestep', 'energy', 0])]
    
# Treat warnings separately

def CheckForParameterConflicts(pf):
    """
    Loop over parameters and make sure none of them are in conflict.
    """
        
    probs = []
    for conflict in known_conflicts:
        
        ok = []
        for element in conflict:
            if type(pf[element[0]]) is list:
                if int(element[1] in pf[element[0]]) == element[2]:
                    ok.append(False)
                else:
                    ok.append(True)
            else:
            
                if pf[element[0]] == element[1]:
                    ok.append(False)
                else:
                    ok.append(True)
                                
        if not np.any(ok):
            probs.append(conflict)
        
    if probs:
        errmsg = []
        for i, con in enumerate(probs):
            for element in con:
                try:
                    errmsg.append('%s = %g' % (element[0], element[1]))
                except TypeError:
                    errmsg.append('%s = %s' % (element[0], element[1]))
            
            if len(probs) > 1 and i != len(probs):
                errmsg.append('\nAND\n')
        
        conflicts = True
    else:
        conflicts = False
            
    if conflicts:
        print 'ERROR -- PARAMETER VALUES IN CONFLICT:'
        for msg in errmsg:
            print msg
        print '\n'    
    
    return conflicts
            