"""

generate_optical_depth_tables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jun 14 09:20:22 2013

Description: Generate optical depth lookup table.

Note: This can be run in parallel, e.g.,

    mpirun -np 4 python generate_optical_depth_tables.py

"""

import sys
import ares

# Initialize radiation background
def_kwargs = \
{
 'tau_Emin': 2e2,
 'tau_Emax': 3e4,
 'tau_Emin_pin': True,
 'tau_fmt': 'hdf5',
 'tau_redshift_bins': 400,
 'approx_He': 1,
 'include_He': 1,
 'initial_redshift': 60,
 'final_redshift': 5,
 'first_light_redshift': 60,
}

kwargs = def_kwargs.copy()
kwargs.update(ares.util.get_cmd_line_kwargs(sys.argv))

# Create OpticalDepth instance
igm = ares.solvers.OpticalDepth(**kwargs)
    
# Impose an ionization history: neutral for all times
igm.ionization_history = lambda z: 0.0

# Tabulate tau and save
tau = igm.TabulateOpticalDepth()
igm.save(suffix=kwargs['tau_fmt'], clobber=False)

    
