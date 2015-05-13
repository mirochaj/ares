"""

test_rt06_2.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Dec 26 18:37:48 2012

Description: This is Test problem #2 from the Radiative Transfer
Comparison Project (Iliev et al. 2006; RT06).

"""

import ares

sim = ares.simulations.RaySegment(problem_type=12, tables_dlogN=[0.1]*3,
    tables_discrete_gen=True, tables_energy_bins=250, epsilon_dt=0.005,
    initial_timestep=1e-6)

sim.run()

anl = ares.analysis.RaySegment(sim)

ax1 = anl.RadialProfile('Tk', t=[10, 30, 100])

ax2 = anl.RadialProfile('h_1', t=[10, 30, 100], fig=2)
anl.RadialProfile('h_2', t=[10, 30, 100], ax=ax2, ls='--')

ax3 = anl.RadialProfile('he_1', t=[10, 30, 100], fig=3)
anl.RadialProfile('he_2', t=[10, 30, 100], ax=ax3, ls='--')
anl.RadialProfile('he_3', t=[10, 30, 100], ax=ax3, ls=':')


