"""

test_21cm_tanh.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Sep  9 20:03:58 MDT 2014

Description: 

"""

import ares

sim = ares.simulations.Global21cm(tanh_model=True)
sim.run()

anl = ares.analysis.Global21cm(sim)
ax = anl.GlobalSignature(color='b', label='tanh')

sim2 = ares.simulations.Global21cm()
sim2.run()

anl2 = ares.analysis.Global21cm(sim2)

anl2.GlobalSignature(ax=ax, color='k', label=r'$f_{\mathrm{coll}}$')

ax.legend(loc='lower right', fontsize=14)
