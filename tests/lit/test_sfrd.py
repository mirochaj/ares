"""

test_sfrd.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May 28 10:03:30 MDT 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

# Create a Robertson2015 module
r15 = ares.util.read_lit('robertson2015')

# Plot the best-fit SFRD model
z = np.arange(0, 8, 0.05)

# Plot 1,000 samples drawn from the Robertson et al. posteriors
pop = ares.analysis.Population(r15)
models = pop.SamplePosterior(z, r15._SFRD, r15.sfrd_pars, r15.sfrd_err)

ax = pl.subplot(111)
for i in range(int(models.shape[1])):
    ax.semilogy(z, models[:,i], color='b', alpha=0.05)

# Plot best-fit
ax.semilogy(z, r15.SFRD(z))    
ax.set_xlabel(r'$z$')
ax.set_ylabel(ares.util.labels['sfrd'])
    
pl.show()
