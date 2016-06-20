import time
import numpy as np
from ares.inference.Priors import GaussianPrior, UniformPrior, BetaPrior,\
    ParallelepipedPrior
from ares.inference.PriorSet import PriorSet
import matplotlib.pyplot as pl
import matplotlib.cm as cm

sample_size = int(5e4)
t00 = time.time()

# Switches to turn on and off particular tests.
log_transform_test = True
square_transform_test = True
arcsin_transform_test = True
two_param_indep_test = True
three_param_dep_test = True
seven_param_test = True


################### 1-parameter PriorSets (transform tests) ###################

if log_transform_test:
    ps1 = PriorSet()
    ps1.add_prior(GaussianPrior(5., 1.), 'x', 'log')
    t0 = time.time()
    sample = [ps1.draw()['x'] for i in range(sample_size)]
    print ('It took %.3f s to draw %i ' % (time.time()-t0,sample_size,)) +\
        'points from a 1 parameter lognormal distribution.'
    pl.figure()
    pl.hist(sample, bins=np.arange(0., 1501., 15.), histtype='step',\
        color='b', linewidth=2, label='sampled', normed=True)
    xs = np.arange(0.1, 1500., 0.1)
    pl.plot(xs, map(lambda x : np.exp(ps1.log_prior({'x': x})), xs),\
        linewidth=2, color='r', label='e^(log_prior)')
    pl.title('Normal distribution in log space', size='xx-large')
    pl.xlabel('Value', size='xx-large')
    pl.ylabel('PDF', size='xx-large')
    pl.legend(fontsize='xx-large', loc='upper right')
    pl.tick_params(labelsize='xx-large', width=2, length=6)

if square_transform_test:
    ps2 = PriorSet()
    ps2.add_prior(UniformPrior(1., 90.), 'x', 'square')
    t0 = time.time()
    sample = [ps2.draw()['x'] for i in range(sample_size)]
    print ('It took %.3f s to draw %i ' % (time.time()-t0,sample_size,)) +\
        'points from a 1D uniform distribution (in square space).'
    pl.figure()
    pl.hist(sample, bins=100, histtype='step', color='b',\
        linewidth=2, label='sampled', normed=True)
    xs = np.arange(0, 10, 0.001)
    pl.plot(xs, map(lambda x : np.exp(ps2.log_prior({'x': x})), xs),\
        linewidth=2, color='r', label='e^(log_prior)')
    pl.legend(fontsize='xx-large', loc='upper left')
    pl.title('Uniform distribution in square space', size='xx-large')
    pl.xlabel('Value', size='xx-large')
    pl.ylabel('PDF', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)

if arcsin_transform_test:
    ps2 = PriorSet()
    ps2.add_prior(UniformPrior(0, np.pi / 2.), 'x', 'arcsin')
    t0 = time.time()
    sample = [ps2.draw()['x'] for i in range(sample_size)]
    print ('It took %.3f s to draw %i ' % (time.time()-t0,sample_size,)) +\
        'points from a 1D uniform distribution (in arcsin space).'
    pl.figure()
    pl.hist(sample, bins=100, histtype='step', color='b',\
        linewidth=2, label='sampled', normed=True)
    xs = np.arange(0, 1.001, 0.001)
    pl.plot(xs, map(lambda x : np.exp(ps2.log_prior({'x': x})), xs),\
        linewidth=2, color='r', label='e^(log_prior)')
    pl.legend(fontsize='xx-large', loc='upper left')
    pl.title('Uniform distribution in arcsin space', size='xx-large')
    pl.xlabel('Value', size='xx-large')
    pl.ylabel('PDF', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)

##################### 2-parameter PriorSet (independent) ######################
###############################################################################

if two_param_indep_test:
    ps3 = PriorSet()
    ps3.add_prior(UniformPrior(-3., 7.), 'x')
    ps3.add_prior(GaussianPrior(5., 9.), 'y')
    t0 = time.time()
    sample = [ps3.draw() for i in range(sample_size)]
    print ('It took %.3f s to draw %i' % (time.time()-t0,sample_size,)) +\
          ' points from a 2 parameter pdf made up of a ' +\
          'uniform distribution times a Gaussian.'
    xs = [sample[i]['x'] for i in range(sample_size)]
    ys = [sample[i]['y'] for i in range(sample_size)]
    pl.figure()
    pl.hist2d(xs, ys, bins=100, cmap=cm.hot)
    pl.title("PriorSet 2 independent parameter test. x is Unif(-3, 7) and" +\
        " y is Gaussian(5., 9.)", size='xx-large')
    pl.xlabel('x', size='xx-large')
    pl.ylabel('y', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)


###################### 3-parameter PriorSet (dependent) #######################
###############################################################################

if three_param_dep_test:
    ps4 = PriorSet()
    ps4.add_prior(GaussianPrior([10., -5.], [[2., 1.], [1., 2.]]), ['x', 'y'])
    ps4.add_prior(UniformPrior(-3., 17.), 'z')
    t0 = time.time()
    sample = [ps4.draw() for i in range(sample_size)]
    print ('It took %.3f s to draw %i' % (time.time()-t0,sample_size,)) +\
           ' points from a 3 parameter pdf made up of a 2D ' +\
           'gaussian and a 1D uniform distribution'
    xs = [sample[i]['x'] for i in range(sample_size)]
    ys = [sample[i]['y'] for i in range(sample_size)]
    zs = [sample[i]['z'] for i in range(sample_size)]
    pl.figure()
    pl.hist2d(xs, ys, bins=100, cmap=cm.hot)
    pl.title('PriorSet 3 parameters with correlation between x and y',\
        size='xx-large')
    pl.xlabel('x', size='xx-large')
    pl.ylabel('y', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)

########################### 7-parameter PriorSet ##############################
###############################################################################

if seven_param_test:
    ps5 = PriorSet()
    ps5.add_prior(GaussianPrior(5., 4.), 'a')
    ps5.add_prior(BetaPrior(18., 3.), 'b')
    ps5.add_prior(\
        GaussianPrior([5., 2., 9.],\
                      [[4., 1., 1.], [1., 4., 1.], [1., 1., 4.]]),\
                      ['c', 'd', 'e'])
    ps5.add_prior(ParallelepipedPrior([69., 156.],\
                                      [[1.,-1.], [1.,1.]], [1., 1.]),\
                                      ['f', 'g'])
    t0 = time.time()
    sample = [ps5.draw() for i in range(sample_size)]
    print ('It took %.3f s to draw %i ' % (time.time()-t0,sample_size,)) +\
        'points from a mixed distribution with 7 ' +\
        'parameters, in groups of 3, 2, 1, and 1.'
    all_vals = {}
    for char in list('abcdefg'):
        all_vals[char] = [sample[i][char] for i in range(sample_size)]
    pl.figure()
    pl.hist2d(all_vals['b'], all_vals['g'], bins=100, cmap=cm.hot)
    pl.title('PriorSet seven-parameter test. b should be in [0,1] and' +\
        ' g should be around 156', size='xx-large')
    pl.xlabel('b', size='xx-large')
    pl.ylabel('g', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)

###############################################################################

print 'The full test took %.3f s.' % (time.time()-t00,)
pl.show()

