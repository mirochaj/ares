import time
import numpy as np
from ares.inference.Priors import GaussianPrior, UniformPrior,\
    ParallelepipedPrior, ExponentialPrior, BetaPrior, GammaPrior,\
    TruncatedGaussianPrior
import matplotlib.pyplot as pl
import matplotlib.cm as cm

def_cm = cm.Greys

t00 = time.time()
sample_size = int(1e4)

uniform_test = True
exponential_test = True
beta_test = True
gamma_test = True
truncated_gaussian_test = True
univariate_gaussian_test = True
multivariate_gaussian_test = True
parallelepiped_test = True



############################# UniformPrior test  ##############################
###############################################################################

if uniform_test:
    low = -27.
    high = 19.
    up1 = UniformPrior(high, low)
    up2 = UniformPrior(low, high)
    assert up1.numparams == 1
    assert ((up1.low == up2.low) and (up1.high == up2.high))
    t0 = time.time()
    uniform_sample = [up1.draw() for i in range(sample_size)]
    print ('It took %.5f s to draw %i' % (time.time()-t0, sample_size)) +\
          ' points from a univariate uniform distribution.'
    pl.figure()
    pl.hist(uniform_sample, bins=100, histtype='step', color='b', linewidth=2,\
        normed=True, label='sampled')
    xs = np.arange(-30., 20., 0.01)
    pl.plot(xs, map((lambda x : np.exp(up1.log_prior(x))), xs), linewidth=2,\
        color='r', label='e^(log_prior)')
    pl.title('Uniform prior on [%s,%s]' %\
        (up1.low,up1.high,), size='xx-large')
    pl.xlabel('Value', size='xx-large')
    pl.ylabel('PDF', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)
    pl.legend(fontsize='xx-large', loc='lower center')

###############################################################################
###############################################################################


############################# ExponentialPrior test ###########################
###############################################################################

if exponential_test:
    ep = ExponentialPrior(0.1, shift=-5.)
    assert ep.numparams == 1
    t0 = time.time()
    expon_sample = [ep.draw() for i in range(sample_size)]
    print ('It took %.5f s to draw %i ' % (time.time()-t0, sample_size)) +\
          'points from an exponential distribution.'
    pl.figure()
    pl.hist(expon_sample, bins=100, histtype='step', color='b', linewidth=2,\
        normed=True, label='sampled')
    xs = np.arange(-10., 40., 0.01)
    pl.plot(xs, map((lambda x : np.exp(ep.log_prior(x))), xs), linewidth=2,\
        color='r', label='e^(log_prior)')
    pl.legend(fontsize='xx-large', loc='upper right')
    pl.title('Exponential sample test (mean=5 and shift=-5)', size='xx-large')
    pl.xlabel('Value', size='xx-large')
    pl.ylabel('PDF', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)

###############################################################################
###############################################################################


############################### BetaPrior test ################################
###############################################################################

if beta_test:
    bp = BetaPrior(9, 1)
    assert bp.numparams == 1
    t0 = time.time()
    beta_sample = [bp.draw() for i in range(sample_size)]
    print 'It took %.5f s to draw %i points from a beta distribution.' %\
        (time.time()-t0,sample_size,)
    pl.figure()
    pl.hist(beta_sample, bins=100, linewidth=2, color='b', histtype='step',\
        normed=True, label='sampled')
    xs = np.arange(0.3, 1.0, 0.001)
    pl.plot(xs, map((lambda x : np.exp(bp.log_prior(x))), xs), linewidth=2,\
        color='r', label='e^(log_prior)')
    pl.title('Beta distribution test', size='xx-large')
    pl.xlabel('Value', size='xx-large')
    pl.ylabel('PDF', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)
    pl.legend(fontsize='xx-large', loc='upper left')

###############################################################################
###############################################################################


############################### GammaPrior test ###############################
###############################################################################

if gamma_test:
    gp = GammaPrior(4, 1)
    assert gp.numparams == 1
    t0 = time.time()
    gamma_sample = [gp.draw() for i in range(sample_size)]
    print 'It took %.5f s to draw %i points from a gamma distribution.' %\
        (time.time()-t0,sample_size,)
    pl.figure()
    pl.hist(gamma_sample, bins=100, linewidth=2, color='b', histtype='step',\
        label='sampled', normed=True)
    xs = np.arange(0., 18., 0.001)
    pl.plot(xs, map((lambda x : np.exp(gp.log_prior(x))), xs), linewidth=2,\
        color='r', label='e^(log_prior)')
    pl.title('Gamma distribution test', size='xx-large')
    pl.xlabel('Value', size='xx-large')
    pl.ylabel('PDF', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)
    pl.legend(fontsize='xx-large')

###############################################################################
###############################################################################


########################  TruncatedGaussianPrior tests ########################
###############################################################################

if truncated_gaussian_test:
    tgp = TruncatedGaussianPrior(0., 1., -2., 1.)
    assert tgp.numparams == 1
    t0 = time.time()
    truncated_gaussian_sample = [tgp.draw() for i in range(sample_size)]
    print ('It took %.5f s to draw %i ' % (time.time()-t0,sample_size,)) +\
           'points from a truncated Gaussian distribution.'
    pl.figure()
    pl.hist(truncated_gaussian_sample, bins=100, linewidth=2, color='b',\
        histtype='step', label='sampled', normed=True)
    xs = np.arange(-2.5, 2.501, 0.001)
    pl.plot(xs, map((lambda x : np.exp(tgp.log_prior(x))), xs), linewidth=2,\
        color='r', label='e^(log_prior)')
    pl.title('Truncated Gaussian distribution test', size='xx-large')
    pl.xlabel('Value', size='xx-large')
    pl.ylabel('PDF', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)
    pl.legend(fontsize='xx-large')

############################# GaussianPrior tests #############################
###############################################################################

if univariate_gaussian_test:
    umean = 12.5
    uvar = 2.5
    ugp = GaussianPrior(umean, uvar)
    assert ugp.numparams == 1
    t0 = time.time()
    ugp_sample = [ugp.draw() for i in range(sample_size)]
    print (('It took %.3f s for a sample ' % (time.time()-t0)) +\
          ('of size %i' % (sample_size,)) +\
          ' to be drawn from a univariate Gaussian.')
    pl.figure()
    pl.hist(ugp_sample, bins=100, histtype='step', color='b', linewidth=2,\
        label='sampled', normed=True)
    xs = np.arange(5., 20., 0.01)
    pl.plot(xs, map((lambda x : np.exp(ugp.log_prior(x))), xs), linewidth=2,\
        color='r', label='e^(log_prior)')
    pl.title('Univariate Gaussian prior with mean=%s and variance=%s' %\
        (umean,uvar,), size='xx-large')
    pl.xlabel('Value', size='xx-large')
    pl.ylabel('PDF', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)
    pl.legend(fontsize='xx-large')

if multivariate_gaussian_test:
    mmean = [-7., 20.]
    mcov = [[125., 75.], [75., 125.]]
    mgp = GaussianPrior(mmean, mcov)
    assert mgp.numparams == 2
    t0 = time.time()
    mgp_sample = [mgp.draw() for i in range(sample_size)]
    print (('It took %.3f s for a sample ' % (time.time()-t0)) +\
          ('of size %i to be drawn from a multivariate' % (sample_size,)) +\
          (' (%i parameters) Gaussian.' % mgp.numparams))
    mgp_xs = [mgp_sample[i][0] for i in range(sample_size)]
    mgp_ys = [mgp_sample[i][1] for i in range(sample_size)]
    pl.figure()
    pl.hist2d(mgp_xs, mgp_ys, bins=100, cmap=def_cm)
    pl.title('Multivariate Gaussian prior (2 dimensions) with ' +\
             ('mean=%s and covariance=%s' % (mmean,mcov,)), size='xx-large')
    pl.xlabel('x', size='xx-large')
    pl.ylabel('y', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)

###############################################################################
###############################################################################


########################### ParallelepipedPrior test ##########################
###############################################################################

if parallelepiped_test:
    center = [-15., 20.]
    face_directions = [[1., 1.], [1., -1.]]
    distances = [10., 1.]
    pp = ParallelepipedPrior(center, face_directions, distances,\
        norm_dirs=False)
    assert pp.numparams == 2
    t0 = time.time()
    sample = [pp.draw() for i in range(sample_size)]
    print (('It took %.5f s to draw %i ' % (time.time()-t0,sample_size,)) +\
          'points from a bivariate parallelogram shaped uniform distribution.')
    xs = [sample[i][0] for i in range(sample_size)]
    ys = [sample[i][1] for i in range(sample_size)]
    pl.figure()
    pl.hist2d(xs, ys, bins=100, cmap=def_cm)
    pl.title('Parallelogram shaped uniform dist. centered at %s.' % center,\
        size='xx-large')
    pl.xlabel('x', size='xx-large')
    pl.ylabel('y', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)
    for point in [(-20.2, 16.3), (-20.1, 14.5), (-20., 14.6), (-19., 14.6),\
                  (-11., 25.4), (-10., 25.4), (-9.6, 25.), (-9.6, 24.)]:
        assert pp.log_prior(point) == -np.inf
    for point in [(-19.5, 14.6), (-20.4, 15.5), (-9.6, 24.5), (-10.5, 25.4)]:
        assert pp.log_prior(point) == pp.log_prior(center)

###############################################################################
###############################################################################

print 'The full test took %.3f s' % (time.time()-t00,)
pl.show()
