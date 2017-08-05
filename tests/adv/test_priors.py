import time
import numpy as np
from ares.inference.Priors import GaussianPrior, UniformPrior,\
    ParallelepipedPrior, ExponentialPrior, DoubleSidedExponentialPrior,\
    BetaPrior, GammaPrior, TruncatedGaussianPrior, LinkedPrior,\
    SequentialPrior, GriddedPrior, EllipticalPrior, PoissonPrior
import matplotlib.pyplot as pl
import matplotlib.cm as cm

def_cm = cm.bone

t00 = time.time()
sample_size = int(1e5)

uniform_test = True
poisson_test = True
exponential_test = True
double_sided_exponential_test = True
beta_test = True
gamma_test = True
truncated_gaussian_test = True
elliptical_test = True
univariate_gaussian_test = True
multivariate_gaussian_test = True
parallelepiped_test = True
linked_test = True
sequential_test = True
gridded_test = True


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

if poisson_test:
    pp = PoissonPrior(10.)
    assert pp.numparams == 1
    t0 = time.time()
    pp_sample = [pp.draw() for i in range(sample_size)]
    print ('It took %.5f s to draw %i ' % (time.time() - t0, sample_size)) +\
          'points from a double-sided exponential distribution.'
    pl.figure()
    pl.hist(pp_sample, bins=np.arange(-49.5, 51, 1), histtype='step',\
        color='b', linewidth=2, normed=True, label='sampled')
    (start, end) = (-20, 20)
    xs = np.linspace(start, end, end - start + 1).astype(int)
    pl.plot(xs, map((lambda x : np.exp(pp.log_prior(x))), xs), linewidth=2,\
        color='r', label='e^(log_prior)')
    pl.legend(fontsize='xx-large', loc='upper right')
    pl.title('Poisson prior test', size='xx-large')
    pl.xlabel('Value', size='xx-large')
    pl.ylabel('PDF', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)


############################# ExponentialPrior test ###########################
###############################################################################

if exponential_test:
    ep = ExponentialPrior(0.1, shift=-5.)
    assert ep.numparams == 1
    t0 = time.time()
    expon_sample = [ep.draw() for i in range(sample_size)]
    print ('It took %.5f s to draw %i ' % (time.time() - t0, sample_size)) +\
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

if double_sided_exponential_test:
    dsep = DoubleSidedExponentialPrior(0., 1.)
    assert dsep.numparams == 1
    t0 = time.time()
    dsexpon_sample = [dsep.draw() for i in range(sample_size)]
    print ('It took %.5f s to draw %i ' % (time.time() - t0, sample_size)) +\
          'points from a double-sided exponential distribution.'
    pl.figure()
    pl.hist(dsexpon_sample, bins=100, histtype='step', color='b', linewidth=2,\
        normed=True, label='sampled')
    xs = np.arange(-9., 9., 0.01)
    pl.plot(xs, map((lambda x : np.exp(dsep.log_prior(x))), xs), linewidth=2,\
        color='r', label='e^(log_prior)')
    pl.legend(fontsize='xx-large', loc='upper right')
    pl.title('Double-sided exponential sample test', size='xx-large')
    pl.xlabel('Value', size='xx-large')
    pl.ylabel('PDF', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)


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

############################ EllipticalPrior test #############################
###############################################################################

if elliptical_test:
    ellmean = [4.76, -12.64]
    ellcov = [[1, -0.5], [-0.5, 1]]
    ellp = EllipticalPrior(ellmean, ellcov)
    t0 = time.time()
    ellp_sample = [ellp.draw() for i in range(sample_size)]
    print (('It took %.3f s for a sample ' % (time.time()-t0)) +\
          ('of size %i' % (sample_size,)) +\
          ' to be drawn from a uniform multivariate elliptical prior.')
    ellp_xs = [ellp_sample[idraw][0] for idraw in range(sample_size)]
    ellp_ys = [ellp_sample[idraw][1] for idraw in range(sample_size)]
    pl.figure()
    pl.hist2d(ellp_xs, ellp_ys, bins=50, cmap=def_cm)
    pl.title('Multivariate elliptical prior (2 dimensions) with ' +\
             ('mean=%s and covariance=%s' % (ellmean, ellcov,)),\
             size='xx-large')
    pl.xlabel('x', size='xx-large')
    pl.ylabel('y', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)
    xs = np.arange(2.7, 6.9, 0.05)
    ys = np.arange(-14.7, -10.5, 0.05)
    row_size = len(xs)
    (xs, ys) = np.meshgrid(xs, ys)
    logpriors = np.ndarray(xs.shape)
    for ix in range(row_size):
        for iy in range(row_size):
            logpriors[ix,iy] = ellp.log_prior([xs[ix,iy], ys[ix,iy]])
    pl.figure()
    pl.imshow(np.exp(logpriors), cmap=def_cm, extent=[2.7, 6.85,-14.7,-10.45],\
        origin='lower')
    pl.title('e^(log_prior) for EllipticalPrior',\
        size='xx-large')
    pl.xlabel('x', size='xx-large')
    pl.ylabel('y', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)

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
          ('of size %i to be drawn from a multivariate ' % (sample_size,)) +\
          ('(%i parameters) Gaussian.' % mgp.numparams))
    mgp_xs = [mgp_sample[i][0] for i in range(sample_size)]
    mgp_ys = [mgp_sample[i][1] for i in range(sample_size)]
    pl.figure()
    pl.hist2d(mgp_xs, mgp_ys, bins=100, cmap=def_cm)
    pl.title('Multivariate Gaussian prior (2 dimensions) with ' +\
             ('mean=%s and covariance=%s' % (mmean,mcov,)), size='xx-large')
    pl.xlabel('x', size='xx-large')
    pl.ylabel('y', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)
    xs = np.arange(-50., 40.1, 0.1)
    ys = np.arange(-25., 65.1, 0.1)
    row_size = len(xs)
    (xs, ys) = np.meshgrid(xs, ys)
    logpriors = np.ndarray(xs.shape)
    for ix in range(row_size):
        for iy in range(row_size):
            logpriors[ix,iy] = mgp.log_prior([xs[ix,iy], ys[ix,iy]])
    pl.figure()
    pl.imshow(np.exp(logpriors), cmap=def_cm, extent=[-50.,40.,-25.,65.],\
        origin='lower')
    pl.title('e^(log_prior) for GaussianPrior',\
        size='xx-large')
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
    xs = np.arange(-20.5, -9.4, 0.1)
    ys = np.arange(14.5, 25.6, 0.1)
    (xs, ys) = np.meshgrid(xs, ys)
    (x_size, y_size) = xs.shape
    logpriors = np.ndarray(xs.shape)
    for ix in range(x_size):
        for iy in range(y_size):
            logpriors[ix,iy] = pp.log_prior([xs[ix,iy], ys[ix,iy]])
    pl.figure()
    pl.imshow(np.exp(logpriors), cmap=def_cm, extent=[-25.,-4.9,14.,26.1],\
        origin='lower')
    pl.title('e^(log_prior) for ParallelepipedPrior distribution',\
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


############################## LinkedPrior test ###############################
###############################################################################

if linked_test:
    lp = LinkedPrior(GaussianPrior(0., 1.), 2)
    t0 = time.time()
    sample = [lp.draw() for i in range(sample_size)]
    print ("It took %.3f s to draw %i" % (time.time()-t0,sample_size,)) +\
          " vectors from a LinkedPrior with a Normal(0,1) distribution."
    sam_xs = [sample[i][0] for i in range(sample_size)]
    sam_ys = [sample[i][1] for i in range(sample_size)]
    pl.figure()
    pl.hist2d(sam_xs, sam_ys, bins=100, cmap=def_cm)
    pl.title('Sampled distribution of a LinkedPrior ' +\
             'with a Normal(0,1) distribution', size='xx-large')
    pl.xlabel('x', size='xx-large')
    pl.ylabel('y', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)
    xs = np.arange(-3., 3.03, 0.03)
    ys = np.arange(-3., 3.03, 0.03)
    row_size = len(xs)
    (xs, ys) = np.meshgrid(xs, ys)
    logpriors = np.ndarray(xs.shape)
    for ix in range(row_size):
        for iy in range(row_size):
            logpriors[ix,iy] = lp.log_prior([xs[ix,iy], ys[ix,iy]])
    pl.figure()
    pl.imshow(np.exp(logpriors), cmap=def_cm, extent=[-3.,3.,-3.,3.],\
        origin='lower')
    pl.title('e^(log_prior) for LinkedPrior with a Normal(0,1) distribution',\
        size='xx-large')
    pl.xlabel('x', size='xx-large')
    pl.ylabel('y', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)

###############################################################################
###############################################################################


############################ SequentialPrior test #############################
###############################################################################

if sequential_test:
    sp = SequentialPrior(UniformPrior(0., 1.), 2)
    t0 = time.time()
    sample = [sp.draw() for i in range(sample_size)]
    print ("It took %.3f s to draw %i" % (time.time()-t0,sample_size,)) +\
          " vectors from a SequentialPrior with a Unif(0,1) distribution."
    sam_xs = [sample[i][0] for i in range(sample_size)]
    sam_ys = [sample[i][1] for i in range(sample_size)]
    pl.figure()
    pl.hist2d(sam_xs, sam_ys, bins=100, cmap=def_cm)
    pl.title('Sampled distribution of a LinkedPrior ' +\
             'with a Unif(0,1) distribution', size='xx-large')
    pl.xlabel('x', size='xx-large')
    pl.ylabel('y', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)
    xs = np.arange(0., 1.01, 0.01)
    ys = np.arange(0., 1.01, 0.01)
    row_size = len(xs)
    (xs, ys) = np.meshgrid(xs, ys)
    logpriors = np.ndarray(xs.shape)
    for ix in range(row_size):
        for iy in range(row_size):
            logpriors[ix,iy] = sp.log_prior([xs[ix,iy], ys[ix,iy]])
    pl.figure()
    pl.imshow(np.exp(logpriors), cmap=def_cm, extent=[0.,1.,0.,1.],\
        origin='lower')
    pl.title('e^(log_prior) for SequentialPrior with Unif(0,1) distribution',\
        size='xx-large')
    pl.xlabel('x', size='xx-large')
    pl.ylabel('y', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)
    

###############################################################################
###############################################################################


############################# GriddedPrior test ###############################
###############################################################################

if gridded_test:
    def pdf_func(x,y):
        if (y > (10. - ((x ** 2) / 5.))) and\
           (y < ((4. * x) + 30.)) and\
           (y < (-4. * x) + 30.) and\
           (x >= -10.) and (x <= 10.):
            return np.exp(-((x ** 2) + ((y - 10.) ** 2)) / 200.)
        else:
            return 0.
    
    xs = np.arange(-20., 20.1, 0.1)
    ys = np.arange(-10., 30., 0.1)
    pdf = np.ndarray((len(xs), len(ys)))
    for ix in range(len(xs)):
        for iy in range(len(ys)):
            pdf[ix,iy] = pdf_func(xs[ix], ys[iy])
    gp = GriddedPrior([xs, ys], pdf=pdf)
    t0 = time.time()
    sample = [gp.draw() for i in range(sample_size)]
    print ("It took %.3f s to draw %i " % (time.time()-t0, sample_size,)) +\
           "points from a user-defined distribution with " +\
           ("%i pixels." % (len(xs) * len(ys),))
    sampled_xs = [sample[i][0] for i in range(sample_size)]
    sampled_ys = [sample[i][1] for i in range(sample_size)]
    pl.figure()
    pl.hist2d(sampled_xs, sampled_ys, bins=100, cmap=def_cm)
    pl.title('Points sampled from a user-defined distribution',\
        size='xx-large')
    pl.xlabel('x', size='xx-large')
    pl.ylabel('y', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)

    pdf_from_log_prior = np.ndarray((len(ys), len(xs)))
    Xs, Ys = np.meshgrid(xs, ys)
    for ix in range(len(xs)):
        for iy in range(len(ys)):
            pdf_from_log_prior[iy,ix] =\
                np.exp(gp.log_prior([Xs[iy,ix], Ys[iy,ix]]))
    pl.figure()
    pl.imshow(pdf_from_log_prior / np.max(pdf_from_log_prior), origin='lower',\
        cmap=def_cm, extent=[-20., 20., -10., 30.])
    pl.gca().set_aspect('equal', adjustable='box')
    pl.title('e^(log_prior) for same distribution as previous sample',\
        size='xx-large')
    pl.xlabel('x', size='xx-large')
    pl.ylabel('y', size='xx-large')
    pl.tick_params(labelsize='xx-large', width=2, length=6)

print 'The full test took %.3f s' % (time.time()-t00,)
pl.show()
