"""

ModelFit.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Apr 28 11:19:03 MDT 2014

Description: For analysis of MCMC fitting.

"""

import numpy as np
import re, pickle, os
import matplotlib as mpl
from ..util import labels
import matplotlib.pyplot as pl
from ..physics import Cosmology
from .MultiPlot import MultiPanel
from ..inference import ModelGrid
from matplotlib.patches import Rectangle
from ..physics.Constants import nu_0_mhz
from .Global21cm import Global21cm as aG21
from ..util.ParameterFile import count_populations
from ..simulations.Global21cm import Global21cm as sG21
from ..util.Stats import Gauss1D, GaussND, error_1D, rebin
from ..util.ReadData import read_pickled_dataset, read_pickled_dict
from ..util.SetDefaultParameterValues import SetAllDefaults, TanhParameters

tanh_pars = TanhParameters()

try:
    import emcee
    have_emcee = True
except ImportError:
    have_emcee = False

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

def_inset_pars = \
{
 'size':None,
 'margin':(0,0), 
 'align':(0,1),
}

def_kwargs = \
{
 'labels': True,
}

suffixes = ['chain', 'pinfo', 'logL', 'blobs', 'binfo']

def parse_blobs(name):
    nsplit = name.split('_')
    
    if len(nsplit) == 2:
        pre, post = nsplit
    elif len(nsplit) == 3:
        pre, mid, post = nsplit
    
        pre = pre + mid
    
    if pre in labels:
        pass
        
    return None 

def subscriptify_str(s):

    raise NotImplementedError('fix me')
    
    if re.search("_", par):
        m = re.search(r"\{([0-9])\}", par)

    # If it already has a subscript, add to it

def logify_str(s, sub=None):
    s_no_dollar = str(s.replace('$', ''))
    
    if sub is not None:
        new_s = subscriptify_str(s_no_dollar)
    else:
        new_s = s_no_dollar
        
    return r'$\mathrm{log}_{10}' + new_s + '$'
        
def err_str(label, mu, err, log):
    l = str(label.replace('$', ''))

    if log:
        s = '\mathrm{log}_{10}' + l
    else:
        s = l

    s += '=%.3g^{+%.2g}_{-%.2g}' % (mu, err[0], err[1])
    
    return r'$%s$' % s

def def_par_names(N):
    return [i for i in np.arange(N)]

def def_par_labels(i):
    return 'parameter # %i' % i

class DummySampler:
    def __init__(self):
        pass
        
class ModelSubSet(object):
    def __init__(self):
        pass

class ModelSet(object):
    def __init__(self, data):
        """
        Parameters
        ----------
        data : instance, str
            Either an emcee.EnsembleSampler instance or the prefix for
            a bunch of files ending in .chain.pkl, .pinfo.pkl, etc.

        """

        # Read in data from emcee.EmsembleSampler object

        if have_emcee:
            if isinstance(data, emcee.EnsembleSampler):
                self.chain = data.flatchain
                self.Nd = int(self.chain.shape[-1])
                self.parameters = def_par_names(self.Nd)
                self.is_log = [False] * self.Nd
                self.blobs = None
                self.blob_names = []

        # Read in data from file (assumed to be pickled)
        if type(data) == str:
            prefix = data

            # Read MCMC chain
            self.chain = read_pickled_dataset('%s.chain.pkl' % prefix)

            # Figure out if this is an MCMC run or a model grid
            try:
                self.logL = read_pickled_dataset('%s.logL.pkl' % prefix)
                self.is_mcmc = True
            except IOError:
                self.is_mcmc = False
                f = open('%s.grid.pkl' % prefix, 'rb')
                self.axes = pickle.load(f)
                f.close()

            self.Nd = int(self.chain.shape[-1])

            # Read parameter names and info
            if os.path.exists('%s.pinfo.pkl' % prefix):
                f = open('%s.pinfo.pkl' % prefix, 'rb')
                self.parameters, self.is_log = pickle.load(f)
                f.close()

                if rank == 0:
                    print "Loaded %s.pinfo.pkl." % prefix
            else:
                self.parameters = range(self.Nd)
                self.is_log = [False] * self.Nd
            
            if os.path.exists('%s.blobs.pkl' % prefix):
                try:
                    blobs = read_pickled_dataset('%s.blobs.pkl' % prefix)

                    self.mask = np.zeros_like(blobs)    
                    self.mask[np.isinf(blobs)] = 1
                    self.mask[np.isnan(blobs)] = 1
                    self.blobs = np.ma.masked_array(blobs, mask=self.mask)
                except:
                    if rank == 0:
                        print "WARNING: Error loading blobs."    
                    
                f = open('%s.binfo.pkl' % prefix, 'rb')
                self.blob_names, self.blob_redshifts = \
                    map(list, pickle.load(f))
                f.close()

                if rank == 0:
                    print "Loaded %s.binfo.pkl." % prefix

            else:
                self.blobs = self.blob_names = self.blob_redshifts = None

            if os.path.exists('%s.fail.pkl' % prefix):
                self.fails = read_pickled_dict('%s.fail.pkl' % prefix)
                
                if rank == 0:
                    print "Loaded %s.fail.pkl." % prefix
            else:
                self.fails = None
                
            if os.path.exists('%s.setup.pkl' % prefix):
                f = open('%s.setup.pkl' % prefix, 'rb')
                self.base_kwargs = pickle.load(f)
                f.close()
                
                if rank == 0:
                    print "Loaded %s.setup.pkl." % prefix
            else:
                self.base_kwargs = None    
                
            if not self.is_mcmc:
                f = open('%s.grid.pkl' % prefix, 'rb')
                self.axes = pickle.load(f)
                f.close()
                
                if rank == 0:
                    print "Loaded %s.grid.pkl." % prefix
                    
                self.grid = ModelGrid(**self.base_kwargs)
                self.grid.set_axes(**self.axes)
                
                # Only exists for parallel runs
                if os.path.exists('%s.load.pkl' % prefix):
                    f = open('%s.load.pkl' % prefix, 'rb')
                    self.load = pickle.load(f)
                    f.close()
                
                if rank == 0:
                    print "Loaded %s.load.pkl." % prefix
                
        elif isinstance(data, ModelSubSet):
            self.chain = data.chain
            self.is_log = data.is_log
            self.base_kwargs = data.base_kwargs
            self.fails = data.fails
            
            self.mask = np.zeros_like(data.blobs)    
            self.mask[np.isinf(data.blobs)] = 1
            self.mask[np.isnan(data.blobs)] = 1
            self.blobs = np.ma.masked_array(data.blobs, mask=self.mask)

            self.blob_names = data.blob_names
            self.blob_redshifts = data.blob_redshifts
            self.parameters = data.parameters
            self.is_mcmc = data.is_mcmc
            
            if self.is_mcmc:
                self.logL = data.logL
            else:
                try:
                    self.load = data.load
                except AttributeError:
                    pass
                    
                self.axes = data.axes
                self.grid = data.grid
                
            self.Nd = int(self.chain.shape[-1])       
                
        else:
            raise TypeError('Argument must be emcee.EnsembleSampler instance or filename prefix')              
    
    def SelectModels(self):
        """
        Draw a rectangle on supplied matplotlib.axes.Axes instance, return
        information about those models.
        """
                
        if not hasattr(self, '_ax'):
            raise AttributeError('No axis found.')        
                
        self._op = self._ax.figure.canvas.mpl_connect('button_press_event', 
            self._on_press)
        self._or = self._ax.figure.canvas.mpl_connect('button_release_event', 
            self._on_release)
                
    def _on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata
        
    def _on_release(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata
        
        self._ax.figure.canvas.mpl_disconnect(self._op)
        self._ax.figure.canvas.mpl_disconnect(self._or)
        
        # Width and height of rectangle
        dx = abs(self.x1 - self.x0)
        dy = abs(self.y1 - self.y0)
        
        # Find lower left corner of rectangle
        lx = self.x0 if self.x0 < self.x1 else self.x1
        ly = self.y0 if self.y0 < self.y1 else self.y1
        
        # Lower-left
        ll = (lx, ly)
        
        # Upper right
        ur = (lx + dx, ly + dy)
    
        origin = (self.x0, self.y0)
        rect = Rectangle(ll, dx, dy, fc='none', ec='k')
        
        self._ax.add_patch(rect)
        self._ax.figure.canvas.draw()
        
        # Figure out what these values translate to.
        
        # Get basic info about plot
        x = self.plot_info['x']
        y = self.plot_info['y']
        z = self.plot_info['z']
        take_log = self.plot_info['log']
        multiplier = self.plot_info['multiplier']
        
        # Index corresponding to this redshift
        iz = self.blob_redshifts.index(z)
        
        # Container for results
        chain = []
        blobs = []
        logL = []
        
        # Loop over all models and isolate those selected
        for i, pars in enumerate(self.chain):
                
            xy_link = []
            for j, element in enumerate([x, y]):
                
                if element in self.parameters:
                    k = self.parameters.index(element)
                    val = pars[k]
                elif element in self.blob_names:
                    k = self.blob_names.index(element)
                    val = self.blobs[i,iz,k]
                else:
                    k = self.derived_blob_names.index(element)
                    val = self.derived_blobs[i,iz,k]
            
                # Undo any log-10 or multiplication operations
                val /= multiplier[j]
                
                if take_log[j]:
                    val = np.log10(val)

                if (j == 0) and not (ll[0] <= val <= ur[0]):
                    break
                
                if (j == 1) and not (ll[1] <= val <= ur[1]):
                    break
                
                xy_link.append(val)    
            
            if len(xy_link) != 2:
                continue
            
            chain.append(pars)
            blobs.append(self.blobs[i])
            
            if self.is_mcmc:
                logL.append(self.logL[i])
                            
        model_set = ModelSubSet()
        model_set.chain = np.array(chain)
        model_set.base_kwargs = self.base_kwargs.copy()
        model_set.fails = []
        model_set.blobs = np.array(blobs)
        model_set.blob_names = self.blob_names
        model_set.blob_redshifts = self.blob_redshifts
        model_set.is_log = self.is_log
        model_set.parameters = self.parameters
        
        model_set.is_mcmc = self.is_mcmc
        
        if self.is_mcmc:
            model_set.logL = logL
        else:
            model_set.load = self.load
            model_set.grid = self.grid
            model_set.axes = self.axes
        
        self.model_set = model_set
        
        print "Selected %i models. Saved to model_set attribute." % len(chain)
        print "Supply model_set attribute as argument to new ModelSet instance."
        
    def Dump(self, prefix, modelset):
        pass
        
    def ReRunModels(self, models, **kwargs):
        """
        Take list of dictionaries and re-run each as a Global21cm model.
        """
        
        ax = None
        anl_inst = []
        for model in models:
            p = self.base_kwargs.copy()
            
            if type(model) is dict:
                p.update(model)
            else:
                new = self.link_to_dict(model)
                p.update(new)    
            
            try:
                sim = sG21(**p)
                sim.run()
            except SystemExit:
                pass
            
            anl = aG21(sim)
            ax = anl.GlobalSignature(ax=ax, **kwargs)
            
            anl_inst.append(anl)
            
        return ax, anl_inst
        
    @property
    def plot_info(self):
        if not hasattr(self, '_plot_info'):
            self._plot_info = None
        
        return self._plot_info
        
    @plot_info.setter
    def plot_info(self, value):
        self._plot_info = value

    def sort_by_Tmin(self):
        """
        If doing a multi-pop fit, re-assign population ID numbers in 
        order of increasing Tmin.
        
        Doesn't return anything. Replaces attribute 'chain' with new array.
        """

        # Determine number of populations
        tmp_pf = {key : None for key in self.parameters}
        Npops = count_populations(**tmp_pf)

        if Npops == 1:
            return
        
        # Check to see if Tmin is common among all populations or not    
    
    
        # Determine which indices correspond to Tmin, and population #
    
        i_Tmin = []
        
        # Determine which indices 
        pops = [[] for i in range(Npops)]
        for i, par in enumerate(self.parameters):

            # which pop?
            m = re.search(r"\{([0-9])\}", par)

            if m is None:
                continue

            num = int(m.group(1))
            prefix = par.strip(m.group(0))
            
            if prefix == 'Tmin':
                i_Tmin.append(i)

        self._unsorted_chain = self.chain.copy()

        # Otherwise, proceed to re-sort data
        tmp_chain = np.zeros_like(self.chain)
        for i in range(self.chain.shape[0]):

            # Pull out values of Tmin
            Tmin = [self.chain[i,j] for j in i_Tmin]
            
            # If ordering is OK, move on to next link in the chain
            if np.all(np.diff(Tmin) > 0):
                tmp_chain[i,:] = self.chain[i,:].copy()
                continue

            # Otherwise, we need to fix some stuff

            # Determine proper ordering of Tmin indices
            i_Tasc = np.argsort(Tmin)
            
            # Loop over populations, and correct parameter values
            tmp_pars = np.zeros(len(self.parameters))
            for k, par in enumerate(self.parameters):
                
                # which pop?
                m = re.search(r"\{([0-9])\}", par)

                if m is None:
                    tmp_pars.append()
                    continue

                pop_num = int(m.group(1))
                prefix = par.strip(m.group(0))
                
                new_pop_num = i_Tasc[pop_num]
                
                new_loc = self.parameters.index('%s{%i}' % (prefix, new_pop_num))
                
                tmp_pars[new_loc] = self.chain[i,k]

            tmp_chain[i,:] = tmp_pars.copy()
                        
        del self.chain
        self.chain = tmp_chain

    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology()
        
        return self._cosm                        
                            
    @property
    def ref_pars(self):
        if not hasattr(self, '_ref_pars'):
            self._ref_pars = SetAllDefaults()
        
        return self._ref_pars
        
    @property
    def derived_blob_names(self):
        if not hasattr(self, '_derived_blob_names'):
            # Things we know how to calculate
            self._derived_blob_names = ['nu']

            tanh_fit = False
            for par in self.parameters:
                if par in tanh_pars:
                    tanh_fit = True

            if not tanh_fit:
                for sp in ['h_1', 'he_1', 'he_2']:
                    self._derived_blob_names.append('igm_gamma_%s' % sp)

            if not tanh_fit:    
                self._derived_blob_names.append('igm_heat')

            if 'Ts' in self.blob_names:
                self._derived_blob_names.append('contrast')
            
            if ('igm_h_1' in self.blob_names) and \
                'igm_h_2' not in self.blob_names:
                self._derived_blob_names.append('igm_h_2')
            if ('igm_he_1' and 'igm_he_2' in self.blob_names) and \
                'igm_he_3' not in self.blob_names:
                self._derived_blob_names.append('igm_he_3')    
                
            if 'tau_e' in self.blob_names:
                self._derived_blob_names.append('tau_tot')    
                
        return self._derived_blob_names

    @property
    def derived_blobs(self):
        """
        Total rates, convert to rate coefficients.
        """

        if hasattr(self, '_derived_blobs'):
            return self._derived_blobs

        try:
            gamma = self._compute_gamma_tot()        
        except:
            pass
        try:
            heat = self.heat = self._compute_heat_tot()
        except:
            pass

        shape = list(self.blobs.shape[:-1])
        shape.append(len(self.derived_blob_names))

        self._derived_blobs = np.ones(shape) * np.inf
        for k, key in enumerate(self.derived_blob_names):

            try:

                if re.search('igm_gamma', key):
                    self._derived_blobs[:,:,k] = gamma[key]
                elif re.search('igm_heat', key):
                    self._derived_blobs[:,:,k] = heat
                elif key == 'tau_tot':
                    fill = self.blobs[:,-1,self.blob_names.index('tau_e')]
                    for i in range(self._derived_blobs.shape[1]):
                        self._derived_blobs[:,i,k] = fill
                            
                elif key == 'igm_he_3':
                    i_he_1 = self.blob_names.index('igm_he_1')
                    i_he_2 = self.blob_names.index('igm_he_2')
                    self._derived_blobs[:,:,k] = \
                        1. - self.blobs[:,:,i_he_1] - self.blobs[:,:,i_he_2]
                elif key == 'igm_h_2':
                    i_h_1 = self.blob_names.index('igm_h_1')
                    self._derived_blobs[:,:,k] = \
                        1. - self.blobs[:,:,i_h_1]
                elif key == 'nu':
                    i_z = self.blob_names.index('z')
                    self._derived_blobs[:,:,k] = \
                        nu_0_mhz / (1. + self.blobs[:,:,i_z])
                elif key == 'contrast':
                    i_Ts = self.blob_names.index('Ts')
                    for j, redshift in enumerate(self.blob_redshifts):
                        if type(redshift) is str:
                            zindex = self.blob_names.index('z')
                            ztmp = self.blobs[:,j,zindex]
                        else:
                            ztmp = redshift
                        self._derived_blobs[:,j,k] = 1. - \
                            self.cosm.TCMB(ztmp) / self.blobs[:,j,i_Ts]
                            
                else:
                    raise ValueError('dont know derived blob %s!' % key)    
                
            except UnboundLocalError:
                pass
                
        mask = np.zeros_like(self._derived_blobs)    
        mask[np.isinf(self._derived_blobs)] = 1
        
        self.dmask = mask
        
        self._derived_blobs = np.ma.masked_array(self._derived_blobs, 
            mask=mask)

        return self._derived_blobs
        
    def _compute_heat_tot(self):
        """
        Convert heating rate coefficients to a total heating rate using
        number densities of relevant ion species
        """
        
        # (Nsamples, Nredshifts)
        heat = np.zeros(self.blobs.shape[:-1])
        
        for i, sp in enumerate(['h_1', 'he_1', 'he_2']):  
            
            if i > 0 and 'igm_%s' % sp not in self.blob_names:
                print "Total heating rate may be underestimated: no igm_%s in blobs" % sp
                continue
            
            # For slicing
            i_x = self.blob_names.index('igm_%s' % sp)
            i_heat = self.blob_names.index('igm_heat_%s' % sp)

            if sp == 'h_1':
                n = lambda z: self.cosm.nH(z)
            else:
                n = lambda z: self.cosm.nHe(z)

            for j, redshift in enumerate(self.blob_redshifts):
                if type(redshift) is str:
                    zdim = self.blob_names.index('z')
                    ztmp = self.blobs[:,j,zdim]
                else:
                    ztmp = redshift

                # Multiply by number density of absorbers
                heat[:,j] = self.blobs[:,j,i_heat] \
                    * n(ztmp) * self.blobs[:,j,i_x]
                    
        return heat

    def _compute_gamma_tot(self):
        # Total rate coefficient

        # Each has shape (Nsamples, Nredshifts)
        gamma = {'igm_gamma_%s' % sp: np.zeros(self.blobs.shape[:-1]) \
            for sp in ['h_1', 'he_1', 'he_2']}
                   
        for i, sp1 in enumerate(['h_1', 'he_1', 'he_2']):  
            

            x_subject = 'igm_%s' % sp1
            mm = self.blob_names.index(x_subject)    
            
            if sp1 == 'h_1':
                n1 = lambda z: self.cosm.nH(z)
            else:
                n1 = lambda z: self.cosm.nHe(z)    
                  
            for sp2 in ['h_1', 'he_1', 'he_2']:
            
                if sp2 == 'h_1':
                    n2 = lambda z: self.cosm.nH(z)
                else:
                    n2 = lambda z: self.cosm.nHe(z)
                
                blob = 'igm_gamma_%s_%s' % (sp1, sp2)
                k = self.blob_names.index(blob)
                
                x_donor = 'igm_%s' % sp2
                h = self.blob_names.index(x_donor)
                
                for j, redshift in enumerate(self.blob_redshifts):
                    
                    if type(redshift) is str:
                        zindex = self.blob_names.index('z')
                        ztmp = self.blobs[:,j,zindex]
                    else:
                        ztmp = redshift
                                        
                    gamma['igm_gamma_%s' % sp1][:,j] += self.blobs[:,j,k] \
                        * self.blobs[:,j,h] * n2(ztmp) / n1(ztmp) \
                        / self.blobs[:,j,mm]
                        
        return gamma
    
    def set_logL(self, **constraints):
        """
        For ModelGrid calculations, the likelihood must be supplied 
        after-the-fact.
        
        Parameters
        ----------
        constraints : dict
            Constraints to use in calculating logL
            
        Example
        -------
        data = {'z': ['D', mean, 1-sigma error]}
        self.set_logL(data)
            
        Returns
        -------
        Sets "logL" attribute, which is used by several routines.    
            
        """    
            
        if not hasattr(self, 'logL'):
            self.logL = np.zeros(self.chain.shape[0])

        for i in range(self.chain.shape[0]):
            logL = 0.0
            
            if i >= self.blobs.shape[0]:
                break
            
            for element in constraints:
                z, mu, sigma = constraints[element]
                
                j = self.blob_redshifts.index(z)
                k = self.blob_names.index(element)
                
                data = self.blobs[i,j,k]
                
                logL += (data - mu)**2 / 2. / sigma**2
                
            self.logL[i] -= logL
            
        mask = np.isnan(self.logL)
        self.logL[mask] = -np.inf 
            
    def Scatter(self, x, y, z=None, c=None, ax=None, fig=1, slc=None, 
        take_log=False, multiplier=1., **kwargs):
        """
        Show occurrences of turning points B, C, and D for all models in
        (z, dTb) space, with points color-coded to likelihood.
    
        Parameters
        ----------
        x : str
            Field for the x-axis.
        y : str
            Field for the y-axis.
        z : str, float
            Redshift at which to plot x vs. y, if applicable.
        c : str
            Field for (optional) color axis.

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot instance.

        """

        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True

        if type(take_log) == bool:
            take_log = [take_log] * 2
        if type(multiplier) in [int, float]:
            multiplier = [multiplier] * 2

        if z is not None:
            j = self.blob_redshifts.index(z)

        if x in self.parameters:
            xdat = self.chain[:,self.parameters.index(x)]
        else:
            if z is None:
                raise ValueError('Must supply redshift!')
            if x in self.blob_names:
                xdat = self.blobs[:,j,self.blob_names.index(x)]
            else:
                xdat = self.derived_blobs[:,j,self.derived_blob_names.index(x)]
        
        if y in self.parameters:
            ydat = self.chain[:,self.parameters.index(y)]
        else:
            if z is None:
                raise ValueError('Must supply redshift!')
            if y in self.blob_names:
                ydat = self.blobs[:,j,self.blob_names.index(y)]
            else:
                ydat = self.derived_blobs[:,j,self.derived_blob_names.index(y)]
        
        if c in self.parameters:
            cdat = self.chain[:,self.parameters.index(c)]
        elif c == 'load':
            cdat = np.zeros(self.chain.shape[0])
            for i in range(self.chain.shape[0]):
                kw = self.link_to_dict(self.chain[i])
                kvec = self.grid.grid.locate_entry(kw)
                cdat[i] = self.load[kvec]
        elif c is None:
            pass
        else:
            if z is None:
                raise ValueError('Must supply redshift!')
            if c in self.blob_names:
                cdat = self.blobs[:,j,self.blob_names.index(c)]
            else:
                cdat = self.derived_blobs[:,j,self.derived_blob_names.index(c)]
        
        if take_log[0]:
            xdat = np.log10(xdat)
        if take_log[1]:
            ydat = np.log10(ydat)
            
        if hasattr(self, 'weights'):
            ax.scatter(xdat, ydat, c=self.weights, edgecolor='none', **kwargs)
        elif c is not None:
            scat = ax.scatter(xdat, ydat, c=cdat, edgecolor='none', **kwargs)
        else:
            ax.scatter(xdat, ydat, edgecolor='none', **kwargs)
                            
        if take_log[0]:                    
            ax.set_xlabel(logify_str(labels[self.get_par_prefix(x)]))
        else:    
            ax.set_xlabel(labels[self.get_par_prefix(x)])
        
        if take_log[1]: 
            ax.set_ylabel(logify_str(labels[self.get_par_prefix(y)]))
        else:
            ax.set_ylabel(labels[self.get_par_prefix(y)])
                            
        if c is not None:
            cb = pl.colorbar(scat)
            try:
                if take_log[2]: 
                    cb.set_label(logify_str(labels[self.get_par_prefix(c)]))
                else:
                    cb.set_label(labels[self.get_par_prefix(c)])
            except IndexError:
                cb.set_label(labels[self.get_par_prefix(c)])
        
            self._cb = cb
        
        pl.draw()
        
        self._ax = ax
        self.plot_info = {'x': x, 'y': y, 'log': take_log, 
            'multiplier': multiplier, 'z': z}
    
        return ax
    
    def get_par_prefix(self, par):
        m = re.search(r"\{([0-9])\}", par)

        if m is None:
            return par

        # Population ID number
        num = int(m.group(1))

        # Pop ID including curly braces
        prefix = par.strip(m.group(0))
    
        return prefix
    
    @property
    def weights(self):        
        if (not self.is_mcmc) and hasattr(self, 'logL') \
            and (not hasattr(self, '_weights')):
            self._weights = 10**np.array(self.logL) 
            self.new_logL = False

        return self._weights
            
    def clear_logL(self):
        if hasattr(self, 'logL'):
            del self.logL
        
        if hasattr(self, '_weights'):
            del self._weights                
                    
    def get_levels(self, L, nu=[0.95, 0.68]):
        """
        Return levels corresponding to input nu-values, and assign
        colors to each element of the likelihood.
        """
    
        nu, levels = self.confidence_regions(L, nu=nu)
    
        tmp = L.ravel() / L.max()
                                                                      
        return nu, levels
    
    def link_to_dict(self, link):
        pf = {}
        for i, par in enumerate(self.parameters):
            if self.is_log[i]:
                pf[par] = 10**link[i]
            else:
                pf[par] = link[i]
        
        return pf
    
    def get_1d_error(self, par, z=None, bins=20, nu=0.68, take_log=False):
        """
        Compute 1-D error bar for input parameter.
        
        Parameters
        ----------
        par : str
            Name of parameter. 
        bins : int
            Number of bins to use in histogram
        nu : float
            Percent likelihood enclosed by this 1-D error
        
        Returns
        -------
        Tuple, (maximum likelihood value, negative error, positive error).
        
        """
        
        if par in self.parameters:
            j = self.parameters.index(par)
            to_hist = self.chain[:,j]
        elif (par in self.blob_names) or (par in self.derived_blob_names):
            if z is None:
                raise ValueError('Must supply redshift!')
            
            i = self.blob_redshifts.index(z)
            
            if par in self.blob_names:
                j = list(self.blob_names).index(par)
                to_hist = self.blobs[:,i,j].compressed()
            else:
                j = list(self.derived_blob_names).index(par)
                to_hist = self.derived_blobs[:,i,j].compressed()
            if take_log:
                to_hist = np.log10(to_hist)
                        
        hist, bin_edges = \
            np.histogram(to_hist, density=True, bins=bins)

        bc = rebin(bin_edges)
        
        mu, sigma = float(bc[hist == hist.max()]), error_1D(bc, hist, nu=nu)   
        
        return mu, np.array(sigma)
        
    def _get_1d_kwargs(self, **kw):
        
        for key in ['labels', 'colors', 'linestyles']:
        
            if key in kw:
                kw.pop(key)

        return kw
        
    def slice(self, x, y, z):
        """
        Return revised ("sliced") dataset given set of criteria.
        
        This currently only works in planes.
        
        """
        
        # Index corresponding to this redshift
        iz = self.blob_redshifts.index(z)
        
        # Container for results
        chain = []
        blobs = []
        logL = []
        
        # Loop over all models and isolate those selected
        for i, pars in enumerate(self.chain):
                
            xy_link = []
            for j, element in enumerate([x, y]):
                
                if element in self.parameters:
                    k = self.parameters.index(element)
                    val = pars[k]
                elif element in self.blob_names:
                    k = self.blob_names.index(element)
                    val = self.blobs[i,iz,k]
                else:
                    k = self.derived_blob_names.index(element)
                    val = self.derived_blobs[i,iz,k]
            
                # Undo any log-10 or multiplication operations
                val /= multiplier[j]
                
                if take_log[j]:
                    val = np.log10(val)

                if (j == 0) and not (ll[0] <= val <= ur[0]):
                    break
                
                if (j == 1) and not (ll[1] <= val <= ur[1]):
                    break
                
                xy_link.append(val)    
            
            if len(xy_link) != 2:
                continue
            
            chain.append(pars)
            blobs.append(self.blobs[i])
            logL.append(self.logL[i])
                            
        model_set = ModelSubSet()
        model_set.chain = np.array(chain)
        model_set.base_kwargs = self.base_kwargs.copy()
        model_set.fails = []
        model_set.blobs = np.array(blobs)
        model_set.blob_names = self.blob_names
        model_set.blob_redshifts = self.blob_redshifts
        model_set.is_log = self.is_log
        model_set.parameters = self.parameters
        
        model_set.is_mcmc = self.is_mcmc
        
        if self.is_mcmc:
            model_set.logL = logL
        else:
            model_set.axes = self.axes
        
        self.model_set = model_set
        
        print "Selected %i models. Saved to model_set attribute." % len(chain)
        print "Supply model_set attribute as argument to new ModelSet instance."
        
    def ExamineFailures(self, N=1):
        """
        Try to figure out what went wrong with failed models.
        
        Picks a random subset of failed models, plots them, and returns
        the analysis instances associated with each.
        
        Parameters
        ----------
        N : int
            Number of failed models to plot.
            
        """    
        
        kw = self.base_kwargs.copy()
                
        Nf = len(self.fails)
        
        r = np.arange(Nf)
        np.random.shuffle(r)
        
        ax = None
        objects = {}
        for i in range(N):
            
            idnum = r[i]
            
            p = self.base_kwargs.copy()
            p.update(self.fails[idnum])
            
            sim = sG21(**p)
            sim.run()
            
            anl = aG21(sim)
            ax = anl.GlobalSignature(label='fail i=%i' % idnum)
            
            objects[idnum] = anl
            
        return ax, objects
               
    def PosteriorPDF(self, pars, z=None, ax=None, fig=1, multiplier=1.,
        nu=[0.95, 0.68], slc=None, overplot_nu=False, density=True, 
        color_by_nu=False, contour=True, filled=True, take_log=False,
        bins=20, xscale='linear', yscale='linear', skip=0, skim=1, **kwargs):
        """
        Compute posterior PDF for supplied parameters. 
    
        If len(pars) == 2, plot 2-D posterior PDFs. If len(pars) == 1, plot
        1-D marginalized PDF.
    
        Parameters
        ----------
        pars : str, list
            Name of parameter or list of parameters to analyze.
        z : float
            Redshift, if any element of pars is a "blob" quantity.
        plot : bool
            Plot PDF?
        nu : float, list
            If plot == False, return the nu-sigma error-bar.
            If color_by_nu == True, list of confidence contours to plot.
        color_by_nu : bool
            If True, color points based on what confidence contour they lie
            within.
        contour : bool
            Use contours rather than discrete points?
        multiplier : list
            Two-element list of multiplicative factors to apply to elements of
            pars.
        take_log : list
            Two-element list saying whether to histogram the base-10 log of
            each parameter or not.
        skip : int
            Number of steps at beginning of chain to exclude. This is a nice
            way of doing a burn-in after the fact.
        skim : int
            Only take every skim'th step from the chain.

        Returns
        -------
        Either a matplotlib.Axes.axis object or a nu-sigma error-bar, 
        depending on whether we're doing a 2-D posterior PDF (former) or
        1-D marginalized posterior PDF (latter).
    
        """
    
        kw = def_kwargs.copy()
        kw.update(kwargs)
        
        if type(pars) != list:
            pars = [pars]
        if type(take_log) == bool:
            take_log = [take_log] * len(pars)
        if type(multiplier) in [int, float]:
            multiplier = [multiplier] * len(pars)    
    
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
    
        if type(pars) not in [list, tuple]:
            pars = [pars]

        binvec = []
        to_hist = []
        is_log = []
        for k, par in enumerate(pars):

            if par in self.parameters:        
                j = self.parameters.index(par)
                is_log.append(self.is_log[j])
                
                val = self.chain[skip:,j].ravel()[::skim]
                                
                if self.is_log[j]:
                    val += np.log10(multiplier[k])
                else:
                    val *= multiplier[k]
                                
                if take_log[k]:
                    print "WARNING: Maybe don't take log10 of %s." % par
                    to_hist.append(np.log10(val))
                else:
                    to_hist.append(val)
                
            elif (par in self.blob_names) or (par in self.derived_blob_names):
                
                if z is None:
                    raise ValueError('Must supply redshift!')
                    
                i = self.blob_redshifts.index(z)
                
                if par in self.blob_names:
                    j = list(self.blob_names).index(par)
                else:
                    j = list(self.derived_blob_names).index(par)
                
                is_log.append(False)
                
                if par in self.blob_names:
                    val = self.blobs[skip:,i,j][::skim]
                else:
                    val = self.derived_blobs[skip:,i,j][::skim]
                
                if take_log[k]:
                    val += np.log10(multiplier[k])
                else:
                    val *= multiplier[k]
                
                if take_log[k]:
                    to_hist.append(np.log10(val))
                else:
                    to_hist.append(val)

            else:
                raise ValueError('Unrecognized parameter %s' % str(par))

            if self.is_mcmc or (par not in self.parameters):
                if type(bins) == int:
                    valc = to_hist[k]
                    binvec.append(np.linspace(valc.min(), valc.max(), bins))
                elif type(bins[k]) == int:
                    valc = to_hist[k]
                    binvec.append(np.linspace(valc.min(), valc.max(), bins[k]))
                else:
                    binvec.append(bins[k])
            else:
                if take_log[k]:
                    binvec.append(np.log10(self.axes[par]))
                else:
                    binvec.append(self.axes[par])

        if not hasattr(self, 'weights'):
            weights = None
        else:
            weights = self.weights

        if len(pars) == 1:
            
            hist, bin_edges = \
                np.histogram(to_hist[0], density=density, bins=binvec[0], 
                    weights=weights)

            bc = rebin(bin_edges)
        
            tmp = self._get_1d_kwargs(**kw)
            
            ax.plot(bc, hist / hist.max(), drawstyle='steps-mid', **tmp)
            ax.set_xscale(xscale)
            
            if overplot_nu:
                
                try:
                    mu, sigma = bc[hist == hist.max()], error_1D(bc, hist, nu=nu)
                except ValueError:
                    mu, sigma = bc[hist == hist.max()], error_1D(bc, hist, nu=nu[0])
                
                mi, ma = ax.get_ylim()
            
                ax.plot([mu - sigma[0]]*2, [mi, ma], color='k', ls=':')
                ax.plot([mu + sigma[1]]*2, [mi, ma], color='k', ls=':')
            
            ax.set_ylim(0, 1.05)
            
        else:
            
            if to_hist[0].size != to_hist[1].size:
                print 'Looks like calculation was terminated after chain',
                print 'was written to disk, but before blobs. How unlucky!'
                print 'Applying cludge to ensure shape match...'
                
                if to_hist[0].size > to_hist[1].size:
                    to_hist[0] = to_hist[0][0:to_hist[1].size]
                else:
                    to_hist[1] = to_hist[1][0:to_hist[0].size]
                    
            # Compute 2-D histogram
            hist, xedges, yedges = \
                np.histogram2d(to_hist[0], to_hist[1], 
                    bins=[binvec[0], binvec[1]], weights=weights)

            hist = hist.T

            # Recover bin centers
            bc = []
            for i, edges in enumerate([xedges, yedges]):
                bc.append(rebin(edges))
                    
            # Determine mapping between likelihood and confidence contours
            if color_by_nu:
    
                # Get likelihood contours (relative to peak) that enclose
                # nu-% of the area
                nu, levels = self.get_levels(hist, nu=nu)
    
                if filled:
                    ax.contourf(bc[0], bc[1], hist / hist.max(), 
                        levels, **kwargs)
                else:
                    ax.contour(bc[0], bc[1], hist / hist.max(),
                        levels, **kwargs)
                
            else:
                if filled:
                    ax.contourf(bc[0], bc[1], hist / hist.max(), **kw)
                else:
                    ax.contour(bc[0], bc[1], hist / hist.max(), **kw)

            if not gotax:
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
            
            if overplot_nu:
                
                for i in range(2):
                    
                    hist, bin_edges = \
                        np.histogram(to_hist[i], density=density, bins=bins)
                    
                    bc = rebin(bin_edges)
                    
                    mu = bc[hist == hist.max()]
                    
                    try:
                        sigma = error_1D(bc, hist, nu=nu)
                    except ValueError:
                        sigma = error_1D(bc, hist, nu=nu[0])

                    if i == 0:
                        mi, ma = ax.get_ylim()
                    else:
                        mi, ma = ax.get_xlim()

                    if i == 0:
                        ax.plot([mu - sigma[0]]*2, [mi, ma], color='k', ls=':')
                        ax.plot([mu + sigma[1]]*2, [mi, ma], color='k', ls=':')
                    else:
                        ax.plot([mi, ma], [mu - sigma[0]]*2, color='k', ls=':')
                        ax.plot([mi, ma], [mu + sigma[1]]*2, color='k', ls=':')

        if kw['labels']:
            self.set_axis_labels(ax, pars, is_log, take_log)

        pl.draw()

        return ax
        
    def ContourScatter(self, pars, zaxis, z=None, Nscat=1e4, take_log=False, 
        cmap='jet', alpha=1.0, bins=20, vmin=None, vmax=None, zbins=None, 
        **kwargs):
        """
        Show contour plot in 2-D plane, and add colored points for third axis.
        
        Parameters
        ----------
        pars : list
            Plot 2-D posterior PDF for these two parameters
        zaxis : str
            Name of parameter to represent with colored points.
        z : int, float, str
            Redshift (if investigating blobs)
        Nscat : int
            Number of samples plot.
            
        Returns
        -------
        Three objects: the main Axis instance, the scatter plot instance,
        and the colorbar object.
        
        """

        if type(take_log) == bool:
            take_log = [take_log] * 3

        axes = []
        for par in pars:
            if par in self.parameters:
                axes.append(self.chain[:,self.parameters.index(par)])
            elif par in self.blob_names:
                axes.append(self.blobs[:,self.blob_redshifts.index(z),
                    self.blob_names.index(par)])
            elif par in self.derived_blob_names:
                axes.append(self.derived_blobs[:,self.blob_redshifts.index(z),
                    self.derived_blob_names.index(par)])        

        for i in range(2):
            if take_log[i]:
                axes[i] = np.log10(axes[i])

        xax, yax = axes

        if zaxis in self.parameters:        
            zax = self.chain[:,self.parameters.index(zaxis)].ravel()
        elif zaxis in self.blob_names:   
            zax = self.blobs[:,self.blob_redshifts.index(z),
                self.blob_names.index(zaxis)]
        elif zaxis in self.derived_blob_names:   
            zax = self.derived_blobs[:,self.blob_redshifts.index(z),
                self.derived_blob_names.index(zaxis)]
                
        if zax.shape[0] != self.chain.shape[0]:
            if self.chain.shape[0] > zax.shape[0]:
                xax = xax[0:self.blobs.shape[0]]
                yax = yax[0:self.blobs.shape[0]]
                print 'Looks like calculation was terminated after chain',
                print 'was written to disk, but before blobs. How unlucky!'
                print 'Applying cludge to ensure shape match...'
            else:                
                raise ValueError('Shape mismatch between blobs and chain!')    
                
        if take_log[2]:
            zax = np.log10(zax)    
            
        ax = self.PosteriorPDF(pars, z=z, take_log=take_log, filled=False, 
            bins=bins, **kwargs)
        
        # Pick out Nscat random points to plot
        mask = np.zeros_like(xax, dtype=bool)
        rand = np.arange(len(xax))
        np.random.shuffle(rand)
        mask[rand < Nscat] = True
        
        if zbins is not None:
            cmap_obj = eval('mpl.colorbar.cm.%s' % cmap)
            norm = mpl.colors.BoundaryNorm(zbins, cmap_obj.N)
        else:
            norm = None
        
        scat = ax.scatter(xax[mask], yax[mask], c=zax[mask], cmap=cmap,
            zorder=1, edgecolors='none', alpha=alpha, vmin=vmin, vmax=vmax,
            norm=norm)
        cb = pl.colorbar(scat)
        
        cb.set_alpha(1)
        cb.draw_all()

        if zaxis in labels:
            cb.set_label(labels[zaxis])
        elif '{' in zaxis:
            cb.set_label(labels[zaxis[0:zaxis.find('{')]])
        else:
            cb.set_label(zaxis)    
            
        cb.update_ticks()
            
        pl.draw()
        
        return ax, scat, cb
        
    def TrianglePlot(self, pars=None, z=None, panel_size=(0.5,0.5), 
        padding=(0,0), show_errors=False, take_log=False, multiplier=1,
        fig=1, inputs={}, tighten_up=0.0, ticks=5, bins=20, mp=None, skip=0, 
        skim=1, top=None, oned=True, filled=True, box=None, rotate_x=True, 
        **kwargs):
        """
        Make an NxN panel plot showing 1-D and 2-D posterior PDFs.

        Parameters
        ----------
        pars : list
            Parameters to include in triangle plot.
            1-D PDFs along diagonal will follow provided order of parameters
            from left to right. This list can contain the names of parameters,
            so long as the file prefix.pinfo.pkl exists, otherwise it should
            be the indices where the desired parameters live in the second
            dimension of the MCMC chain.

            NOTE: These can alternatively be the names of arbitrary meta-data
            blobs.

            If None, this will plot *all* parameters, so be careful!

        fig : int
            ID number for plot window.
        bins : int,
            Number of bins in each dimension.
        z : int, float, str
            If plotting arbitrary meta-data blobs, must choose a redshift.
            Can be 'B', 'C', or 'D' to extract blobs at 21-cm turning points,
            or simply a number.
        input : dict
            Dictionary of parameter:value pairs representing the input
            values for all model parameters being fit. If supplied, lines
            will be drawn on each panel denoting these values.
        panel_size : list, tuple (2 elements)
            Multiplicative factor in (x, y) to be applied to the default 
            window size as defined in your matplotlibrc file. 
        skip : int
            Number of steps at beginning of chain to exclude. This is a nice
            way of doing a burn-in after the fact.
        skim : int
            Only take every skim'th step from the chain.
        oned : bool    
            Include the 1-D marginalized PDFs?
        filled : bool
            Use filled contours? If False, will use open contours instead.
        color_by_nu : bool
            If True, set contour levels by confidence regions enclosing nu-%
            of the likelihood. Set parameter `nu` to modify these levels.
        nu : list
            List of levels, default is 1,2, and 3 sigma contours (i.e., 
            nu=[0.68, 0.95])
        rotate_x : bool
            Rotate xtick labels 90 degrees.
        
        Returns
        -------
        ares.analysis.MultiPlot.MultiPanel instance.
        
        """    
        
        if pars is None:
            pars = self.parameters
        
        kw = def_kwargs.copy()
        kw.update(kwargs)
        
        if type(take_log) == bool:
            take_log = [take_log] * len(pars)
        if multiplier == 1:
            multiplier = [multiplier] * len(pars)        
        if type(bins) == int:
            bins = [bins] * len(pars)
        
        is_log = []
        for par in pars[-1::-1]:
            if par in self.parameters:
                is_log.append(self.is_log[self.parameters.index(par)])
            elif par in self.blob_names or self.derived_blob_names:
                is_log.append(False)
                
        if oned:
            Nd = len(pars)
        else:
            Nd = len(pars) - 1
                           
        # Multipanel instance
        if mp is None:
            mp = MultiPanel(dims=[Nd]*2, padding=padding, diagonal='lower',
                panel_size=panel_size, fig=fig, top=top, **kw)

        # Loop over parameters
        for i, p1 in enumerate(pars[-1::-1]):
            for j, p2 in enumerate(pars):

                # Row number is i
                # Column number is self.Nd-j-1

                k = mp.axis_number(i, j)

                if k is None:
                    continue
                
                if mp.grid[k] is None:
                    continue
                    
                # Input values (optional)
                if (p1 in self.ref_pars) or (p1 in inputs):
                    if not inputs:
                        val = self.ref_pars[p1]
                    else:
                        val = inputs[p1]
                        
                    if val is None:
                        yin = None
                    elif is_log[i]:
                        yin = np.log10(val)    
                    else:
                        yin = val                        
                else:
                    yin = None
                
                if (p2 in self.ref_pars) or (p2 in inputs):
                    
                    if not inputs:
                        val = self.ref_pars[p2]
                    else:
                        val = inputs[p2]
                        
                    if val is None:
                        xin = None  
                    elif is_log[Nd-j-1]:
                        xin = np.log10(val)
                    else:
                        xin = val

                else:
                    xin = None

                col, row = mp.axis_position(k)    
                                                                                
                # 1-D PDFs on the diagonal    
                if k in mp.diag and oned:

                    self.PosteriorPDF(p1, ax=mp.grid[k], 
                        take_log=take_log[-1::-1][i], z=z,
                        multiplier=[multiplier[-1::-1][i]], 
                        bins=[bins[-1::-1][i]], skip=skip, skim=skim, **kw)

                    if col != 0:
                        mp.grid[k].set_ylabel('')
                    if row != 0:
                        mp.grid[k].set_xlabel('')
                    
                    if show_errors:
                        mu, err = self.get_1d_error(p1)
                                                 
                        mp.grid[k].set_title(err_str(labels[p1], mu, err, 
                            self.is_log[i])) 
                     
                    if not inputs:
                        continue
                        
                    if xin is not None:
                        mp.grid[k].plot([xin]*2, [0, 1.05], 
                            color='k', ls=':', lw=2)    
                            
                    continue

                # If not oned, may end up with some x vs. x plots
                if p1 == p2:
                    continue

                # 2-D PDFs elsewhere
                self.PosteriorPDF([p2, p1], ax=mp.grid[k], z=z,
                    take_log=[take_log[j], take_log[-1::-1][i]],
                    multiplier=[multiplier[j], multiplier[-1::-1][i]], 
                    bins=[bins[j], bins[-1::-1][i]], filled=filled, **kw)
                
                if row != 0:
                    mp.grid[k].set_xlabel('')
                if col != 0:
                    mp.grid[k].set_ylabel('')

                # Input values
                if not inputs:
                    continue
                    
                if xin is not None:
                    mp.grid[k].plot([xin]*2, mp.grid[k].get_ylim(), color='k', 
                        ls=':')
                if yin is not None:
                    mp.grid[k].plot(mp.grid[k].get_xlim(), [yin]*2, color='k', 
                        ls=':')

        if oned:
            mp.grid[np.intersect1d(mp.left, mp.top)[0]].set_yticklabels([])
        
        mp.fix_ticks(oned=oned, N=ticks, rotate_x=rotate_x)
        mp.rescale_axes(tighten_up=tighten_up)
    
        return mp
        
    def RedshiftEvolution(self, blob, ax=None, redshifts=None, fig=1,
        nu=0.68, take_log=False, **kwargs):
        """
        Plot constraints on the redshift evolution of given quantity.
        
        Parameters
        ----------
        blob : str
            
        Note
        ----
        If you get a "ValueError: attempt to get argmin of an empty sequence"
        you might consider setting take_log=True.    
            
        """    
        
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
        
        if redshifts is None:
            redshifts = self.blob_redshifts
            
        for z in redshifts:
            
            # Error on blob
            value, (blob_err1, blob_err2) = \
                self.get_1d_error(blob, z=z, nu=nu, take_log=take_log)
            
            # Error on redshift
            if type(z) == str:
                mu_z, (z_err1, z_err2) = \
                    self.get_1d_error('z', z=z, nu=nu)
                xerr = np.array(z_err1, z_err2).T
            else:
                mu_z = z
                xerr = None
            
            ax.errorbar(mu_z, value, 
                xerr=xerr, 
                yerr=np.array(blob_err1, blob_err2).T, 
                lw=4, elinewidth=4, capsize=6, capthick=3,
                **kwargs)        
        
        # Look for populations
        m = re.search(r"\{([0-9])\}", blob)
        
        if m is None:
            prefix = blob
        else:
            # Population ID number
            num = int(m.group(1))
            
            # Pop ID including curly braces
            prefix = blob.strip(m.group(0))
        
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(labels[prefix])
        pl.draw()
        
        return ax
        
    def add_boxes(self, ax=None, val=None, width=None, **kwargs):
        """
        Add boxes to 2-D PDFs.
        
        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot instance
        
        val : int, float
            Center of box (probably maximum likelihood value)
        width : int, float
            Size of box (above/below maximum likelihood value)
        
        """
        
        if width is None:
            return
        
        iwidth = 1. / width
            
        # Vertical lines
        ax.plot(np.log10([val[0] * iwidth, val[0] * width]), 
            np.log10([val[1] * width, val[1] * width]), **kwargs)
        ax.plot(np.log10([val[0] * iwidth, val[0] * width]), 
            np.log10([val[1] * iwidth, val[1] * iwidth]), **kwargs)
            
        # Horizontal lines
        ax.plot(np.log10([val[0] * iwidth, val[0] * iwidth]), 
            np.log10([val[1] * iwidth, val[1] * width]), **kwargs)
        ax.plot(np.log10([val[0] * width, val[0] * width]), 
            np.log10([val[1] * iwidth, val[1] * width]), **kwargs)

    def extract_blob(self, name, z):
        """
        Extract a 1-D array of values for a given quantity at a given redshift.
        """
    
        i = self.blob_redshifts.index(z)
    
        if name in self.blob_names:
            j = self.blob_names.index(name)
            return self.blobs[:,i,j]
        else:
            j = self.derived_blob_names.index(name)
            return self.derived_blobs[:,i,j]
    
    def max_likelihood_parameters(self):
        """
        Return parameter values at maximum likelihood point.
        """
    
        iML = np.argmax(self.logL)
    
        p = {}
        for i, par in enumerate(self.parameters):
            if self.is_log[i]:
                p[par] = 10**self.chain[iML,i]
            else:
                p[par] = self.chain[iML,i]
    
        return p
        
    def set_axis_labels(self, ax, pars, is_log, take_log=False):
        """
        Make nice axis labels.
        """
        
        p = []
        sup = []
        for par in pars:
            
            if type(par) is int:
                sup.append(None)
                p.append(par)
                continue
                
            m = re.search(r"\{([0-9])\}", par)
            
            if m is None:
                sup.append(None)
                p.append(par)
                continue
            
            p.append(par.strip(m.group(0)))
            sup.append(int(m.group(1)))
        
        del pars
        pars = p
    
        if type(take_log) == bool:
            take_log = [take_log] * len(pars)
    
        if pars[0] in labels:
            if is_log[0] or take_log[0]:
                ax.set_xlabel(logify_str(labels[pars[0]]))
            else:
                ax.set_xlabel(labels[pars[0]])
                    
        elif type(pars[0]) == int:
            ax.set_xlabel(def_par_labels(pars[0]))
        else:
            ax.set_xlabel(pars[0])
    
        if len(pars) == 1:
            ax.set_ylabel('PDF')
    
            pl.draw()
            return
    
        if pars[1] in labels:
            if is_log[1] or take_log[1]:
                ax.set_ylabel(logify_str(labels[pars[1]]))
            else:
                ax.set_ylabel(labels[pars[1]])
        elif type(pars[1]) == int:
            ax.set_ylabel(def_par_labels(pars[1]))
        else:
            ax.set_ylabel(pars[1])
    
        pl.draw()
        
    def confidence_regions(self, L, nu=[0.95, 0.68]):
        """
        Integrate outward at "constant water level" to determine proper
        2-D marginalized confidence regions.
    
        Note: this is fairly crude.
    
        Parameters
        ----------
        L : np.ndarray
            Grid of likelihoods.
        nu : float, list
            Confidence intervals of interest.
    
        Returns
        -------
        List of contour values (relative to maximum likelihood) corresponding 
        to the confidence region bounds specified in the "nu" parameter, 
        in order of decreasing nu.
        """
    
        if type(nu) in [int, float]:
            nu = np.array([nu])
    
        # Put nu-values in ascending order
        if not np.all(np.diff(nu) > 0):
            nu = nu[-1::-1]
    
        peak = float(L.max())
        tot = float(L.sum())
    
        # Counts per bin in descending order
        Ldesc = np.sort(L.ravel())[-1::-1]
    
        j = 0  # corresponds to whatever contour we're on
    
        Lprev = 1.0
        Lencl_prev = 0.0
        contours = [1.0]
        for i in range(1, Ldesc.size):
    
            # How much area (fractional) is enclosed within the current contour?
            Lencl_now = L[L >= Ldesc[i]].sum() / tot
    
            Lnow = Ldesc[i]
    
            # Haven't hit next contour yet
            if Lencl_now < nu[j]:
                pass
    
            # Just passed a contour
            else:
                # Interpolate to find contour more precisely
                Linterp = np.interp(nu[j], [Lencl_prev, Lencl_now],
                    [Ldesc[i-1], Ldesc[i]])
                # Save relative to peak
                contours.append(Linterp / peak)
    
                j += 1
    
            Lprev = Lnow
            Lencl_prev = Lencl_now
    
            if j == len(nu):
                break
    
        # Return values that match up to inputs    
        return nu[-1::-1], contours[-1::-1]
    
    def errors_to_latex(self, pars, nu=0.68, in_units=None, out_units=None):
        """
        Output maximum-likelihood values and nu-sigma errors ~nicely.
        """
                
        if type(nu) != list:
            nu = [nu]
            
        hdr = 'parameter    '
        for conf in nu:
            hdr += '%.1f' % (conf * 100)
            hdr += '%    '
        
        print hdr
        print '-' * len(hdr)    
        
        for i, par in enumerate(pars):
            
            s = str(par)
            
            for j, conf in enumerate(nu):
                
                
                mu, sigma = \
                    map(np.array, self.get_1d_error(par, bins=100, nu=conf))

                if in_units and out_units != None:
                    mu, sigma = self.convert_units(mu, sigma,
                        in_units=in_units, out_units=out_units)

                s += r" & $%5.3g_{-%5.3g}^{+%5.3g}$   " % (mu, sigma[0], sigma[1])
        
            s += '\\\\'
            
            print s
    
    def convert_units(self, mu, sigma, in_units, out_units):
        """
        Convert units on common parameters of interest.
        
        So far, just equipped to handle frequency -> redshift and Kelvin
        to milli-Kelvin conversions. 
        
        Parameters
        ----------
        mu : float
            Maximum likelihood value of some parameter.
        sigma : np.ndarray
            Two-element array containing asymmetric error bar.
        in_units : str
            Units of input mu and sigma values.
        out_units : str
            Desired units for output.
        
        Options
        -------
        in_units and out_units can be one of :
        
            MHz
            redshift
            K
            mK
            
        Returns
        -------
        Tuple, (mu, sigma). Remember that sigma is itself a two-element array.
            
        """
        
        if in_units == 'MHz' and out_units == 'redshift':
            new_mu = nu_0_mhz / mu - 1.
            new_sigma = abs(new_mu - (nu_0_mhz / (mu + sigma[1]) - 1.)), \
                abs(new_mu - (nu_0_mhz / (mu - sigma[0]) - 1.))
                        
        elif in_units == 'redshift' and out_units == 'MHz':
            new_mu = nu_0_mhz / (1. + mu)
            new_sigma = abs(new_mu - (nu_0_mhz / (1. + mu - sigma[0]))), \
                        abs(new_mu - (nu_0_mhz / (1. + mu - sigma[1])))
        elif in_units == 'K' and out_units == 'mK':
            new_mu = mu * 1e3
            new_sigma = np.array(sigma) * 1e3
        elif in_units == 'mK' and out_units == 'K':
            new_mu = mu / 1e3
            new_sigma = np.array(sigma) / 1e3
        else:
            raise ValueError('Unrecognized unit combination')
        
        return new_mu, new_sigma
    