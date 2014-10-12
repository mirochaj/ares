"""

BoundedSpace.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Mar 27 13:53:29 MDT 2014

Description: 

"""

import numpy as np
import os, copy, time, re
import matplotlib.pyplot as pl
from ..simple import Interpret21cm
from rt1d.physics import Hydrogen, Cosmology
from rt1d.physics.Constants import cm_per_mpc, J21_num

try:
    import h5py
except ImportError:
    pass

try:
    from multiplot import multipanel
except ImportError:
    pass

try:
    from mathutils.stats import rebin
except ImportError:
    pass

extras = ['Ja', 'Tk', 'Ts', 'heat_igm', 'Gamma_igm', 'gamma_igm', 
    'Gamma_HII', 'xi', 'xe', 'xavg']
turning_points = ['B', 'C', 'trans', 'D']

class BoundedSpace(object):
    """ Read in data from parameter space exploration."""
    def __init__(self, database=None, path='.'):
        """
        Parameters
        ----------
        database : str
            Filename of HDF5 file created using built-in to_hdf5 method.
        path : str
            Path to output of parameter space explorations.
            
        The main attributes of interest in this class are: 
            self.data : 2-D array
                first dimension: turning point positions and IGM parameters
                 of interest at those points. See self.data_rows for a list of
                 their names.
                second dimension: model number
            self.par_vals : 2-D array
                first dimension: astrophysical parameter values for given
                 model. See self.par_names for list of parameter names.
                second dimension: model number
                
            
        """

        self.path = path
        self.database = database
                
        self._locate()
        
        if self.database is None:
            self._load_rawoutput()
        else:
            self._load_database()
            
        self.hydr = Hydrogen()    
        self.cosm = Cosmology() 
        self.simpl = Interpret21cm()           
                    
    def _locate(self):
        """
        Find all outputs in self.path. 
        
        Save to absolute paths to self.fn, and unique IDs to self.model_id.
        """
        
        self.fn = []
        self.model_id = []
        for fn in os.listdir(self.path):
            
            # Search for processed results only, which are ASCII
            if re.search('.hdf5', fn):
                continue
            if re.search('.pkl', fn):
                continue
                
            # Look for UUID - 32 character string
            UUID = fn[0:fn.find('.')]
                
            if len(UUID) != 32:
                continue
                
            self.fn.append('%s/%s' % (self.path, fn))
            
            self.model_id.append(UUID)
        
        self.Nmods = len(self.fn)
        self.model_num = np.arange(self.Nmods)
        
    @property
    def data_rows(self):
        """
        """
        if not hasattr(self, '_rows'):
        
            self._rows = []
            for tp in turning_points:
                
                self._rows.append('z_%s' % tp)
                self._rows.append('T_%s' % tp)
                                
                for item in extras:
                    self._rows.append('%s_%s' % (item, tp))
        
        return self._rows
        
    @property
    def data_dict(self):
        if not hasattr(self, '_data_dict'):
            self._data_dict = {}
            for i, row in enumerate(self.data_rows):
                self._data_dict[row] = self.data[i,:]

        return self._data_dict
        
    def dictify_data(self, data):
        """
        Convert 2-D array (Nparameters x Nmodels) into a dictionary, with
        each element corresponding to...what again?
        """
        data_dict = {}
        for i, row in enumerate(self.data_rows):
            data_dict[row] = data[i,:]
    
        return data_dict 
                    
    def row(self, parameter):
        """
        Given name of parameter, return corresponding index.
        """
        return self.data_rows.index(parameter)
    
    def col(self, parameter):
        """
        Given name of parameter, return corresponding index.
        """
        return self.data_cols.index(parameter)    
        
    def _load_rawoutput(self):
        """
        Read in data from all files found in self.path.
        
        Need to generalize to handle string parameter values, like
        spectrum_type and fitting_function.
        """ 
        
        Npars = None
        self.par_mismatch = False
        
        # Store parameters
        pars = []
        
        # Store turning points
        data = np.zeros([len(self.data_rows), self.Nmods])
        
        # Loop over files we found
        for i, fn in enumerate(self.fn):
            
            f = open(fn, 'r')
            
            # Read in parameters
            
            tmp_pars = []
            for line in f:
                
                # Once we hit the gap between pars and data
                if not line.strip():
                    break
                
                line_spl = line.split()
                
                if line_spl[1].isalpha():
                    tmp_pars.append([line_spl[0].strip(), line_spl[1].strip()])
                else:    
                    tmp_pars.append([line_spl[0].strip(), float(line_spl[1])])
                
            pars.append(tmp_pars)    
                
            # Read in results

            for j, line in enumerate(f):
                if not line.strip():
                    break
                    
                line_spl = line.split()   

                data[j,i] = float(line_spl[1])

            f.close()    
            
            # Everything the same shape so far?
            if Npars is None:
                Npars = len(tmp_pars)
            else:
                if Npars != len(tmp_pars):
                    self.par_mismatch = True
                    
        self.data = data

        # Re-organize data
        if not self.par_mismatch:
            self.par_names = list(zip(*pars[0])[0])
            self.strpar_names = []

            self.data_cols = copy.deepcopy(self.par_names)
            
            # Must handle parameters whose value is a string separately
            Ns = 0
            Np = len(self.par_names)            
            if 'spectrum_type' in self.par_names:
                Np -= 1
                Ns += 1
                
                self.par_names.remove('spectrum_type')
                self.strpar_names.append('spectrum_type')
            
            if 'fitting_function' in self.par_names:
                Np -= 1
                Ns += 1
            
                self.par_names.remove('fitting_function')
                self.strpar_names.append('fitting_function')    

            self.par_vals = np.zeros([Np, self.Nmods])
            self.strpar_vals = np.zeros([Ns, self.Nmods], dtype='|S3')

            i_str1 = self.data_cols.index('spectrum_type')
            i_str2 = self.data_cols.index('fitting_function')
            
            for i in range(self.Nmods):
                tmp = list(zip(*pars[i])[1])                   
                                                                
                self.strpar_vals[0,i] = tmp.pop(i_str1)
                self.strpar_vals[1,i] = tmp.pop(i_str2)
                self.par_vals[:,i] = tmp
                
        print "Loaded raw output. %i-D parameter space, %i models." \
            % (len(self.data_cols), self.Nmods)
        
    def _load_database(self):
        """
        Loads output created using the to_hdf5 method.
        """
        
        f = h5py.File(self.database)
        
        self.data = f['data'].value
        
        junk, self.Nmods = self.data.shape
        self.par_vals = f['par_vals'].value
        self.par_names = list(f['par_vals'].attrs.get('par_names'))
        self.strpar_vals = f['strpar_vals'].value
        self.strpar_names = list(f['strpar_vals'].attrs.get('strpar_names'))
        
        self.data_cols = list(f['data'].attrs.get('columns'))
        
        try:
            self.model_id = f['model_id'].value
        except:
            pass
        
        f.close()
        
        print "Loaded databases. %i-D parameter space, %i models." \
            % (len(self.data_cols), self.Nmods)
        
    def extract_data(self, pinfo={}, dinfo={}, data=None):
        """
        Extract subset of data based on input conditions.
                
        Parameters
        ----------
        pinfo : dict
            Parameter information (e.g., single out models that have similar
            star formation efficiencies)
            Format: {'parameter1': [min, max], 'parameter2': [min, max]}
        dinfo : dict
            Data information (e.g., single out models based on position of
            turning point)
            Format: {'z_B': [min, max], 'T_B': [min, max]}
            
        OR 
        
            Format: {'Mmin': None}
            
            
        Returns
        -------
        Tuple: data, parameter values, string parameter values, and unique
        identification strings for models where supplied conditions are 
        satisfied.
        
        """
        
        mask = np.ones(self.Nmods, dtype=bool)
        
        for i in range(self.Nmods):
            
            for p in pinfo:
                
                ploc = self.par_names.index(p)
                
                try:
                    pmin, pmax = pinfo[p]
                
                    if p in self.strpar_names:
                        pass
                    else:
                                                
                        if not np.logical_and(self.par_vals[ploc,i] <= pmax, 
                            self.par_vals[ploc,i] >= pmin):
                            mask[i] = False
                            continue
                
                # Can require a parameter to have exactly some value
                except:
                    if p in self.strpar_names:
                        pass
                    else:
                                                
                        if self.par_vals[ploc,i] != pinfo[p]:
                            mask[i] = False
                            continue
            
            for d in dinfo:
                dmin, dmax = dinfo[d]
            
                dloc = self.data_rows.index(d)
            
                if not np.logical_and(self.data[dloc,i] <= dmax, 
                     self.data[dloc,i] >= dmin):
                     mask[i] = False
                     continue   
                     
        # Determine ID hashtags for all extracted models
        if hasattr(self, 'model_id'):
            
            if len(self.model_id) > 0:
                model_ids = []
                for i in range(self.Nmods):
                    if mask[i] == False:
                        continue
                        
                    model_ids.append(self.model_id[i])
            
        else:
            model_ids = None    
            
        return self.data[:,mask], self.par_vals[:,mask], self.strpar_vals, \
            model_ids
        
    def extract_3pt_models(self, lt3=False):
        """
        Return subset of models that exhibit 3 turning points.
        
        Parameters
        ----------
        lt3 : bool
            Set to True if you want to retrieve all odd models, i.e., those
            without 3 turning points.
            
        """
        
        mask = np.ones(self.Nmods, dtype=bool)
        
        for i in range(self.Nmods):
            
            if lt3:
                if not np.any(self.data[:,i] == -99999):
                    mask[i] = False
            else:
                if np.any(self.data[:,i] == -99999):
                    mask[i] = False
            
  
        return self.data[:,mask], self.par_vals[:,mask], self.strpar_vals, \
            mask
        
    def read_histories(self, model_ids, suffix='raw.hdf5', fields=['z', 'dTb']):
        """
        Given a list of UUIDs, read in given fields.
        
        Currently only setup to read additional data if saved in HDF5.
        """    
        
        models = {}
        
        for field in fields:
            models[field] = []
        
        for ID in model_ids:
            f = h5py.File('%s.%s' % (ID, suffix))
            for key in models:
                models[key].append(f[key].value)
                
            f.close()    
        
        return models    
            
    def read_cxrb(self, model_ids, suffix='cxrb.hdf5'):
        """
        Given a list of UUIDs, read in given fields.
        
        Currently only setup to read additional data if saved in HDF5.
        """
        
        models = {}
        
        for field in ['z', 'E', 'flux']:
            models[field] = []
        
        for ID in model_ids:
            f = h5py.File('%s.%s' % (ID, suffix))
            for key in models:
                models[key].append(f[key].value)
                
            f.close()    
        
        return models
        
    def to_hdf5(self, fn=None, data=None):
        """
        Output data to HDF5 for more convenient analysis later.
        """
        
        if fn is None:
            fn = '%s.database.hdf5' % (time.ctime().replace(' ', '_'))
        
        if data is None:
            data = self.data
            
        if os.path.exists(fn):
            raise IOError('%s exists!' % fn)    
        
        f = h5py.File(fn, 'w')
        
        ds1 = f.create_dataset('data', data=data)
        ds1.attrs.create('columns', self.data_cols)
        
        ds2 = f.create_dataset('par_vals', data=self.par_vals)
        ds2.attrs.create('par_names', self.par_names)
        
        ds3 = f.create_dataset('strpar_vals', data=self.strpar_vals)
        ds3.attrs.create('strpar_names', self.strpar_names)
        
        f.create_dataset('model_id', data=self.model_id)
        
        f.close()
        
        print "Wrote %s." % fn
    
    def analyzeC(self, require_3pt=True, bin_dex=0.15, redshifts=[15, 20, 25],
        plot_z_boxes=False):
        """
        Make standard plots for given turning point.
        """    
    
        if require_3pt:
            data = self.dictify_data(self.extract_3pt_models()[0])
        else:
            data = self.data    
            
        # Analytic limits
        Tk = [self.cosm.Tgas(10), self.cosm.TCMB(40)]
        cheat = np.array(map(self.simpl.heating_rate_C, [10, 40]))
                    
        heat = data['heat_igm_C'] \
            / ((1. - data['xe_C']) / self.cosm.nH(data['z_C']))    
    
        heat *= cm_per_mpc**3 / (1. + data['z_C'])**3
    
        h, x, y = np.histogram2d(data['Tk_C'], heat, 
            bins=(np.arange(0, Tk[1] * 1.1, 1), 
                  10**np.arange(np.log10(np.min(cheat) * 0.5), 
                  np.log10(np.max(cheat[1]) * 3), 
                  bin_dex)))    
                  
        # Start figure
        fig = pl.figure(1); ax = fig.add_subplot(111)
                  
        # Filled contour plot
        ax.contourf(rebin(x), rebin(y), h.T, 
            levels=np.logspace(0, np.log10(h.max()), 20))
        ax.set_xlabel(r'$T_K (z_{\mathrm{C}}) \ \left[\mathrm{K} \right]$')
        ax.set_ylabel(r'$\epsilon_X (z_{\mathrm{C}}) \ \left[\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cMpc}^{-3} \right]$')
        ax.set_xscale('linear')
        ax.set_yscale('log')
        
        # Overplot analytic limits
        ls = ['-', '--']
        for i, col in enumerate(['k', 'w']):
            ax.plot(Tk, [np.min(cheat)] *2, color=col, ls=ls[i])
            ax.plot(Tk, [np.max(cheat)] *2, color=col, ls=ls[i])
            ax.plot([Tk[0]] * 2, [np.min(cheat), np.max(cheat)], color=col, ls=ls[i])
            ax.plot([Tk[1]] * 2, [np.min(cheat), np.max(cheat)], color=col, ls=ls[i])
                
        h, x, y = np.histogram2d(data['z_C'], data['T_C'],
            bins=(np.arange(10, 40, 0.1), np.arange(-200, 0, 5)))

        #fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)
        #ax2.contourf(rebin(x), rebin(y), h.T)
        #ax2.set_xlabel(r'$z_{\mathrm{C}}$')
        #ax2.set_ylabel(r'$\delta T_b(z_{\mathrm{C}})$')
        #ax2.set_xscale('linear')
        #ax2.set_yscale('linear')
        #ax2.set_xlim(10, 40)
        #ax2.set_ylim(-350, 0)
                
        # Lines of constant redshift
        if not plot_z_boxes:
            return
            
        ls = ['--', ':', '-.']
        for i, z in enumerate(redshifts):
            Tk = [self.cosm.Tgas(z), self.cosm.TCMB(z)]
            cheat = self.simpl.heating_rate_C(z)
            
            pl.plot([Tk[0]] * 2, cheat, color='w', ls=ls[i])
            pl.plot([Tk[1]] * 2, cheat, color='w', ls=ls[i])
            pl.plot(Tk, [cheat[0]]*2, color='w', ls=ls[i])
            pl.plot(Tk, [cheat[1]]*2, color='w', ls=ls[i]) 
            
            pl.annotate(r'$z=%i$' % z, (np.mean(Tk), cheat[1]), va='bottom',
                ha='center', color='w', fontsize=16)               
            
        #ax.plot(Tk, cheat[:,0], color='w', ls='-')
        #ax.plot(Tk, cheat[:,1], color='w', ls='-')
        #ax.plot([Tk[0]]*2, cheat[0], color='w', ls='-')
        #ax.plot([Tk[1]]*2, cheat[1], color='w', ls='-')
                
        #cb = pl.colorbar()
        #cb.set_label('Model Density', rotation=270)        
                
    def analyzeD(self, require_3pt=True, bin_dex=0.15):
        """
        Make standard plots for given turning point.
        """

        if require_3pt:
            data = self.dictify_data(self.extract_3pt_models()[0])
        else:
            data = self.data
            
        Gamma_RC = data['Gamma_igm_D']
        heat_RC = data['heat_igm_D']
        Gamma = (data['Gamma_igm_D'] + data['gamma_igm_D']) \
            / ((1. - data['xe_D']) / self.cosm.nH(data['z_D']))
        heat = data['heat_igm_D'] \
            / ((1. - data['xe_D']) / self.cosm.nH(data['z_D']))    
            
        heat *= cm_per_mpc**3 / (1. + data['z_D'])**3
            
        h, x, y = np.histogram2d(data['xavg_D'], heat,
            bins=(10**np.arange(-2.5, 0.1, bin_dex),
                  10**np.arange(36, 43, bin_dex)))

        pl.contourf(rebin(x), rebin(y), h.T)
        pl.xlabel(r'$\overline{x}_i (z_{\mathrm{D}})$')
        pl.ylabel(r'$\epsilon_X (z_{\mathrm{D}}) \ \left[\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cMpc}^{-3} \right]$')
        pl.xscale('linear')
        pl.yscale('log')

        pl.xticks(np.arange(0, 1.2, 0.2))
        pl.xlim(0, 0.6)

    def pdf_1d(self, parameter='fstar'):
        """
        Plot a 1-D histogram (posterior probability) for given parameter.
        """
        
        pass
        
    def pdf2d(self, parameter='fstar', pinfo=None, dinfo=None):
        pass

    def plot_grid(self, tp='D', require_3pt=True, bin_dex=0.2, fignum=1):
        
        if require_3pt:
            data = self.dictify_data(self.extract_3pt_models()[0])
        else:
            data = self.data
        
        I = ['Gamma_igm', 'Gamma_HII', 'heat_igm', 'Ja']
        J = ['xi', 'xe', 'xavg', 'Tk']
        col_labels = [r'$x_i (z_{\mathrm{%s}})$' % tp, 
                      r'$x_e (z_{\mathrm{%s}})$' % tp,
                      r'$\overline{x}_i (z_{\mathrm{%s}})$' % tp,
                      r'$T_K (z_{\mathrm{%s}})$' % tp]
        row_labels = [r'$\Gamma_{\mathrm{IGM}} (z_{\mathrm{%s}}) \ \left[\mathrm{s}^{-1}\right]$' % tp,
            r'$\Gamma_{\mathrm{HII}} (z_{\mathrm{%s}}) \ \left[\mathrm{s}^{-1}\right]$' % tp,
            r'$\epsilon_X (z_{\mathrm{%s}}) \ \left[\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cMpc}^{-3} \right]$' % tp,
            r'$J_{\alpha} (z_{\mathrm{%s}}) / J_{21}$' % tp]
        
        mp = multipanel(dims=(len(I),len(J)), panel_size=(0.6, 0.6),
            num=fignum)
        
        for i in range(len(I)): # row
            for j in range(len(J)): # column

                if I[i] in ['Gamma_igm', 'Gamma_HII']:
                    quant = (data['%s_%s' % (I[i], tp)] + data['%s_%s' % (I[i], tp)]) \
                        / ((1. - data['xe_%s' % tp]) / self.cosm.nH(data['z_%s' % tp]))
                    ybins = 10**np.arange(-23, -17.5, bin_dex)
                    
                elif I[i] == 'Ja':
                    quant = data['%s_%s' % (I[i], tp)] / J21_num
                    ybins = 10**np.arange(-13.5, -7, bin_dex) / J21_num
                else:
                    quant = data['heat_igm_%s' % tp] \
                        / ((1. - data['xe_%s' % tp]) / self.cosm.nH(data['z_%s' % tp]))    

                    quant *= cm_per_mpc**3 / (1. + data['z_%s' % tp])**3
                    
                    ybins = 10**np.arange(36, 43, bin_dex)
                   
                if J[j] == 'Tk':
                    xbins = 10**np.arange(0, 4, bin_dex)
                else:
                    xbins = 10**np.arange(-4, 0, bin_dex)
                    
                    
                    
                h, x, y = np.histogram2d(data['%s_%s' % (J[j], tp)], quant,  
                    bins=(xbins, ybins))    
                
                axis_num = mp.axis_number(i, j)
                mp.grid[axis_num].contourf(rebin(x), rebin(y), h.T)

                mp.grid[axis_num].set_xscale('log')                    
                mp.grid[axis_num].set_yscale('log')
                
                # axes labels
                if i == 0:
                    mp.grid[axis_num].set_xlabel(col_labels[j])
                if j == 0:
                    mp.grid[axis_num].set_ylabel(row_labels[i])   
                    
                # Lines of constant ionized fraction
                #if j < 3:
                #    for xi in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
                #        mp.grid[axis_num].loglog([xi] * 2, 
                #            [ybins.min(), ybins.max()], color='w', ls=':')
                    
                        #if i == 0:
                        #    mp.grid[axis_num].annotate('%.2g' % xi, (xi, 1e41), rotation=270, color='w', 
                        #        fontsize=16, ha='left', va='center')
                
                # Remove ticklabels
                if j > 0:
                    mp.grid[axis_num].set_yticklabels([])    
                if i > 0:
                    mp.grid[axis_num].set_xticklabels([])                    
        
                mp.grid[axis_num].set_xlim(xbins.min(), xbins.max())
                
        return mp        
        