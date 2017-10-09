"""

Global21cmSet.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Apr 15 09:48:21 MDT 2015

Description: 

"""

import re
import numpy as np
import matplotlib.pyplot as pl
from .ModelSet import ModelSet
from ..physics.Constants import J21_num, cm_per_mpc
from ..util.SetDefaultParameterValues import SetAllDefaults

class Global21cmSet(ModelSet):
    """
    Basically a ModelSet instance with routines specific to the global 21-cm 
    signal.
    """
    
    @property
    def inputs(self):
        if not hasattr(self, '_inputs'):
            self._inputs = None
        
        return self._inputs
        
    @inputs.setter
    def inputs(self, value):
        self._inputs = value
        
        if self.Npops > 1:
            tmp = {}
            for par in self._inputs:
                for i in range(self.Npops):
                    tmp['{0!s}{{{1}}}'.format(par, i)] = self._inputs[par]

            self._inputs.update(tmp)
    
    def _patch_kwargs(self, **kwargs):
                
        # Patch up kwargs
        if 'colors' not in kwargs:
            if 'filled' not in kwargs:
                kwargs['colors'] = ['g', 'b']
            else:
                if kwargs['filled']:
                    kwargs['colors'] = ['g', 'b']
                else:
                    if 'colors' not in kwargs:
                        kwargs['colors'] = 'k'
        
        if 'color' not in kwargs:
            kwargs['color'] = kwargs['colors'][0]
                        
        if 'linestyles' not in kwargs:
            if 'filled' in kwargs:
                if not kwargs['filled']:
                    kwargs['linestyles'] = ['--', '-']
                                                
        return kwargs
    
    def IGMConstraints(self, pt='C', **kwargs):
        """
        Make a triangle plot of constraints on the turning points.
        
        Parameters
        ----------
        """

        assert pt in list('BCD'), 'Only know how to deal with points B, C, and D!'
    
        if pt == 'B':
            raise NotImplemented('help!')
        elif pt == 'C':
            return self._AnalyzeTurningPointC(**kwargs)
        elif pt == 'D':
            return self._AnalyzeTurningPointD(**kwargs)

    def _AnalyzeTurningPointC(self, mp=None, fig=1, nu=[0.68,0.95], 
        show_analytic=False, annotate=False, multiplier=None,
        inset=False, **kwargs):   
        """
        Plot the three main quantities probed by turning point C.
        
        From left-to-right, this will be the IGM temperature, heating rate
        density, and Ly-a background flux.
        
        Parameters
        ----------
        
        
        Returns
        -------
        ares.analaysis.MultiPlot.MultiPanel instance
        
        """

        kwargs = self._patch_kwargs(**kwargs)

        pars = ['Ja', 'igm_Tk', 'igm_heat']

        if multiplier is None:
            mult = [1. / J21_num, 1., cm_per_mpc**3]
        else:
            mult = multiplier

        # New labels
        labels = \
        { 
         'Ja': r'$J_{\alpha} / J_{21}$',
         'igm_Tk': r'$T_{\mathrm{K}} / \mathrm{K}$',
         'igm_heat': r'$\epsilon_X$',
        }
        
        kwargs['labels'] = labels

        mp = self.TrianglePlot(pars, z='C', color_by_like=True, nu=nu,
            inputs=self.inputs, fig=fig, take_log=[False, False, True], mp=mp, 
            multiplier=mult, **kwargs)

        if show_analytic:

            # 2-D constraint on Tk and igm_heat
            self.zC = zC = self.extract_blob('z', z='C')
            self.TC = TC = self.extract_blob('dTb', z='C')
            
            self.hist, xedges, yedges = np.histogram2d(zC, TC)
            hist = hist.T
            
            self.xe, self.ye = xedges, yedges
            
            # Recover bin centers
            zCb = rebin(xedges)
            TCb = rebin(yedges)
                    
            # Determine mapping between likelihood and confidence contours
    
            # Get likelihood contours (relative to peak) that enclose
            # nu-% of the area
            like, levels = self.get_levels(hist, nu=nu)
            
            # Convert contours on z and dTb to Tk igm_heat contours            
            mp.grid[3].contour(bc[0], bc[1], hist / hist.max())

        return mp
            
    def _AnalyzeTurningPointD(self, mp=None, fig=1, nu=[0.68,0.95], 
        show_analytic=False, multiplier=None, **kwargs):
        """
        Plot the four main quantities probed by turning point D.
        """
        
        kwargs = self._patch_kwargs(**kwargs)
                
        pars = [ 'cgm_h_2', 'cgm_Gamma_h_1', 'igm_Tk', 'igm_heat']

        if multiplier is None:
            mult = [1., 1e17, 1., cm_per_mpc**3]
        else:
            mult = multiplier
        
        # New labels
        labels = \
        { 
         'igm_Tk': r'$T_{\mathrm{K}} / \mathrm{K}$',
         'cgm_h_2': r'$Q_{\mathrm{HII}}$',
         'igm_heat': r'$\epsilon_X$',
         'cgm_Gamma_h_1': r'$\Gamma_{-{}}$'.format(np.log10(mult[1])),
        }
        
        mp_inset = None
        mp = self.TrianglePlot(pars, z='D', color_by_like=True, nu=nu, 
            take_log=[False, True, False, True], labels=labels,
            mp=mp, fig=fig, multiplier=mult, inputs=self.inputs, **kwargs)
      
        return mp
      
    def ThermalHistory(self):
        pass  
        
    def IonizationHistory(self):
        pass        
    
    def LyHistory(self):
        pass
    
    def SignalShape(self, z):
        """
        Make a triangle plot of the curvature at all turning points.
    
        Parameters
        ----------
        """
        pass        
        
    def _split_pop_param(self, name):
        """
        Take in parameter name, determine if it has a population ID # suffix.
        
        Returns
        -------
        (parameter name, pop ID #)
        
        """

        # Look for populations
        m = re.search(r"\{([0-9])\}", name)
        
        if m is None:
            prefix = name
            num = None
        else:
            # Population ID number
            num = int(m.group(1))
            
            # Pop ID excluding curly braces
            prefix = name.split(m.group(0))[0]    
            
        return prefix, num
        
    def TurningPointConstraints(self, pts=None, mp=None, fig=1,
        exclude=None, **kwargs):        
        """
        Plot constraints on the turning points positions.
        
        Parameters
        ----------
        pts : list
            Points to include. If None, will do B, C, and D.
        exclude : list
            For example, to exclude the temperature of turning point B, set
            exclude=('dTb', 'B')
        
        """
        
        had_mp = False
        if mp is not None:
            had_mp = True
        
        if pts is None:
            pts = list('BCD')
            z = ['B'] * 2 + ['C'] * 2 + ['D'] * 2
        else:
            z = []
            for pt in pts:
                z.extend([pt]*2)
                
        to_plot = ['nu', 'dTb'] * len(pts)
        
        # May want to exclude, e.g., temperature of turning point B
        if exclude is not None:
            q_ex, z_ex = exclude
            
            not_ok = np.ones(len(z))
            for i, redshift in enumerate(z):
                not_ok[i] *= (redshift == z_ex) and (to_plot[i] == q_ex)
                
            mask = np.array(np.logical_not(not_ok), dtype=bool)
                
            to_plot = list(np.array(to_plot)[mask])
            z = list(np.array(z)[mask])
        
        # Make the plot!
        mp = self.TrianglePlot(to_plot, z=z, inputs=self.inputs, **kwargs)

        # Add annotations along the diagonal (which column is which pt)
        for i in mp.diag:
            row, col = mp.axis_position(i)
            mp.grid[i].set_title(z[-1::-1][col])
            
        if had_mp:
            pl.draw()
            return mp    

        # Draw the 120 MHz cutoff for DARE
        #if pts == 'D' or (pts == list('BCD')):
        #    mp.grid[4].plot([120]*2, mp.grid[4].get_ylim(), color='k', ls='--')
        #    mp.grid[10].plot([120]*2, mp.grid[10].get_ylim(), color='k', ls='--')
        #    
        #    # Plot saturated limit
        #    no_ion = sG21(tanh_model=True, tanh_x0=0., tanh_Tz0=15)
        #    nu_gt_100 = no_ion.history['nu'] > 100.0
        #    x = no_ion.history['nu'][nu_gt_100]
        #    y = no_ion.history['dTb'][nu_gt_100]
        #    mp.grid[4].plot(x, y, color='k', ls='--')
        #    
        #    pl.draw()
            
        return mp    
        
    def ParameterConstraints(self, pars=None, mp=None, ax=None, fig=1, bins=20, 
        ares_only=True, pop=None, triangle=True, **kwargs):
        """
        Make a triangle plot of constraints on model parameters.
        
        Basically a wrapper around ares.analysis.ModelSet.
    
        Parameters
        ----------
        pars : list
            Parameters to include in 
        
        ares_only : bool
            If True, only includes ares parameters in the triangle plot, i.e.,
            exclude foreground / instrument parameters.
            
        """
                
        kwargs = self._patch_kwargs(**kwargs)
                
        # Set the parameters automatically
        if pars is None:
            if ares_only:
                pars = []
                defs = SetAllDefaults()
                for par in self.parameters:
                    
                    prefix, num = self._split_pop_param(par)
                    
                    if prefix not in defs:
                        continue
                        
                    pars.append(par)
            else:
                pars = self.parameters
        
        # Show constraints for just a single population
        if pop is not None and self.Npops > 1:
            tmp = []
            for par in pars:
                
                prefix, num = self._split_pop_param(par)
                
                if num is None:
                    continue
                
                if num != pop:
                    continue
                
                tmp.append(par)
                
            del pars
            pars = tmp
                
        if not triangle and len(pars) == 2:
            ax = self.PosteriorPDF(pars, color_by_like=True, fig=fig,
                bins=bins, filled=False, ax=ax, **kwargs)
            
            if self.inputs is not None:
                
                p1, p2 = pars
                
                if self.is_log[self.parameters.index(p1)]:
                    v1 = np.log10(self.inputs[p1])
                else:
                    v1 = self.inputs[p1]
                
                if self.is_log[self.parameters.index(p2)]:
                    v2 = np.log10(self.inputs[p2])
                else:
                    v2 = self.inputs[p2]   
                
                ax.plot([v1]*2, ax.get_ylim(), color='k', ls=':')
                ax.plot(ax.get_xlim(), [v2]*2, color='k', ls=':')
                pl.draw()
                
            return ax
                                
        # Make the triangle plot already
        mp = self.TrianglePlot(pars, color_by_like=True, inputs=self.inputs, 
            bins=bins, mp=mp, fig=fig, **kwargs)
            
        return mp
            
            
            
            
