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
                    tmp['%s{%i}' % (par, i)] = self._inputs[par]

            self._inputs.update(tmp)
    
    def TurningPoint(self, pt='C', **kwargs):
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

    def _AnalyzeTurningPointC(self, mp=None, fig=1, nu=[0.68,0.95], bins=20, 
        color='k', show_analytic=False, annotate=False, multiplier=None,
        inset=False):   
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

        colors = color        
        pars = ['igm_Tk', 'igm_heat', 'Ja']

        if multiplier is None:
            mult = [1., cm_per_mpc**3, 1. / J21_num]
        else:
            mult = multiplier

        # New labels
        labels = \
        { 
         'igm_Tk': r'$T_{\mathrm{K}} / \mathrm{K}$',
         'igm_heat': r'$\epsilon_X$',
         'Ja': r'$J_{\alpha} / J_{21}$'
        }

        mp = self.TrianglePlot(pars,    
            z='C', color_by_like=True, filled=False, nu=nu,
            inputs=self.inputs, linestyles=['--', '-'], bins=bins, fig=fig, 
            take_log=[False, True, False], mp=mp, color=color, colors=colors,
            multiplier=mult, labels=labels)

        # Add panels for constraints on the turning point itself?
        if inset:
            mp_inset = self.TrianglePlot(['nu', 'dTb'], fig=mp.fig, z='C', 
                color_by_like=True, diagonal='upper', 
                shift_x=0.05, shift_y=0.05, label_panels=None,
                dims=(3, 3), keep_diagonal=False)
        else:
            mp_inset = None
            
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

        return mp, mp_inset
            
    def _AnalyzeTurningPointD(self, mp=None, fig=1, nu=[0.68,0.95], bins=20, 
        color='k', show_analytic=False, multiplier=None):
        """
        Plot the four main quantities probed by turning point D.
        """
        
        c = colors = color
        
        pars = ['igm_Tk', 'cgm_h_2', 'igm_heat', 'cgm_Gamma_h_1']

        if multiplier is None:
            mult = [1., cm_per_mpc**3, 1. / J21_num, 1e16]
        else:
            mult = multiplier
        
        # New labels
        labels = \
        { 
         'igm_Tk': r'$T_{\mathrm{K}} / \mathrm{K}$',
         'cgm_h_2': r'$Q_{\mathrm{HII}}$',
         'igm_heat': r'$\epsilon_X$',
         'cgm_Gamma_h_1': r'$\Gamma_{16}$',
        }
        
        mp_inset = None
        mp = self.TrianglePlot(pars, 
            z='D', color_by_like=True, filled=False, nu=nu, 
            take_log=[False, False, True, False], labels=labels,
            color=c, colors=c, bins=bins, mp=mp, fig=fig,
            linestyles=['--', '-'])
      
        return mp, mp_inset
      
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
        
    def ParameterConstraints(self, pars=None, mp=None, fig=1, bins=20, 
        ares_only=True, color='k'):
        """
        Make a triangle plot of constraints on model parameters.
    
        Parameters
        ----------
        ares_only : bool
            If True, only includes ares parameters in the triangle plot, i.e.,
            exclude foreground / instrument parameters.
            
        """
        
        c = colors = color
        
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
                
        mp = self.TrianglePlot(pars, color_by_like=True, inputs=self.inputs, 
            bins=bins, linestyles=['--', '-'], filled=False, color=c, colors=c, 
            mp=mp)
            
        return mp
            
            
            
            