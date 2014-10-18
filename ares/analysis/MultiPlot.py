"""

MultiPlot.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-06-27.

Description: Make multipanel plots with shared axes, with or without AxesGrid.
     
"""

import numpy as np
from math import ceil
import matplotlib.pyplot as pl

# Matplotlibrc defaults
figsize = pl.rcParams['figure.figsize']
wspace = pl.rcParams['figure.subplot.wspace']
hspace = pl.rcParams['figure.subplot.hspace']

def AxisConstructor(nr, nc, panel): 
    return pl.subplot(nr, nc, panel)
    
defs = \
{
 'dims':(2,2), 
 'padding':(0,0), 
 'panel_size':(1,1), 
 'num':1,
 'diagonal':None,
 'left':None, 
 'right':None, 
 'bottom':None, 
 'top':None, 
 'preserve_margins':True
}

class MultiPanel:
    def __init__(self, **kwargs):
        """
        Initialize multipanel plot object.
        
        Parameters
        ----------
        dims : tuple
            (nrows, ncols)
        padding : tuple
            separation of subplots in inches, (wspace, hspace)
        panel_size : tuple
            fraction of default plot-area in each dimension for a single pane
        num : int
            identification number for figure in case you plan on making a
            bunch simultaneously
        diagonal : bool
            If True, do not initialize panels above the diagonal. Will throw
            an error if dims[0] != dims[1].
        preserve_margins : bool
            Keep proper size of left, right, bottom, top fixed? If False, 
            will be fraction of figsize, as usual in matplotlib.

        """

        tmp = defs.copy()
        tmp.update(kwargs)

        for kw in tmp:
            exec('%s = tmp[\'%s\']' % (kw, kw))
        
        if left is None:
            left = pl.rcParams['figure.subplot.left']
        if right is None:
            right = pl.rcParams['figure.subplot.right']
        if bottom is None:
            bottom = pl.rcParams['figure.subplot.bottom']
        if top is None:
            top = pl.rcParams['figure.subplot.top']

        self.square = dims[0] == dims[1]
        
        if (diagonal is not None) and not self.square:
            raise ValueError('Must have square matrix to use diagonal=True')
        
        self.dims = dims
        self.J, self.K = dims
        self.padding = padding
        
        self.pane_size = np.array(figsize) * np.array([right-left, top-bottom])
        self.pane_size *= np.array(panel_size)
        
        self.panel_size = np.zeros(2)
        self.panel_size[0] = self.pane_size[0] * self.K + padding[0] * (self.K - 1)
        self.panel_size[1] = self.pane_size[1] * self.J + padding[1] * (self.J - 1)     
        self.panel_size[0] += figsize[0] * (left + (1. - right))
        self.panel_size[1] += figsize[1] * (bottom + (1. - top))   
                                                                
        self.diagonal = diagonal
        self.share_x = self.padding[1] == 0
        self.share_y = self.padding[0] == 0        
        self.share_all = self.share_x and self.share_y
            
        # Create figure        
        self.fig = pl.figure(num, self.panel_size)

        # Adjust padding
        if preserve_margins:
            l = left * figsize[0] / self.panel_size[0]
            r = (left * figsize[0] + self.K * self.pane_size[0]) \
                / self.panel_size[0]
            b = bottom * figsize[1] / self.panel_size[1]
            t = (bottom * figsize[1] + self.J * self.pane_size[1]) \
                / self.panel_size[1]
        else:
            l, r, b, t = left, right, bottom, top
        
        self.fig.subplots_adjust(left=l, right=r, bottom=b, top=t, 
            wspace=self.padding[0], hspace=self.padding[1])
        
        # Important attributes for identifying individual panels
        self.N = int(np.prod(self.dims))
        self.elements = list(np.reshape(np.arange(self.N), self.dims))
        self.elements.reverse()
        self.elements = np.array(self.elements)
        
        # Dimensions of everything (in fractional units)
        self.window = {'left': l, 'right': r, 'top': t, 
            'bottom': b, 'pane': ((r-l) / float(dims[0]), (t-b) / float(dims[1]))}
        
        self.xaxes = self.elements[-1]
        self.yaxes = zip(*self.elements)[0]                  
        self.lowerleft = self.elements[-1][0]
        self.lowerright = self.elements[-1][-1]
        self.upperleft = self.elements[0][0]
        self.upperright = self.elements[0][-1]
                
        if self.square:
            self.diag = np.diag(self.elements)        
        else:
            self.diag = None        
                
        self.left = []
        self.right = []
        self.bottom = []
        self.top = []
        for i in xrange(self.N):
            j, k = self.axis_position(i)
            
            if k == 0:
                self.bottom.append(i)
            if k == self.K - 1:
                self.top.append(i)    
            if j == 0:
                self.left.append(i)
            if j == self.J - 1:
                self.right.append(i)       

        # Create subplots
        l = self.elements.flatten()
        self.grid = [None for i in xrange(self.N)]
        for i in xrange(self.N):                
            j, k = self.axis_position(i)
            
            if diagonal == 'lower':
                if k >= (self.dims[1] - j) and i not in self.diag:
                    continue
            if diagonal == 'upper':
                if k < (self.dims[1] - j) and i not in self.diag:
                    continue        
            
            #if diagonal == 'lower' and j == k and (j, k) != (0, 0):
            #    continue
            #if diagonal == 'upper' and j == k and (j, k) != (self.J-1, self.K-1):
            #    continue

            self.grid[i] = AxisConstructor(self.J, self.K, l[i]+1)
            
    def axis_position(self, i):
        """
        Given axis ID number, return indices describing its (x, y) position.
        
        So, (column, row).
        """
        
        # This is hacky but oh well
        if self.dims[0] == 1:
            return (i, 0)
        elif self.dims[1] == 1:
            return (0, i)        
        else:
            return i % self.dims[1], int(i / self.dims[0])
        
    def axis_number(self, j, k):
        """
        Given indices describing a panel's (x,y) position, return ID number.
        
        Parameters
        ----------
        j : int
            index describing y location (i.e., row #). 0 < j < self.dims[0]
        k : int
            index describing x location (i.e., col #). 0 < k < self.dims[1]
        
        """
        
        i = j * self.dims[1] + k
        if self.above_diagonal(i):
            return None
        if i >= self.N:
            return None
        return i
    
    def above_diagonal(self, i):
        """ Is the given element above the diagonal? """
        
        if self.diagonal is None:
            return False
        
        if i in self.diag:
            return False
        
        j, k = self.axis_position(i)
        
        if self.diagonal == 'lower':
            if j == k and j > (self.J / 2.):
                return True
            else:
                return False
        elif self.diagonal == 'upper' and j < k:
            return True
        else:
            return False
            
    def rescale_axes(self, x=True, y=True, xlim=None, ylim=None,    
        tighten_up=0):
        """ 
        Force x/y limits of each column/row to be the same. 
        
        Assumes elements along the diagonal are 1D profiles, not 2D images, 
        so skips over them when applying the changes.
        
        Parameters
        ----------
        tighten_up : float
            Clip x and y limits by this fractional amount.
        
        """                               
        
        # First, figure out what limits should be
        col_xlim = [[1e10, -1e10] for i in xrange(self.dims[0])]
        row_ylim = [[1e10, -1e10] for i in xrange(self.dims[1])]
        
        # Loop over axes
        for i in xrange(self.N):
            if self.grid[i] is None:
                continue
            
            # column, row
            j, k = self.axis_position(i)
            
            if self.above_diagonal(i):
                continue
            
            if x and xlim is None:
                col_xlim[j][0] = min(col_xlim[j][0], self.grid[i].dataLim.min[0])
                col_xlim[j][1] = max(col_xlim[j][1], self.grid[i].dataLim.max[0])    
            elif x:
                col_xlim[j][0] = xlim[0]
                col_xlim[j][1] = xlim[1]
                
            if self.diagonal is not None and i in self.diag:
                continue
                
            if y and (ylim is None): 
                row_ylim[k][0] = min(row_ylim[k][0], self.grid[i].dataLim.min[1])
                row_ylim[k][1] = max(row_ylim[k][1], self.grid[i].dataLim.max[1])    
            elif y:
                row_ylim[k][0] = ylim[0]
                row_ylim[k][1] = ylim[1]    
                
        # Apply limits    
        for i in xrange(self.N):
            if self.grid[i] is None:
                continue
                
            # column, row    
            j, k = self.axis_position(i)
            
            col_tmp = [col_xlim[j][0] * (1. + tighten_up * np.sign(col_xlim[j][0])),
                       col_xlim[j][1] * (1. - tighten_up * np.sign(col_xlim[j][1]))]
                       
            row_tmp = [row_ylim[k][0] * (1. + tighten_up * np.sign(row_ylim[k][0])),
                       row_ylim[k][1] * (1. - tighten_up * np.sign(row_ylim[k][1]))]
            
            self.grid[i].set_xlim(col_tmp)
            
            if self.diagonal and i in self.diag:
                continue

            self.grid[i].set_ylim(row_tmp)
    
        pl.draw()         
    
    def fix_axes_ticks(self, axis='x', style=None, dtype=float, N=None, 
        rotate_x=False):
        """
        Remove overlapping tick labels between neighboring panels.
        """
        
        # Grab functions we need by name
        get_lim = "get_%slim" % axis
        get_ticks = "get_%sticks" % axis
        get_ticklabels = "get_%sticklabels" % axis
        set_ticks = "set_%sticks" % axis
        set_ticklabels = "set_%sticklabels" % axis
        shared = eval("self.share_%s" % axis)
                
        if axis == 'x':
            j = 0
            if shared:
                axes = self.xaxes
            else:
                axes = np.arange(self.N)
        elif axis == 'y':
            j = 1
            if shared:
                axes = self.yaxes
            else:
                axes = np.arange(self.N)
        else:
            raise ValueError('axis must be set to \'x\' or \'y\'')
        
        # Loop over axes and make corrections
        for i in axes:
            if self.diagonal:
                if self.above_diagonal(i):
                    continue
                    
            if self.axis_position(i)[j] == self.dims[j]:
                pass
            
            # Retrieve current ticks, tick-spacings, and axis limits
            ticks = eval("list(self.grid[%i].%s())" % (i, get_ticks))
            
            if not ticks:
                continue
            
            ticklabels = eval("[tick for tick in self.grid[%i].%s()]" \
                % (i, get_ticklabels))
            #ticklabels = eval("list(self.grid[%i].%s())" % (i, get_ticklabels))
            
            dt = np.diff(ticks)[0]
            limits = eval("self.grid[%i].%s()" % (i, get_lim))
            
            pos = self.axis_position(i)[j]

            # Mess with upper limits
            ul = None
            if shared and pos < (self.dims[int(not j)] - 1):
                ul = -1
                
            labels = [tick.get_text() for tick in ticklabels]
                                           
            Nticks = len(labels)
            
            if N is not None:
                mi, ma = round(limits[0], 0), round(limits[1])
                dt = (ma - mi) / N
                
                if dt < 1:
                    dt = np.round(dt, abs(int(np.log10(dt))) + 1)
                    ticks = np.arange(mi, mi + (N+1)*dt, dt)
                else:
                    ticks = np.round(np.linspace(mi, ma, N), 1)
                
                labels = ['%g' % val for val in ticks]
                                              
            
            if ul is None:
                eval("self.grid[%i].%s(ticks[0:])" % (i, set_ticks))
                if rotate_x:
                    eval("self.grid[%i].%s(labels[0:], rotation=270)" \
                        % (i, set_ticklabels))
                else:
                    eval("self.grid[%i].%s(labels[0:])" % (i, set_ticklabels))
            else:
                eval("self.grid[%i].%s(ticks[0:%i])" % (i, set_ticks, ul))
                
                if rotate_x:
                    eval("self.grid[%i].%s(labels[0:%i], rotation=270)" \
                        % (i, set_ticklabels, ul))
                else:
                    eval("self.grid[%i].%s(labels[0:%i])" % (i, set_ticklabels, ul))
    
            if style is not None: 
                self.grid[i].ticklabel_format(style=style)
        
        # Remove ticklabels of interior panels completely
        if shared:
            for i in xrange(self.N):
                if self.grid[i] is None:
                    continue
                
                if (self.diagonal is not None):
                    if self.above_diagonal(i):
                        continue
                    
                if i not in axes:
                    eval("self.grid[%i].%s([])" % (i, set_ticklabels))
                
        pl.draw()

    def fix_ticks(self, noxticks=False, noyticks=False, style=None, N=None,
        rotate_x=False, xticklabels=None, yticklabels=None):
        """
        Call once all plotting is done, will eliminate redundant tick marks 
        and what not.
        """
        
        pl.draw()
        
        self.fix_axes_ticks(axis='x', N=N, rotate_x=rotate_x)
        self.fix_axes_ticks(axis='y', N=N)
        
        if self.diagonal == 'lower':
            self.grid[np.intersect1d(self.left, self.top)[0]].set_yticklabels([])
        
        # Remove ticks alltogether (optionally)            
        for j in xrange(self.dims[0]):
            for k in xrange(self.dims[1]):
                i = self.axis_number(j,k)  
                
                if i is None:
                    continue
                    
                if self.grid[i] is None:
                    continue  
                    
                if self.diagonal == 'lower':
                    if i not in self.bottom:
                        self.grid[i].set_xticklabels([])    
        
                if self.diagonal and (i in self.diag):
                    self.grid[i].set_yticks([])
        
                if noxticks:
                    self.grid[i].set_xticks([])

                if xticklabels is not None:
                    x = self.grid[i].get_xlim()
                    bins = np.arange(round(min(x), 1), round(max(x), 1), 
                        xticklabels)
                    self.grid[i].set_xticks(bins)    

                if noyticks:
                    self.grid[i].set_yticks([])
                
                if yticklabels is not None:
                    y = self.grid[i].get_ylim()
                    bins = np.arange(round(min(y), 1), round(max(y), 1), 
                        yticklabels)
                    self.grid[i].set_yticks(bins)    
                
        pl.draw()     
        
    def fix_auto(self):
        """
        Automatically figure out what needs fixing. Rescale by default.
        """
        if self.share_x:
            self.rescale_axes(x=True, y=False)
            self.fix_axes_ticks(axis='x')
        if self.share_y:
            self.rescale_axes(x=False, y=True)
            self.fix_axes_ticks(axis='y')
            
    def add_labels(self, labels):
        """
        Add axis labels only on left and bottom. Meant for diagonal=True.
        
        labels : list
            List of axes labels in order of ascending panel in x. i.e., from
            leftmost to rightmost plots.
            
        """
        for i, axis in enumerate(self.bottom):
            self.grid[axis].set_xlabel(labels[i])
                
        for i, axis in enumerate(np.array(self.left)[-1::-1]):
            if axis == self.upperleft:
                continue
                
            self.grid[axis].set_ylabel(labels[i])        
        
        pl.draw()          
    
    def global_xlabel(self, label, xy=(0.5, 0.025), size='x-large'):
        """ Set shared xlabel. """        
        
        self.fig.text(xy[0], xy[1], label, 
            ha='center', va='center', size=size)
            
    def global_ylabel(self, label, xy=(0.025, 0.5), size='x-large'):
        """ Set shared ylabel. """        
        
        self.fig.text(xy[0], xy[1], label, 
            ha='center', va='center', rotation='vertical', size=size)
    
    def title(self, label, xy=(0.5, 0.95), size='x-large'):
        self.fig.text(xy[0], xy[1], label, 
            ha='center', va='center', size=size)

    def draw(self):
        pl.draw()
        
        