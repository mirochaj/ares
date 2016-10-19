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

def AxisConstructor(fig, nr, nc, panel): 
    return fig.add_subplot(nr, nc, panel)
    
defs = \
{
 'dims':(2,2), 
 'padding':(0,0), 
 'panel_size':(1,1), 
 'fig':1,
 'diagonal':None,
 'left':None, 
 'right':None, 
 'bottom':None, 
 'top':None, 
 'preserve_margins':True,
 'keep_diagonal': True,
 'shift_x': 0.0,
 'shift_y': 0.0,
 'active_panels': None,
}

class MultiPanel(object):
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
        fig : int
            identification number for figure in case you plan on making a
            bunch simultaneously
        diagonal : bool
            If True, do not initialize panels above the diagonal. Will throw
            an error if dims[0] != dims[1].
        preserve_margins : bool
            Keep proper size of left, right, bottom, top fixed? If False, 
            will be fraction of figsize, as usual in matplotlib.
        active_panels : tuple
            If, for example, dims = (4, 4) but active = (3, 3), then only
            the 9 panels anchored to the origin will be created.
            
        Example
        -------
        >>> from ares.analysis import MultiPanel
        >>> mp = MultiPanel(dims=(2,2))
        >>> # axes stored in mp.grid    

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
            
        self.l = left
        self.r = right
        self.b = bottom
        self.t = top    
            
        self.square = dims[0] == dims[1]
        
        if (diagonal is not None) and not self.square:
            raise ValueError('Must have square matrix to use diagonal=True')

        self.dims = tuple(dims)
        self.J, self.K = dims # J = nrows, K = ncols
        self.nrows = self.J
        self.ncols = self.K
        
        if type(padding) is float:
            padding = tuple([padding]* 2)
    
        self.padding = padding
                
        # Size of an individual panel (in inches)
        self.pane_size = np.array(figsize) * np.array([right-left, top-bottom])
        self.pane_size *= np.array(panel_size)

        # Now, figure out the size of the entire figure (in inches)
        self.panel_size = np.zeros(2)
        
        # After these two lines, self.panel_size is equal to the size of the
        # panel-filled area of the window (in inches)
        self.panel_size[0] = self.pane_size[0] * self.K + padding[0] * (self.K - 1)
        self.panel_size[1] = self.pane_size[1] * self.J + padding[1] * (self.J - 1)     

        # Add empty area above/below and left/right of panel-filled area
        self.panel_size[0] += figsize[0] * (left + (1. - right))
        self.panel_size[1] += figsize[1] * (bottom + (1. - top))

        self.panel_size_rel = self.pane_size / self.panel_size

        self.diagonal = diagonal
        self.keep_diagonal = keep_diagonal
        self.share_x = self.padding[1] == 0
        self.share_y = self.padding[0] == 0        
        self.share_all = self.share_x and self.share_y

        self.dx = shift_x
        self.dy = shift_y

        # Create figure
        if type(fig) is not int:
            self.fig = fig
            new_fig = False
            l, r = fig.subplotpars.left, fig.subplotpars.right
            b, t = fig.subplotpars.bottom, fig.subplotpars.top
        else:
            self.fig = pl.figure(fig, self.panel_size)
            new_fig = True

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
        #self.window = {'left': l, 'right': r, 'top': t, 
        #    'bottom': b, 'pane': ((r-l) / float(dims[0]), (t-b) / float(dims[1]))}
        
        self.xaxes = self.elements[-1]
        self.yaxes = zip(*self.elements)[0]                  
        self.lowerleft = self.elements[-1][0]
        self.lowerright = self.elements[-1][-1]
        self.upperleft = self.elements[0][0]
        self.upperright = self.elements[0][-1]
                
        if self.square:
            self.diag = np.diag(self.elements)        
            self.interior = list(self.elements.ravel())
            for element in self.diag:
                self.interior.remove(element)
        else:
            self.diag = None        
            
        self.left = []
        self.right = []
        self.bottom = []
        self.top = []
        for i in xrange(self.N):
            k, j = self.axis_position(i)  # col, row
            
            if j == 0:
                self.bottom.append(i)
            if j == self.nrows - 1:
                self.top.append(i)    
            if k == 0:
                self.left.append(i)
            if k == self.ncols - 1:
                self.right.append(i)       

        # Create subplots
        e_fl = self.elements.flatten()
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
            
            if self.square:
                if i in self.diag and not keep_diagonal:
                    continue
            
            if new_fig:
                self.grid[i] = AxisConstructor(self.fig, self.J, self.K, e_fl[i]+1)
            else:

                # col, row = j, k

                lef = l + j * self.panel_size_rel[0] \
                    + self.padding[0] + self.dx
                bot = b + k * self.panel_size_rel[1] \
                    + self.padding[1] + self.dy

                rect = [lef, bot, self.panel_size_rel[0], self.panel_size_rel[1]]

                self.grid[i] = self.fig.add_axes(rect)
            
    @property
    def active_elements(self):
        if not hasattr(self, '_active_elements'):
            self._active_elements = []
            for i in range(self.N):
                if self.grid[i] is None:
                    continue
                
                self._active_elements.append(i)
                
        return self._active_elements
            
    @property
    def elements_by_column(self):
        """
        Create a list of panel ID numbers sorted by column number. 
        
        Each element of the list is a sublist of ID numbers in order of increasing
        row, from bottom to top.
        """
        if not hasattr(self, '_columns'):
            self._columns = [[] for i in range(self.dims[1])]
            for element in self.active_elements:
                col, row = self.axis_position(element)        
                
                self._columns[col].append(element)
                
        return self._columns
        
    @property
    def elements_by_row(self):
        """
        Create a list of panel ID numbers sorted by column number. 
    
        Each element of the list is a sublist of ID numbers in order of increasing
        row, from bottom to top.
        """
        if not hasattr(self, '_rows'):
            self._rows = [[] for i in range(self.dims[0])]
            for element in self.active_elements:
                col, row = self.axis_position(element)
                self._rows[row].append(element)
    
        return self._rows
        
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
            return tuple(np.argwhere(self.elements[-1::-1] == i)[0][-1::-1])

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
        
        if j >= self.dims[0]:
            return None
        if k >= self.dims[1]:
            return None
        
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
            
    def align_labels(self, xpadding=0.5, ypadding=None):
        """
        Re-draw labels so they are a constant distance away from the *axis*,
        not the axis *labels*.
        """
        
        if ypadding is None:
            ypadding = xpadding
        
        for i in self.bottom:
            self.grid[i].xaxis.set_label_coords(0.5, -xpadding)
        for i in self.left:
            self.grid[i].yaxis.set_label_coords(-ypadding, 0.5)
            
        pl.draw()    
            
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
        rotate_x=False, rotate_y=False):
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
        
        # Get locations of ticks on bottom row
        if axis is 'x':
            ticks_by_col = []
            for i in self.bottom:
                ticks_by_col.append(self.grid[i].get_xticks())
        
        # Get locations of ticks on left column
        if axis is 'y':
            ticks_by_row = []
            for i in self.left:
                ticks_by_row.append(self.grid[i].get_xticks())
            
        # Figure out if axes are shared or not    
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
        
        if not shared:
            return
        
        # Loop over axes and make corrections
        for i in axes:
            
            # Skip non-existent elements
            if self.diagonal:
                if self.above_diagonal(i):
                    continue
            
            if self.grid[i] is None:
                continue
            
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
                mi, ma = round(limits[0], 1), round(limits[1], 1)
                
                prec = 2
                while mi == ma:
                    mi, ma = round(limits[0], prec), round(limits[1], prec)
                    prec += 1

                dt = (ma - mi) / float(N)

                if dt < 1:
                    dt = np.round(dt, abs(int(np.log10(dt))) + 1)
                    ticks = np.arange(mi, mi + (N+1)*dt, dt)
                else:
                    ticks = np.round(np.linspace(mi, ma, N), 1)
                
                labels = ['%g' % val for val in ticks]
                                              
            if (axis == 'x' and rotate_x):
                rotate = rotate_x
            elif (axis == 'y' and rotate_y):
                rotate = rotate_y
            else:
                rotate = False
                        
            if ul is None:
                eval("self.grid[%i].%s(ticks[0:])" % (i, set_ticks))
                                
                if rotate:
                    if type(rotate) == bool:
                        eval("self.grid[%i].%s(labels[0:], rotation=90)" \
                            % (i, set_ticklabels))
                    else:
                        eval("self.grid[%i].%s(labels[0:], rotation=%g)" \
                                % (i, set_ticklabels, rotate))        
                else:
                    eval("self.grid[%i].%s(labels[0:])" % (i, set_ticklabels))
            else:
                eval("self.grid[%i].%s(ticks[0:%i])" % (i, set_ticks, ul))
                
                if rotate:
                    if type(rotate) == bool:
                        eval("self.grid[%i].%s(labels[0:%i], rotation=90)" \
                            % (i, set_ticklabels, ul))
                    else:
                        eval("self.grid[%i].%s(labels[0:%i], rotation=%g)" \
                            % (i, set_ticklabels, ul, rotate))      
                else:
                    eval("self.grid[%i].%s(labels[0:%i])" % (i, set_ticklabels, ul))
    
            if style is not None: 
                self.grid[i].ticklabel_format(style=style)
                
        # Loop over columns, force those not in row 0 to share ticks with 
        # whatever tick marks there are in row #0
        if axis == 'x':
            
            for k in range(len(self.elements_by_column)):
                loc = self.axis_number(0, k)
                xticks = self.grid[loc].get_xticks()
                xlim = self.grid[loc].get_xlim()
                
                for h, element in enumerate(self.elements_by_column[k]):
                    if element in self.bottom:
                        continue        
                        
                    self.grid[element].set_xticks(xticks)
                    self.grid[element].set_xlim(xlim)
            
        # Same deal for y ticks
        if axis == 'y':
            for k in range(len(self.elements_by_row)):
                loc = self.axis_number(k, 0)
                yticks = self.grid[loc].get_yticks()
                ylim = self.grid[loc].get_ylim()
                
                for h, element in enumerate(self.elements_by_row[k]):
                    if element in self.left:
                        continue  
                    if self.diag is not None:
                        if element in self.diag:
                            continue              
                        
                    self.grid[element].set_yticks(yticks)
                    self.grid[element].set_ylim(ylim)
                                  
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

    def fix_axes_labels(self):
        for i in xrange(self.N):

            if self.grid[i] is None:
                continue

            # (column, row)
            j, k = self.axis_position(i)

            if j > 0:
                self.grid[i].set_ylabel('')
            if k > 0:
                self.grid[i].set_xlabel('')

    def fix_ticks(self, noxticks=False, noyticks=False, style=None, N=None,
        rotate_x=False, rotate_y=False, xticklabels=None, yticklabels=None, 
        oned=True):
        """
        Call once all plotting is done, will eliminate redundant tick marks 
        and what not.
        
        Parameters
        ----------
        
        """
        
        pl.draw()
        
        self.fix_axes_labels()
        
        self.fix_axes_ticks(axis='x', N=N, rotate_x=rotate_x)
        self.fix_axes_ticks(axis='y', N=N, rotate_y=rotate_y)
        
        if self.diagonal == 'lower' and oned:
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
        
                if self.diagonal and (i in self.diag) and oned:
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
    
    def set_ticks(self, ticks, column=None, row=None, minor=False, round_two=False):
        """
        Replace ticks and labels for an entire column or row all at once.
        
        If operating on a column, ticks are assumed to be for x axes.
        If operating on a row, ticks are assumed to be for y axes.
        
        """
        
        assert (column is not None) or (row is not None), \
            "Must supply column or row number!"
        assert not ((column is not None) and (row is not None)), \
            "Must supply column *or* row number!"
        
        # Could do this more compactly but who cares.
        if column is not None:
            elements = self.elements_by_column
            for j, panel_set in enumerate(elements):
                for k, panel in enumerate(panel_set):
                    if j != column:
                        continue
                        
                    self.grid[panel].set_xticks(ticks, minor=minor)
                    if (not minor):
                        self.grid[panel].set_xticklabels(map(str, ticks))
                        
            # Just apply to relevant rows, too.
            if not round_two:
                self.set_ticks(ticks, row=self.nrows-column-1, minor=minor, 
                    round_two=True)
        else:
            elements = self.elements_by_row
            for j, panel_set in enumerate(elements):
                for k, panel in enumerate(panel_set):
                    if j != row:
                        continue
                        
                    if panel in self.diag:
                        continue
            
                    self.grid[panel].set_yticks(ticks, minor=minor)
                    if (not minor):
                        self.grid[panel].set_yticklabels(map(str, ticks))
             
            if not round_two:
                self.set_ticks(ticks, column=self.nrows-row-1, minor=minor, 
                    round_two=True)
                               
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
        
    def save(self, fn):
        pl.savefig(fn)    
        
        