"""

ProgressBar.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Dec 27 16:27:51 2012

Description: Wrapper for progressbar2.

"""

from .PrintInfo import width

try:
    import progressbar
    pb = True
except ImportError:
    pb = False
    
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1    

class ProgressBar(object):
    def __init__(self, maxval, name='ares', use=True):
        self.maxval = maxval            
        self.use = use
        
        self.has_pb = False
        if pb and rank == 0 and use:
            self.widget = ["{!s}: ".format(name), progressbar.Percentage(), ' ', \
              progressbar.Bar(marker='#'), ' ', \
              progressbar.ETA(), ' ']

    def start(self):
        if pb and rank == 0 and self.use:
            self.pbar = progressbar.ProgressBar(widgets=self.widget,
                max_value=self.maxval, redirect_stdout=False, 
                term_width=width+1).start()                
            self.has_pb = True

    def update(self, value):
        if self.has_pb:
            self.pbar.update(value)

    def finish(self):
        if self.has_pb:
            self.pbar.finish()

