#!/usr/bin/env python

import os, re, urllib, shutil, sys, tarfile

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if '--fresh' in sys.argv:
    fresh = True
    sys.argv.remove('--fresh')
else:
    fresh = False
    
# Add CMD line option for preferred file format?
    
ares_link = 'https://bitbucket.org/mirochaj/ares'

ares_packages = \
    ['ares', 'ares.analysis', 'ares.simulations', 'ares.populations',
     'ares.util', 'ares.solvers', 'ares.static', 'ares.sources', 
     'ares.physics', 'ares.inference', 'ares.phenom']

ares_descr = \
"""
The Accelerated Reionization Era Simulations (ares) code was designed to 
rapidly generate models for the global 21-cm signal. It can also be used as 
a 1-D radiative transfer code, stand-alone non-equilibrium chemistry solver, 
or global radiation background calculator.
"""

setup(name='ares',
      version='0.1',
      description='Accelerated Reionization Era Simulations',
      long_description=ares_descr,
      author='Jordan Mirocha',
      author_email='mirochaj@gmail.com',
      url=ares_link,
      packages=ares_packages,
     )
          
# Try to set up $HOME/.ares
HOME = os.getenv('HOME')
if not os.path.exists('%s/.ares' % HOME):
    try:
        os.mkdir('%s/.ares' % HOME)
    except:
        pass
 
# Create files for defaults and labels in HOME directory
for fn in ['defaults', 'labels']:
    if not os.path.exists('%s/.ares/%s.py' % (HOME, fn)):
        try:
            f = open('%s/.ares/%s.py' % (HOME, fn), 'w')
            print >> f, "pf = {}"
            f.close()
        except:
            pass
    
"""
Auxiliary data download.
"""

# Format: [URL, file 1, file 2, ..., file to run when done]

aux_data = \
{
 'hmf': ['%s/downloads' % ares_link, 
    'hmf_ST_logM_1200_4-16_z_1141_3-60.pkl',
    None],
 'inits': ['%s/downloads' % ares_link, 
     'initial_conditions.npz',
     None],    
 'optical_depth': ['%s/downloads' % ares_link, 
    'optical_depth_H_400x1616_z_10-50_logE_2-4.7.pkl',
    'optical_depth_He_400x1616_z_10-50_logE_2-4.7.pkl',
    None],
 'secondary_electrons': ['http://www.astro.ucla.edu/~sfurlane/docs',
    'elec_interp.tar.gz', 
    'read_FJS10.py'],
 'starburst99': ['http://www.stsci.edu/science/starburst99/data',
    'data.tar.gz', 
    None],                        
 'hm12': ['http://www.ucolick.org/~pmadau/CUBA/Media',
    'UVB.out', 
    'emissivity.out', 
    None],
 'bpass_v1': ['http://bpass.auckland.ac.nz/2/files'] + \
    ['sed_bpass_z%s_tar.gz' % Z for Z in ['001', '004', '008', '020', '040']] + \
    [None],

}

print '\n'
os.chdir('input')

for direc in aux_data:
    if not os.path.exists(direc):
        os.mkdir(direc)
    
    os.chdir(direc)
    
    web = aux_data[direc][0]
    for fn in aux_data[direc][1:-1]:
    
        if os.path.exists(fn):
            if fresh:
                os.remove(fn)
            else:
                continue
    
        print "Downloading %s/%s..." % (web, fn)
        urllib.urlretrieve('%s/%s' % (web, fn), fn)
        
        if not re.search(fn, 'tar'):
            continue
            
        tar = tarfile.open(fn)
        tar.extractall()
        tar.close()
    
    # Run a script [optional]
    if aux_data[direc][-1] is not None:
        execfile(aux_data[direc][-1])
    
    os.chdir('..')

# Go back down to the root level, otherwise the user will get slightly 
# incorrect instructions for how to set the ARES environment variable
os.chdir('..')

ARES_env = os.getenv('ARES')
cwd = os.getcwd()

##
# TELL PEOPLE TO SET ENVIRONMENT VARIABLE
##
if not ARES_env:

    import re    
    shell = os.getenv('SHELL')

    print "\n"
    print "#"*78
    print "It would be in your best interest to set an environment variable"
    print "pointing to this directory.\n"

    if shell:    

        if re.search('bash', shell):
            print "Looks like you're using bash, so add the following to your .bashrc:"
            print "\n    export ARES=%s" % cwd
        elif re.search('csh', shell):
            print "Looks like you're using csh, so add the following to your .cshrc:"
            print "\n    setenv ARES %s" % cwd

    print "\nGood luck!"
    print "#"*78        
    print "\n"

# Print a warning if there's already an environment variable but it's pointing
# somewhere other than the current directory
elif ARES_env != cwd:
    
    print "\n"
    print "#"*78
    print "It looks like you've already got an ARES environment variable set",
    print "but it's \npointing to a different directory:"
    print "\n    ARES=%s" % ARES_env
    
    print "\nHowever, we're currently in %s.\n" % cwd
    
    print "Is this a different ares install (might not cause problems),",
    print "or perhaps just"
    print "a typo in your environment variable?"
    
    print "#"*78        
    print "\n"

