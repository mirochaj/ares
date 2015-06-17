#!/usr/bin/env python

import os, urllib

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

ares_link = 'https://bitbucket.org/mirochaj/ares'

ares_packages = \
    ['ares', 'ares.analysis', 'ares.simulations', 'ares.populations',
     'ares.util', 'ares.solvers', 'ares.static', 'ares.sources', 
     'ares.physics', 'ares.inference']

setup(name='ares',
      version='0.1',
      description='Accelerated Reionization Era Simulations',
      author='Jordan Mirocha',
      author_email='mirochaj@gmail.com',
      url=ares_link,
      packages=ares_packages,
     )
          
# Try to set up $HOME/.ares: this won't work on WINDOWS at the moment
HOME = os.getenv('HOME')
if not os.path.exists('%s/.ares' % HOME):
    try:
        os.mkdir('%s/.ares' % HOME)
    except:
        pass
 
if not os.path.exists('%s/.ares/defaults.py' % HOME):
    try:
        f = open('%s/.ares/defaults.py' % HOME, 'w')
        print >> f, "pf = {}"
        f.close()
    except:
        pass

if not os.path.exists('%s/.ares/labels.py' % HOME):   
    try:
        f = open('%s/.ares/labels.py' % HOME, 'w')
        print >> f, "pf = {}"
        f.close()
    except:
        pass
    
# Setup input directory, in preparation for file download
if not os.path.exists('input'):
    os.mkdir('input')
    
os.chdir('input')

##
# DOWNLOAD SOME FILES
##

# Link prefixes we need
bitbucket_DL = 'https://bitbucket.org/mirochaj/ares/downloads'
sfurlane_xray = 'http://www.astro.ucla.edu/~sfurlane/docs'

# Filenames
fn_hmf = 'hmf_ST_logM_1200_4-16_z_1121_4-60.pkl'
fn_ics_np = 'initial_conditions.npz'
fn_tau = 'optical_depth_H_400x1616_z_10-50_logE_2-4.7.pkl'
fn_tau2 = 'optical_depth_He_400x1616_z_10-50_logE_2-4.7.pkl'
fn_elec = 'elec_interp.tar.gz'

# First, secondary electron data from Furlanetto & Stoever (2010)
if not os.path.exists('secondary_electrons'):
    os.mkdir('secondary_electrons')
    
if not os.path.exists('secondary_electrons/%s' % fn_elec):
    os.chdir('secondary_electrons')
    print "\nDownloading %s/%s..." % (sfurlane_xray, fn_elec)
    urllib.urlretrieve('%s/%s' % (sfurlane_xray, fn_elec), fn_elec)
    os.chdir('..')

if not os.path.exists('secondary_electrons/secondary_electron_data.pkl'):
    os.chdir('secondary_electrons')
    import tarfile
    tar = tarfile.open(fn_elec)
    tar.extractall()
    tar.close()
    
    # Convert data to more convenient format
    execfile('read_FJS10.py')
    
    os.chdir('..')

# Now, files from bitbucket (HMF, optical depth, etc.)
if not os.path.exists('hmf'):
    os.mkdir('hmf')
        
if not os.path.exists('hmf/%s' % fn_hmf):
    os.chdir('hmf')
    print "\nDownloading %s/%s..." % (bitbucket_DL, fn_hmf)
    urllib.urlretrieve('%s/%s' % (bitbucket_DL, fn_hmf), fn_hmf)
    os.chdir('..')

if not os.path.exists('inits'):
    os.mkdir('inits')

if not os.path.exists('inits'):
    os.mkdir('inits')
    
if not os.path.exists('inits/%s' % fn_ics_np):
    os.chdir('inits')
    print "\nDownloading %s/%s..." % (bitbucket_DL, fn_ics_np)
    urllib.urlretrieve('%s/%s' % (bitbucket_DL, fn_ics_np), fn_ics_np)
    os.chdir('..')  
      
if not os.path.exists('optical_depth'):
    os.mkdir('optical_depth')

if not os.path.exists('optical_depth/%s' % fn_tau):
    os.chdir('optical_depth')
    print "\nDownloading %s/%s..." % (bitbucket_DL, fn_tau)
    urllib.urlretrieve('%s/%s' % (bitbucket_DL, fn_tau), fn_tau)
    os.chdir('..')

if not os.path.exists('optical_depth/%s' % fn_tau2):    
    os.chdir('optical_depth')
    print "\nDownloading %s/%s..." % (bitbucket_DL, fn_tau2)
    urllib.urlretrieve('%s/%s' % (bitbucket_DL, fn_tau2), fn_tau2)
    os.chdir('..')

# Go back down to the root level, otherwise the user will get slightly 
# incorrect instructions for how to set the ARES environment variable
os.chdir('..')    

##
# TELL PEOPLE TO SET ENVIRONMENT VARIABLE
##
if not os.getenv('ARES'):
    
    import re
    
    cwd = os.getcwd()
    shell = os.getenv('SHELL')
    
    print "\n"
    print "#"*92
    print "It would be in your best interest to set an environment variable",
    print "pointing to this directory."
        
    if shell:    
        
        if re.search('bash', shell):
            print "Looks like you're using bash, so add the following to your .bashrc:"
            print "\n    export ARES=%s" % cwd
        elif re.search('csh', shell):
            print "Looks like you're using csh, so add the following to your .cshrc:"
            print "\n    setenv ARES %s" % cwd

    print "#"*92        
    print "\n"
