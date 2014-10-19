#!/usr/bin/env python

import os, sys, urllib
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
          
# Try to set up $HOME/.ares
HOME = os.getenv('HOME')
if not os.path.exists('%s/.ares' % HOME):
    try:
        os.mkdir('%s/.ares' % HOME)
        f = open('%s/.ares/defaults.py' % HOME, 'w')
        print >> f, "pf = {}"
        f.close()
    except:
        pass
    
cwd = os.getcwd()    
    
# Download some files
if not os.path.exists('input'):
    os.mkdir('input')
    
os.chdir('input')

##
# DOWNLOAD SOME FILES
##
bitbucket_DL = 'https://bitbucket.org/mirochaj/glorb/downloads'
fn_hmf = 'hmf_PS_logM_240_4-16_z_1521_4-80.pkl'
fn_ics_h5 = 'initial_conditions.hdf5'
fn_ics_np = 'initial_conditions.npz'
fn_tau = 'optical_depth_H_400x1884_z_10-40_logE_2-4.7.hdf5'
fn_elec = 'secondary_electron_data.hdf5'

if not os.path.exists(fn_elec):
    print "\nDownloading %s/%s..." % (bitbucket_DL, fn_elec)
    urllib.urlretrieve('%s/%s' % (bitbucket_DL, fn_elec), fn_elec)

if not os.path.exists('hmf'):
    os.mkdir('hmf')
        
if not os.path.exists('hmf/%s' % fn_hmf):
    os.chdir('hmf')
    print "\nDownloading %s/%s..." % (bitbucket_DL, fn_hmf)
    urllib.urlretrieve('%s/%s' % (bitbucket_DL, fn_hmf), fn_hmf)
    os.chdir('..')

if not os.path.exists('inits'):
    os.mkdir('inits')

if not os.path.exists('inits/%s' % fn_ics_h5):
    os.chdir('inits')
    print "\nDownloading %s/%s..." % (bitbucket_DL, fn_ics_h5)
    urllib.urlretrieve('%s/%s' % (bitbucket_DL, fn_ics_h5), fn_ics_h5)
    os.chdir('..')

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

##
# TELL PEOPLE TO SET ENVIRONMENT VARIABLE
##
if not os.getenv('ARES'):
    
    import re
    
    shell = os.getenv('SHELL')
    
    print "\nIt would be in your best interest to set an environment variable",
    print "pointing to this directory."
        
    if re.search('bash', shell):
        print "Looks like you're using bash, so add the following to your .bashrc:"
        print "\n    export ARES=%s" % cwd
    elif re.search('csh', shell):
        print "Looks like you're using csh, so add the following to your .cshrc:"
        print "\n    setenv ARES %s" % cwd
        

