#!/usr/bin/env python

import os, urllib, shutil

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

ares_link = 'https://bitbucket.org/mirochaj/ares'

ares_packages = \
    ['ares', 'ares.analysis', 'ares.simulations', 'ares.populations',
     'ares.util', 'ares.solvers', 'ares.static', 'ares.sources', 
     'ares.physics', 'ares.inference', 'ares.phenom']

setup(name='ares',
      version='0.1',
      description='Accelerated Reionization Era Simulations',
      author='Jordan Mirocha',
      author_email='mirochaj@gmail.com',
      url=ares_link,
      packages=ares_packages,
      #entry_points={'console_scripts': [
      #'ares = ares.util.CommandLine:run_main',
      #]},
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
s99_seds = 'http://www.stsci.edu/science/starburst99/data'
hm12_cuvb = 'http://www.ucolick.org/~pmadau/CUBA/Media'
bpass_seds = 'http://bpass.auckland.ac.nz/2/files'

# Filenames
fn_hmf = 'hmf_ST_logM_1200_4-16_z_1141_3-60.pkl'
fn_ics_np = 'initial_conditions.npz'
fn_tau = 'optical_depth_H_400x1616_z_10-50_logE_2-4.7.pkl'
fn_tau2 = 'optical_depth_He_400x1616_z_10-50_logE_2-4.7.pkl'
fn_elec = 'elec_interp.tar.gz'
fn_cuvb = 'UVB.out'
fn_emiss = 'emissivity.out'

print '\n'

# First, secondary electron data from Furlanetto & Stoever (2010)
if not os.path.exists('secondary_electrons'):
    os.mkdir('secondary_electrons')
    
if not os.path.exists('secondary_electrons/%s' % fn_elec):
    os.chdir('secondary_electrons')
    print "Downloading %s/%s..." % (sfurlane_xray, fn_elec)
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

# Next, starburst 99 dataset from Leitherer et al. 1999 (original paper)
if not os.path.exists('starburst99'):
    os.mkdir('starburst99')

if not os.path.exists('starburst99/data.tar.gz'):
    os.chdir('starburst99')
    print "Downloading %s/data.tar.gz..." % s99_seds
    urllib.urlretrieve('%s/data.tar.gz' % s99_seds, 'data.tar.gz')
    os.chdir('..')
    
if not os.path.exists('starburst99/fig1a.dat'):
    os.chdir('starburst99')
    
    import tarfile
    tar = tarfile.open('data.tar.gz')
    tar.extractall()
    tar.close()
    
    for fn in os.listdir('data'):
        shutil.move('data/%s' % fn, '.')
    os.rmdir('data')
    
    os.chdir('..')
    
# Next, BPASS dataset from Eldridge et al. 2009
if not os.path.exists('bpass'):
    os.mkdir('bpass')

for Z in ['001', '004', '008', '020', '040']:
    fn = 'sed_bpass_z%s_tar.gz' % Z
    if not os.path.exists('bpass/%s' % fn):
        os.chdir('bpass')
        print "Downloading %s/%s..." % (bpass_seds, fn)
        urllib.urlretrieve('%s/%s' % (bpass_seds, fn), fn)
        os.chdir('..')
    
    if not os.path.exists('bpass/SEDS/sed.bpass.constant.cloudy.bin.z%s' % Z):
        os.chdir('bpass')
    
        import tarfile
        tar = tarfile.open(fn)
        tar.extractall()
        tar.close()
    
        os.chdir('..')    
    
# Next, Haardt & Madau (2012) data
if not os.path.exists('hm12'):
    os.mkdir('hm12')    

if not os.path.exists('hm12/%s' % fn_cuvb):
    os.chdir('hm12')
    print "Downloading %s/%s..." % (hm12_cuvb, fn_cuvb)
    urllib.urlretrieve('%s/%s' % (hm12_cuvb, fn_cuvb), fn_cuvb)
    os.chdir('..')
    
if not os.path.exists('hm12/%s' % fn_emiss):
    os.chdir('hm12')
    print "Downloading %s/%s..." % (hm12_cuvb, fn_emiss)
    urllib.urlretrieve('%s/%s' % (hm12_cuvb, fn_emiss), fn_emiss)
    os.chdir('..')

# Now, files from bitbucket (HMF, optical depth, etc.)
if not os.path.exists('hmf'):
    os.mkdir('hmf')
        
if not os.path.exists('hmf/%s' % fn_hmf):
    os.chdir('hmf')
    print "Downloading %s/%s..." % (bitbucket_DL, fn_hmf)
    urllib.urlretrieve('%s/%s' % (bitbucket_DL, fn_hmf), fn_hmf)
    os.chdir('..')

if not os.path.exists('inits'):
    os.mkdir('inits')

if not os.path.exists('inits'):
    os.mkdir('inits')
    
if not os.path.exists('inits/%s' % fn_ics_np):
    os.chdir('inits')
    print "Downloading %s/%s..." % (bitbucket_DL, fn_ics_np)
    urllib.urlretrieve('%s/%s' % (bitbucket_DL, fn_ics_np), fn_ics_np)
    os.chdir('..')  
      
if not os.path.exists('optical_depth'):
    os.mkdir('optical_depth')

if not os.path.exists('optical_depth/%s' % fn_tau):
    os.chdir('optical_depth')
    print "Downloading %s/%s..." % (bitbucket_DL, fn_tau)
    urllib.urlretrieve('%s/%s' % (bitbucket_DL, fn_tau), fn_tau)
    os.chdir('..')

if not os.path.exists('optical_depth/%s' % fn_tau2):    
    os.chdir('optical_depth')
    print "Downloading %s/%s..." % (bitbucket_DL, fn_tau2)
    urllib.urlretrieve('%s/%s' % (bitbucket_DL, fn_tau2), fn_tau2)
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

