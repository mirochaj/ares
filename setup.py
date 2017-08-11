#!/usr/bin/env python

import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

ares_link = 'https://bitbucket.org/mirochaj/ares'

setup(name='ares',
      version='0.1',
      description='Accelerated Reionization Era Simulations',
      author='Jordan Mirocha',
      author_email='mirochaj@gmail.com',
      url=ares_link,
      packages=['ares', 'ares.analysis', 'ares.simulations',
       'ares.populations', 'ares.util', 'ares.solvers', 'ares.static',
       'ares.sources', 'ares.physics', 'ares.inference', 'ares.phenom'],
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

##
# TELL PEOPLE TO SET ENVIRONMENT VARIABLE
##

ARES_env = os.getenv('ARES')
cwd = os.getcwd()

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

