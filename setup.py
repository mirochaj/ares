#!/usr/bin/env python
from __future__ import print_function
import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='ares',
      version='0.1',
      description='Accelerated Reionization Era Simulations',
      author='Jordan Mirocha',
      author_email='mirochaj@gmail.com',
      url='https://github.com/mirochaj/ares',
      packages=['ares', 'ares.analysis', 'ares.data', 'ares.simulations', 'ares.obs',
       'ares.populations', 'ares.util', 'ares.solvers', 'ares.static',
       'ares.sources', 'ares.physics', 'ares.inference', 'ares.phenom'],
     )

# Try to set up $HOME/.ares
HOME = os.getenv('HOME')
if not os.path.exists('{!s}/.ares'.format(HOME)):
    try:
        os.mkdir('{!s}/.ares'.format(HOME))
    except:
        pass

# Create files for defaults and labels in HOME directory
for fn in ['defaults', 'labels']:
    if not os.path.exists('{0!s}/.ares/{1!s}.py'.format(HOME, fn)):
        try:
            f = open('{0!s}/.ares/{1!s}.py'.format(HOME, fn), 'w')
            print("pf = {}", file=f)
            f.close()
        except:
            pass
