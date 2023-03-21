#!/usr/bin/env python
import io
import os
from setuptools import find_namespace_packages, setup

# get readme
with io.open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

h5py_reqs = ["h5py"]
mcmc_reqs = ["shapely", "descartes"]
mpi_reqs = ["mpi4py"]
hmf_reqs = ["hmf", "camb"]
mpmath_reqs = ["mpmath"]
progressbar_reqs = ["progressbar2"]
doc_reqs = ["sphinx", "numpydoc", "nbsphinx"]
tests_reqs = ["pytest", "coverage"]
all_optional_reqs = (
    h5py_reqs
    + mcmc_reqs
    + mpi_reqs
    + hmf_reqs
    + mpmath_reqs
    + progressbar_reqs
    + doc_reqs
    + tests_reqs
)

setup_args = {
    "name": "ares",
    "description": "Accelerated Reionization Era Simulations",
    "long_description": readme,
    "long_description_content_type": "text/markdown",
    "author": "Jordan Mirocha",
    "author_email": "mirochaj@gmail.com",
    "url": "https://github.com/mirochaj/ares",
    "package_dir": {"ares": "ares"},
    "packages": find_namespace_packages(),
    "use_scm_version": True,
    "install_requires": [
        "numpy",
        "matplotlib",
        "scipy",
    ],
    "extras_require": {
        "h5py": h5py_reqs,
        "mcmc": mcmc_reqs,
        "mpi": mpi_reqs,
        "hmf": hmf_reqs,
        "mpmath": mpmath_reqs,
        "progressbar": progressbar_reqs,
        "tests": tests_reqs,
        "all": all_optional_reqs,
    },
    entry_points={"console_scripts": ["ares=ares.util.cli:main"]},
    "classifiers": [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Langauge :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    "keywords": "astronomy cosmology reionization",
}

if __name__ == "__main__":
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

    # run package setup
    setup(**setup_args)
