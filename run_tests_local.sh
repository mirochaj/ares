#!/bin/bash

# Can never remember all the flags
pytest --cov-config=.coveragerc --cov=ares --cov-report=html -v tests/*.py

rm -f test_*.pkl test_*.txt test_*.hdf5 hmf*.pkl hmf*.hdf5
