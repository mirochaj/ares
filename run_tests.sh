#!/bin/bash

# Can never remember all the flags
pytest --cov-config=.coveragerc --cov=ares --cov-report=html -v tests/*.py

