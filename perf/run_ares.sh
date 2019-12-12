#!/bin/bash

scriptname=$1
Niter=$2
fnout=${scriptname/py/eps}

python -m cProfile -o output.pstats $scriptname $Niter
gprof2dot -f pstats output.pstats | dot -Teps -o $fnout

