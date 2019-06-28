#!/bin/bash
################################################################################
# This script will execute run_CosmoRec.py for a bunch of different cosmologies.
# Pass three arguments to this script:
# (i) path to your CosmoRec executable (not including executable itself)
# (ii) Planck likelihood to use, e.g.,
#      plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing_4
# (iii) Number of models to run in total.
################################################################################

CR=$1
like=$2
num=$3
start=${4:-0}

ctr=$start
while [ $ctr -le $3 ]
do
  python run_CosmoRec.py $1 $2 $ctr
  ((ctr++))
done

