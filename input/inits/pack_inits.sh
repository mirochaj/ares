#!/bin/bash

me=$(whoami)

echo ''
echo '############################### inits ###############################'

if [ $me = 'jordanmirocha' ]
then
  echo \# Hey, J.M., right this way.	
else
  printf '# Are you sure you want to proceed [yes/no]: '
  read -r areyousure 
  
  if [ $areyousure = 'yes' ]
  then
    echo \# OK, hope you know what you are doing.
  else
    exit 1
  fi	
fi 

printf '# Please specify the path (up to but not including) your CosmoRec executable: '
read -r cosmorec

echo \# Generating initial conditions with $cosmorec

python generate_inits_tables.py $cosmorec


tar -czvf inits.tar.gz inits_planck_TTTEEE_lowl_lowE_best.txt

# Copy to dropbox
echo Created tarball inits.tar.gz.

# Copy to dropbox
FILE=$DROPBOX/ares
if [ -d "$FILE" ]
then
  :
else 
  mkdir $FILE
  echo "Created $FILE."
fi

if [ -d "$FILE/input" ]
then
  :
else 
  mkdir $FILE/input
  echo "Created $FILE/input."
fi

if [ -d "$FILE/input/inits" ]
then
  :
else 
  mkdir $FILE/input/inits	
  echo "Created $FILE/input/inits."
fi

cp inits.tar.gz $FILE/input/inits
echo Copied inits.tar.gz to $FILE/input/inits

echo '#####################################################################'
echo ''