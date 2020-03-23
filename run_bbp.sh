pip install numpy
pip install matplotlib
pip install scipy
pip install pytest
export ARES=$BITBUCKET_CLONE_DIR
cd $ARES          
python setup.py install
python remote.py minimal basic
cd tests
echo "backend : Agg" > $HOME/matplotlibrc
export MATPLOTLIBRC=$HOME
py.test -v test_*.py

