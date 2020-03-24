export ARES=$TRAVIS_BUILD_DIR    
python setup.py install
python remote.py minimal basic
echo "backend : Agg" > $HOME/matplotlibrc
export MATPLOTLIBRC=$HOME
py.test -v --cov=ares tests/test*.py

