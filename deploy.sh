export ARES=$TRAVIS_BUILD_DIR    
python setup.py install
python remote.py minimal basic
cd tests
echo "backend : Agg" > $HOME/matplotlibrc
export MATPLOTLIBRC=$HOME
py.test -v test_*.py

