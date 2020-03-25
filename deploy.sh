export ARES=$TRAVIS_BUILD_DIR    
python setup.py install
python remote.py minimal basic
echo "backend : Agg" > $HOME/matplotlibrc
export MATPLOTLIBRC=$HOME
pytest tests/test*.py -v --cov=ares 

