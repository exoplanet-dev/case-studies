#!/bin/bash

set -e

__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
conda activate base

git checkout master
git pull origin master
git submodule update

conda env update --prefix ./env -f exoplanet/environment.yml --prune
conda activate ./env
python -m pip install -U -r requirements-notebooks.txt

CACHEDIR=`pwd`/theano_cache
rm -rf $CACHEDIR
export THEANO_FLAGS=base_compiledir=$CACHEDIR

git branch -D auto_notebooks || true
git checkout -b auto_notebooks master

cd docs
conda env export > auto_environment.yml

python run_notebooks.py $*

cp notebooks/notebook_setup.py _static/notebooks/notebook_setup.py
git add _static/notebooks/notebook_setup.py
git add _static/notebooks/*.ipynb

git -c user.name='exoplanetbot' -c user.email='exoplanetbot' commit -am "updating notebooks [ci skip]"
git push -q -f https://dfm:`cat .github_api_key`@github.com/dfm/exoplanet-docs.git auto_notebooks

cd ..
git checkout master
conda deactivate

mail -s "autoexoplanet finished" "foreman.mackey@gmail.com" <<EOF
run_notebooks finished running
EOF
