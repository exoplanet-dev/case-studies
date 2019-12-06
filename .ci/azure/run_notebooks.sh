#!/bin/bash

set -e

echo $CONDA_PATH
echo $OUTPUT_PATH

. $CONDA/etc/profile.d/conda.sh
conda activate $CONDA_PATH
conda env export > $OUTPUT_PATH/environment.yml
export THEANO_FLAGS=base_compiledir=`pwd`/theano_cache
cd docs
python run_notebooks.py notebooks/rv.ipynb

cp _static/notebooks/*.ipynb $OUTPUT_PATH
cp notebooks/notebook_setup.py $OUTPUT_PATH
cp notebook_errors.log $OUTPUT_PATH
