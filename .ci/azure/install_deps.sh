#!/bin/bash

set -e

. $CONDA/etc/profile.d/conda.sh
conda env update --prefix $CONDA_PATH -f exoplanet/environment.yml -q --prune
conda activate $CONDA_PATH
python -m pip install -U -r requirements-notebooks.txt
python -m pip install -e ./exoplanet
