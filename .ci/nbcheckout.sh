#!/bin/bash

for f in docs/notebooks/*.ipynb
do
    val=$(cat $f | nbstripout -t | diff $f -)
    if [[ ! -z "$val" ]]
    then
        echo "ipynb has output still: $f"
        exit 1
    fi
done