exoplanet | case studies
========================

This is where the case studies for [exoplanet](https://github.com/exoplanet-dev/exoplanet) live!

Contributing a case study
-------------------------

First: thank you for considering contributing a case study!

Take a look at the [exoplanet developer docs](https://docs.exoplanet.codes/en/latest/user/dev/) for information about how to set up your system.
Then, you can fork this repo and add your case study as a notebook to the `docs/notebooks` directory.

A few things to double check:

1. The format should follow one of the existing case studies. In particular, the first two cells should be `%matplotlib inline` and `%run notebook_setup`, followed by a markdown cell describing the case study.
2. Make sure that you **clear the outputs of your notebook before committing**. Otherwise this repo will get unruly! I normally use [nbstripout](https://github.com/kynan/nbstripout) to make sure that this always happens.

Finally, please edit the `docs/conf.py` file to include the info about your case study in the `case_studies` variable.
If it's not clear what to do there, we can discuss it in the pull request thread.
