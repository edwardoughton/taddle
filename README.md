Telecoms Analytics for Demand using Deep Learning (taddle)
===========================================

Description
===========
**taddle** is a predictive codebase for predicting telecom demand in areas of poor data
availability.


Setup and configuration
=======================

All code for **taddle** is written in Python (Python>=3.5) and has a number of dependencies.
See `requirements.txt` for a full list.

Using conda
-----------

The recommended installation method is to use [conda](http://conda.pydata.org/miniconda.html),
which handles packages and virtual environments,
along with the `conda-forge` channel which has a host of pre-built libraries and packages.

Create a conda environment called `taddle`:

    conda create --name taddle python=3.5

Activate it (run each time you switch projects)::

    activate taddle

First, install required packages including `pandas`,  `requests` and `rasterstats`:

    conda install pandas requests rasterstats

To visualize the results, install `matplotlib` and `seaborn`:

    conda install matplotlib seaborn

And then run:

    python vis/vis.py

Background and funding
======================

**taddle** has been funded by UK EPSRC via the Infrastructure Transitions Research
Consortium (EP/N017064/1) and a subsequent EPSRC Impact Accelerator Award.

Contributors
============
- Edward J. Oughton (University of Oxford)
- Jatin Mathur (University of Illinois)

Quick Start
===========
- download:
    - gadm data
    - lsms data
    - worldpop
- run scripts/get_data.py
- run scripts/grid.py
- run scripts/preprocess.py
- run scripts/predict.py
- run vis/vis.py
