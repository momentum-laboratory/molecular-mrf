# Python CEST-MRF 
This code is for quickly generating CEST-MRF dictionaries using parallel execution.
Written by Nikita Vladimirov, some modifications by Or Perlman and built on and inspired by:
1. [pypulseq-cest](https://github.com/KerstinKaspar/pypulseq-cest/blob/main/pypulseq_cest/parser.py) 
2. [CEST-MRF](https://github.com/operlman/cest-mrf)
3. [pulseq-cest Matlab](https://github.com/kherz/pulseq-cest/tree/master) 

## Prerequisites:
- Python = 3.9
- git
- SWIG

## Installation
Run the following command inside the terminal (preferably in a clean conda env or similar):

`pip install -e .`

The argument `-e` allows installation in editable mode, so you can change sources inside your local copy.

## Structure:
- `cest_mrf/simulation/*` - this folder contains the sources of the simulation function and SimulationParametersMRF class.
- `cest_mrf/sequence/*` - this folder contains the sources of functions that create sequences and save them in `.seq` files
- `cest_mrf/dictionary/*` - this folder contains the sources of functions for preparing the dictionary and its generation.
- `cest_mrf/sim_lib/*` - this folder contains the sources of the latest version of  [pulseq-cest Matlab](https://github.com/kherz/pulseq-cest/tree/master) 
with an interface for using it in Python (SWIG). In the folder, you can find `readme.md`, which contains a guide to rebuild the library if you need to.
