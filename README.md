# CEST-MRF
This code is for quickly generating CEST-MRF dictionaries using parallel execution.
Written by Nikita Vladimirov, some modifications by Or Perlman and built on and inspired by:
1. [pypulseq-cest](https://github.com/KerstinKaspar/pypulseq-cest/blob/main/pypulseq_cest/parser.py) 
2. [CEST-MRF](https://github.com/operlman/cest-mrf)
3. [pulseq-cest Matlab](https://github.com/kherz/pulseq-cest/tree/master) 


## Prerequisites:
- Python >= 3.9
- git

## Installation
Run the following command inside the terminal (preferably in a clean conda env or similar):
`cd open-py-cest-mrf`
`pip install -e .`

The argument `-e` allows installation in editable mode, so you can change sources inside your local copy.

## Structure:
`open-py-cest-mrf` - main CEST MRF simulator package, refer to the README.md inside for further details

Each example folder contains pairs of .ipynb files and .py which you can use. Follow the .ipynb files step-by-step to repeat Nature Protocols figures. Additional details on each example can be found in Nature Protocol paper.

`dot_prod_example` - example of performing dot. product matching on the L-arginine phantom data. You have to run it first before moving to `deep_reco_example` since it uses DP-generated masks

`deep_reco_example` - example of quantification using deep learning

`sequential_nn_example` - example of implementation of sequential neural-network CEST MRF quantification, i.e. by using additional information such as T1/T2 maps. Examples are provided on Iohexol data and MT mouse data.

`human_example` - example of neural-network quantification inference on the clinical human data

`metrics_example` - example of using Monte Carlo and Cramer Rao Bound for MRF schedule quality assessment.
