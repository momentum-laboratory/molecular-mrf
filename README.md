# Quantitative CEST-MRF starter pack
This repository contains a CEST-MRF signal simulator and an exemplary guide on how to perform CEST-MRF step by step. The detailed protocol for CEST-MRF experiments can be found in the Nature Protocols paper. [Nature paper](https://github.com/operlman/cest-mrf)

The CEST-MRF package (open-py-cest-mrf) facilitates rapid generation of CEST-MRF dictionaries through parallel execution. It was authored by Nikita Vladimirov with contributions from Or Perlman and is inspired by and builds upon the following works:
1. [pypulseq-cest](https://github.com/KerstinKaspar/pypulseq-cest/blob/main/pypulseq_cest/parser.py)
2. [CEST-MRF](https://github.com/operlman/cest-mrf)
3. [pulseq-cest Matlab](https://github.com/kherz/pulseq-cest/tree/master)

## Prerequisites:
- Python >= 3.9
- Git

## Installation

To install, execute the following command in the terminal (ideally within a clean conda environment or similar):
```
cd open-py-cest-mrf
pip install -e .
```
The `-e` argument enables editable mode installation, allowing you to modify the source code directly in your local copy.

### Docker
Alternatively, you can use Docker. Build the image with the provided Dockerfile using the following command:
```
docker build -t pycest-starter .
```
Or, use the prebuilt image from Docker Hub (not yet available):
```
docker pull vnikale/pycest-starter
docker run vnikale/pycest-starter
```


## Structure:
`open-py-cest-mrf` is the main package for the CEST MRF simulator. For further details, refer to the README.md within the package. 

Each example folder contains a pair of .ipynb files and .py files. To replicate the figures from the Nature Protocols, follow the steps in the .ipynb files. The Nature Protocol paper provides additional information on each example. Some folders may contain a `visualization.ipynb` file that recreates figures from the paper. [Nature paper]

To run .py examples you have to treat everything as a package, e.g.:
```
python -m dot_prod_example.dp_preclinical
python -m dot_prod_example.dp_clinical
etc
```

`dot_prod_example` demonstrates how to perform dot product matching on L-arginine phantom data, including .seq, .yaml, and MRF dictionary generation. This example should be run first before proceeding to `deep_reco_example` as it uses DP-generated masks. It includes `dp_clinical.ipynb` (L-arg 3T data) and `dp_preclinical.ipynb (L-arg 9.4T data)`, along with corresponding .py files. 

`deep_reco_example` showcases quantification using deep learning, including a .yaml file, MRF dictionary generation, and network training and testing. It includes `deep_reco_clinical.ipynb` (L-arg 3T data) and `deep_reco_preclinical.ipynb ` (L-arg 9.4T data), with corresponding .py files forthcoming. 

`sequential_nn_example` offers an example of implementing sequential neural-network CEST MRF quantification by utilizing additional information such as T1/T2 maps. This includes .seq, .yaml files, dictionary generation, and network training and testing on Iohexol data and MT mouse data. It features `sequential_example_iohexenol.ipynb` (Iohexol 4.7T data) and `sequential_example_iohexenol.ipynb` (Mouse MR 9.4T data) along with corresponding .py files.

`human_example` is an example of neural-network quantification inference on clinical human data. It includes `drone_example.ipynb` along with corresponding .py file. 

`metrics_example` demonstrates the use of Monte Carlo and Cramer Rao Bound for assessing the quality of MRF schedules. It includes `monte_carlo.ipynb` and `crlb.ipynb`, along with corresponding .py files.