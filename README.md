# Quantitative Molecular Imaging using Deep Magnetic Resonance Fingerprinting 
This repository contains a CEST-MRF (Bloch-McConnell-based) signal simulator and an extensive step-by-step guide for semisolid MT and CEST MRF experiments. This repository is part of a detailed protocol paper entitled "Quantitative Molecular Imaging using Deep Magnetic Resonance Fingerprinting" (under review at Nat. Prot.). 

[![DOI](https://zenodo.org/badge/770832769.svg)](https://doi.org/10.5281/zenodo.14211516)


## The key publications associated with this protocol are:
1. Perlman, O., Ito, H., Herz, K. et al. Quantitative imaging of apoptosis following oncolytic virotherapy by magnetic resonance fingerprinting aided by deep learning. Nat. Biomed. Eng 6, 648–657 (2022). (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9091056/)

2. Cohen, O, Yu, VY, Tringale, KR, et al. CEST MR fingerprinting (CEST-MRF) for brain tumor quantification using EPI readout and deep learning reconstruction. Magn Reson Med. 89, 233-249 (2023). (https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.29448)

3. Kang, B, Kim, B, Schär, M, Park, H, Heo, H-Y. Unsupervised learning for magnetization transfer contrast MR fingerprinting: Application to CEST and nuclear Overhauser enhancement imaging. Magn Reson Med. 85, 2040–2054 (2021). (https://doi.org/10.1002/mrm.28573)

 
The CEST-MRF package (open-py-cest-mrf) facilitates rapid generation of CEST-MRF dictionaries through parallel execution. It was developed by Nikita Vladimirov and is inspired by and builds upon the following works:
1. [pypulseq-cest](https://github.com/KerstinKaspar/pypulseq-cest/blob/main/pypulseq_cest/parser.py)
2. [CEST-MRF](https://github.com/operlman/cest-mrf)
3. [pulseq-cest Matlab](https://github.com/kherz/pulseq-cest/tree/master)

The repository includes data, sample code, trained neural networks, and pulse sequences provided by several research groups and scientists, including Christian Farrar, Ouri Cohen, Hye-Young Heo, Moritz Zaiss, Nikita Vladimirov, and Or Perlman.

## Prerequisites:
- Python version 3.9
- Git
- SWIG (can be installed using `sudo apt-get install swig` or [SWIG](https://www.swig.org/download.html))

## Installation

To install, execute the following command in the terminal (ideally within a clean conda environment or similar):
```
cd open-py-cest-mrf
pip install -e .
```
The `-e` argument enables editable mode installation, allowing you to modify the source code directly in your local copy. 

If you do not plan to use the Jupyter notebook examples, you must manually install PyTorch and OpenCV, as they are treated as external libraries and are not required by the simulator. You can install them using the following commands:
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install opencv-python
```
If you are using the deep_reco.ipynb Jupyter notebook examples, these libraries will be installed in the first cell of the notebook.


### A known bug and solution
It has been found that the current version can yield errors in Pulseq or NumPy on some machines. Please do the following (yes, 2 times):

```
pip uninstall numpy
pip uninstall numpy

pip uninstall pulseq
pip uninstall pulseq

pip install -e .
```


### Docker
Alternatively, you can use Docker, which has everything preinstalled. Build the image with the provided Dockerfile using the following command:
```
docker build -t pycest-starter .
```
Or, use the prebuilt image from Docker Hub (not yet available):
```
docker pull vnikale/pycest-starter
docker run vnikale/pycest-starter
```

## Structure:

`open-py-cest-mrf` is the main package for the CEST MRF simulator. For further details, refer to open-py-cest-mrf/README.md.
The repository includes 5 main exemplary case studies, arranged into separate folders. Each folder contains a Jupyther notebook (.ipynb) and a python script (.py) file. To reproduce the figures from the Nat. Prot. paper, follow the steps described in the .ipynb files. The protocol paper provides additional background information and explanations for each example. 

Some folders may contain a `visualization.ipynb` file that recreates the original figures from the paper, **once you run the main files** to create maps for the visualization.
Also, you can find a folder `expected_out` that contains expected figures and `.mat` with quantitative maps.

To run .py examples you have to treat the repo as a package, e.g. (run it in **the root folder**):
```
python -m dot_prod_example.preclinical
python -m dot_prod_example.clinical
```

`dot_prod_example` demonstrates how to perform dot product matching using L-arginine phantom data. It includes .seq, .yaml, and MRF dictionary generation. This example should be run first before proceeding to `deep_reco_example` as it uses DP-generated masks. It includes `clinical.ipynb` (L-arg 3T data) and `preclinical.ipynb (L-arg 9.4T data)`, along with the corresponding .py files. 

`deep_reco_example` showcases quantification using deep learning, including a .yaml file, MRF dictionary generation, network training, and inference. It includes `clinical.ipynb` (L-arg 3T data) and `preclinical.ipynb` (L-arg 9.4T data), as well as the corresponding .py files. 

`sequential_nn_example` offers an example of implementing sequential neural-network CEST MRF quantification by utilizing additional information such as T1/T2 maps. It includes .seq, .yaml files, dictionary generation, network training, and inference applied to Iohexol data and MT mouse data. It features `iohexenol.ipynb` (Iohexol 4.7T data) and `mouse.ipynb` (Mouse MT 9.4T data) along with corresponding .py files. 

`human_example` is an example of neural-network quantification inference on clinical human data. It includes `inference.ipynb` along with the corresponding .py file. 

`unsupervised_example` is an example of CNN-based inference, following unsupervised training on human data. It includes `cnn_inference.ipynb` along with the corresponding .py file. 

`metrics_example` demonstrates the use of the Monte Carlo and the Cramer Rao Bound for assessing the encoding capability of MRF acquisition schedules. It includes `monte_carlo.ipynb` and `crlb.ipynb`.

The `supplementary` folder contains additional materials: in the `phantom_cad` folder, there is an `.stl` file containing the CAD file for the phantom holder. In the `published_pulse_sequences` folder, you can find the pulse sequences mentioned in the Nature Protocols paper. For details, refer to the corresponding `README` file contained in the folder.
