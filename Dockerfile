FROM continuumio/miniconda3

RUN mkdir -p /usr/src/app
COPY . /usr/src/app/
WORKDIR /usr/src/app

RUN apt-get update && \
    apt-get install -y gcc g++ libstdc++6 swig git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/envs/pycest/bin:$PATH
#RUN conda env create -f env.yaml
RUN conda create -n pycest python==3.9 --yes
RUN echo "source activate pycest" > ~/.bashrc
SHELL ["conda", "run", "-n", "pycest", "/bin/bash", "-c"]


WORKDIR /usr/src/app/open-py-cest-mrf
RUN pip install -e .
WORKDIR /usr/src/app

RUN conda install nvidia/label/cuda-11.8.0::cuda-toolkit
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install opencv-python
