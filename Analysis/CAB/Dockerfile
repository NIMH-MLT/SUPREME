FROM ubuntu:16.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt-get install -y ca-certificates wget apt-transport-https wget git 

RUN wget  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda

ENV PATH="/usr/local/cuda-9.1/bin:/opt/conda/bin:${PATH}"

RUN conda config --set auto_activate_base false \
    && conda create -c bioconda -n supreme_env python=3.8 scoop numpy=1.18.1 pandas=1.0.1 scipy=1.4.1 matplotlib joblib jupyter notebook\
    && conda info -e

SHELL ["conda", "run", "-n", "supreme_env", "/bin/bash", "-c"]    

RUN git clone https://github.com/compmem/RunDEMC.git \
    && cd RunDEMC \
    && git checkout f2d07db \
    && python setup.py install 


COPY . /CAB
WORKDIR /CAB
ENV PYTHONWARNINGS=ignore
ENTRYPOINT ["/opt/conda/bin/conda", "run", "-n", "supreme_env", "--no-capture-output", "python", "-W", "ignore", "./cab_SUPREME.py"]

