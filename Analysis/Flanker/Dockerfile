FROM ubuntu:16.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt-get install -y ca-certificates wget apt-transport-https wget git clang-3.8 lldb-3.8\
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub \
    && apt-key add 7fa2af80.pub \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604-keyring.gpg \
    && apt-key add cuda-ubuntu1604-keyring.gpg \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb \
    && dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb \
    && apt-get update \
    && apt-get install -y cuda-9-1

RUN wget  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda

ENV PATH="/usr/local/cuda-9.1/bin:/opt/conda/bin:${PATH}"

RUN conda config --set auto_activate_base false \
    && conda create -c rmg -n supreme_env scoop python=2.7 numpy=1.16 pandas=0.22 joblib=0.13 pip=18 \
    && conda info -e

SHELL ["conda", "run", "-n", "supreme_env", "/bin/bash", "-c"]    

RUN pip install --upgrade "pip < 21.0" \
    && pip install mako==1.0.7 pytools==2018.1 pytest==3.4.2 decorator==4.2.1 appdirs==1.4.3 setuptools==38.6.1 scipy==1.01 matplotlib==2.2.2 \
    && git clone https://github.com/compmem/RunDEMC.git \
    && cd RunDEMC \
    && git checkout f2d07db \
    && python setup.py install 

RUN wget https://files.pythonhosted.org/packages/58/33/cced4891eddd1a3ac561ff99081019fddc7838a07cace272c941e3c2f915/pycuda-2018.1.1.tar.gz \
    && tar -xzvf pycuda-2018.1.1.tar.gz \
    && cd pycuda-2018.1.1 \
    && python configure.py --cuda-root=/usr/local/cuda-9.1/ \
    && make \
    && make install


