# 파이썬 3.9 & 3.10
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel 

WORKDIR /step1
EXPOSE 8890

# CUDA 경로 및 환경 변수 추가
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
# ENV PATH /opt/conda/envs/ldm/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda-11.6
ENV PATH=$CUDA_HOME/include:$PATH


COPY environment.yaml environment.yaml

# git wget curl bzip2설치
RUN apt-get update --fix-missing && apt-get install -y git wget curl bzip2 && apt-get clean && rm -rf /var/lib/apt/lists/*

# conda install
RUN /bin/bash -c "git clone https://github.com/Stability-AI/stablediffusion.git"
RUN /bin/bash -c "cd stablediffusion; conda update -n base conda; conda clean --all; conda env create -f environment.yaml"
RUN /bin/bash -c "conda init bash"
RUN /bin/bash -c "source activate ldm"
RUN /bin/bash -c "pip install jupyterlab"
# RUN /bin/bash -c "conda install -c conda-forge gcc"
# RUN /bin/bash -c "conda install -c conda-forge gxx_linux-64==9.5.0"
RUN /bin/bash -c "pwd"
RUN /bin/bash -c "cd .."
RUN /bin/bash -c "git clone https://github.com/facebookresearch/xformers.git"
RUN /bin/bash -c "cd xformers; git submodule update --init --recursive;pip install -r requirements.txt; pip install -e .; cd ../stablediffusion"
# RUN conda clean --all
# RUN conda env create -f environment.yaml

ENTRYPOINT [ "jupyter", "lab", "--ip=0.0.0.0", "--port=8890", "--allow-root", "--no-browser" ]
