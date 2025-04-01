# Use an official conda base image from ContinuumIO
# FROM continuumio/miniconda3:latest
FROM nvcr.io/nvidia/pytorch:24.07-py3
LABEL author="SonND"

# update pakages
# RUN apt-get update
# RUN apt-get -y install gcc

# Set environment variables
# ENV CONDA_ENV_NAME=vd-instruct-py310
ENV WANDB_API_KEY=07ccc6b761d5c6126e5ea0cb287f222b6fa8d136

# clone DocOwl repo
RUN git clone https://github.com/nguyendinhson-kaist/mPLUG-DocOwl.git
RUN cd mPLUG-DocOwl && git checkout slurm-train

# install DocOwl
RUN pip install --upgrade pip
RUN cd mPLUG-DocOwl/DocOwl1.5 && pip install -e ".[train]"

RUN cd mPLUG-DocOwl/DocOwl1.5 && pip install flash-attn --no-build-isolation

# verify installation
RUN python -c "import torch"
RUN python -c "import mplug_docowl"

CMD ["/bin/bash"]