# Copyright (c) Meta, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# Installing the required packages
RUN apt update -y
RUN apt install -y git curl

# Installing Anaconda
WORKDIR /root
RUN curl https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh -o anaconda_installer.sh
RUN bash anaconda_installer.sh -b -p
ENV PATH="/root/anaconda3/bin:$PATH"

# Installing recommmended pre-requirements
RUN conda install "pytorch<1.13.0,>=1.4.0" torchvision torchaudio -c pytorch-lts -c nvidia
RUN pip install spacy==3.2.4 tokenizers pandas transformers fairseq contractions boto3==1.17.95 botocore==1.20.95

# Configuring packages for English
RUN python -m spacy download en_core_web_sm
RUN echo "import nltk; nltk.download(\"stopwords\"); nltk.download(\"punkt\")" > nltk_dl_script.py
RUN python nltk_dl_script.py

# Download the ParlAI Github repo
RUN git clone https://github.com/facebookresearch/ParlAI.git ~/ParlAI

# Running ParlAI install
RUN cd ~/ParlAI && \
    pip install -r requirements.txt && \
    python setup.py develop

CMD ["parlai", "party"]
