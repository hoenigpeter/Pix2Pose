FROM tensorflow/tensorflow:1.15.5-gpu-py3
#FROM nvcr.io/nvidia/tensorflow:20.10-tf1-py3
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update -y
RUN apt-get install -y python3-dev python3-pip python3-tk vim git libgtk2.0-dev 
WORKDIR /Pix2Pose
COPY . .
RUN apt-get update && apt-get install -y \
    python-pip
RUN python3 -m pip install --upgrade pip
RUN pip3 install Cython
RUN pip3 install --upgrade -r requirements.txt
RUN apt-get install -y sudo