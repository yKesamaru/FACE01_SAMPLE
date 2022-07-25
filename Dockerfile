FROM nvidia/cuda:11.0.3-base-ubuntu20.04
LABEL maintainer="yKesamaru <y.kesamaru@tokai-kaoninsho.com>"


ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute

# modified from https://github.com/danielguerra69/ubuntu-xrdp/blob/master/Dockerfile
ENV DEBIAN_FRONTEND noninteractive
RUN sed -i "s/# deb-src/deb-src/g" /etc/apt/sources.list
RUN apt-get -y update
RUN apt-get -yy upgrade
ENV BUILD_DEPS="git autoconf pkg-config libssl-dev libpam0g-dev \
    libx11-dev libxfixes-dev libxrandr-dev nasm xsltproc flex \
    bison libxml2-dev dpkg-dev libcap-dev"
RUN apt-get -yy install  sudo apt-utils software-properties-common $BUILD_DEPS



RUN apt install -y --fix-missing build-essential
RUN apt install -y --fix-missing cmake
RUN apt install -y --fix-missing curl
RUN apt install -y --fix-missing ffmpeg
RUN apt install -y --fix-missing fonts-mplus
RUN apt install -y --fix-missing gfortran
RUN apt install -y --fix-missing git
RUN apt install -y --fix-missing graphicsmagick
RUN apt install -y --fix-missing libatlas-base-dev
RUN apt install -y --fix-missing libavcodec-dev
RUN apt install -y --fix-missing libavformat-dev
RUN apt install -y --fix-missing libgraphicsmagick1-dev
RUN apt install -y --fix-missing libgtk2.0-dev
RUN apt install -y --fix-missing libjpeg-dev
RUN apt install -y --fix-missing liblapack-dev
RUN apt install -y --fix-missing libopenexr-dev
RUN apt install -y --fix-missing libpng-dev
RUN apt install -y --fix-missing libsm6
RUN apt install -y --fix-missing libswscale-dev
RUN apt install -y --fix-missing libwebp-dev
RUN apt install -y --fix-missing libx11-dev
RUN apt install -y --fix-missing libxext6
RUN apt install -y --fix-missing libxrender-dev
RUN apt install -y --fix-missing pkg-config
RUN apt install -y --fix-missing python-pip-whl
RUN apt install -y --fix-missing python3-dev
RUN apt install -y --fix-missing python3-numpy
RUN apt install -y --fix-missing python3-pip
RUN apt install -y --fix-missing python3-setuptools
RUN apt install -y --fix-missing python3-tk
RUN apt install -y --fix-missing python3-venv
RUN apt install -y --fix-missing python3-pkg-resources
RUN apt install -y --fix-missing software-properties-common
RUN apt install -y --fix-missing wget
RUN apt install -y --fix-missing zip

# ubuntu-desktop
RUN apt install -y --fix-missing xserver-xorg
RUN apt install -y --fix-missing xfce4

RUN apt clean && rm -rf /tmp/* /var/tmp/*

# Install FACE01
# RUN cat << EOS >> ~/.bashrc
#     export PATH="/usr/local/cuda/bin/:$PATH"
#     export LD_LIBRARY_PATH="/usr/local/cuda/lib64/:$LD_LIBRARY_PATH"
#     export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
#     export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# EOS
# RUN source ~/.bashrc

RUN git clone https://github.com/yKesamaru/FACE01_SAMPLE.git

# RUN python3 -m pip cache remove dlib
RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools
RUN python3 -m pip install -U wheel
RUN python3 -m pip install -r FACE01_SAMPLE/requirements.txt

RUN tar -jxvf FACE01_SAMPLE/dlib-19.24.tar.bz2
# `--clean` see bellow
# [Have you done sudo python3 setup.py install --clean yet?](https://github.com/davisking/dlib/issues/1686#issuecomment-471509357)
RUN python3 FACE01_SAMPLE/dlib-19.24/setup.py install --clean

# RUN python3 ./CALL_FACE01.py
CMD xfce4


# COPY . /root/disaster
# WORKDIR /root/disaster
# RUN python3 -m pip install -r requirements.txt

# CMD export LC_ALL=C.UTF-8 && \
#     export LANG=C.UTF-8 && \
#     # cd /root/disaster/create_face_data/shelter01/ && \
#     # python3 ./create_face_data_app.py && \
#     cd /root/disaster/web_app/ && \
#     export FLASK_APP=main.py && \
#     flask run --host=0.0.0.0


