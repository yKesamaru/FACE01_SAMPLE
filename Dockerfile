FROM nvidia/cuda:11.0.3-base-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute

# FROM ubuntu:20.04
LABEL maintainer="yKesamkaoninsho.com>"

ENV DEBIAN_FRONTEND noninteractive

COPY docker/sources.list /etc/apt/
RUN sed -i "s/# deb-src/deb-src/g" /etc/apt/sources.list
RUN apt-get update && apt-get install -y \
	software-properties-common \
	apt-utils \
	sudo \
	supervisor \
	autoconf \
	bison \
	build-essential \
	cmake \
	curl \
	dbus-x11 \
	dpkg-dev \
	fcitx-imlist \
	fcitx-mozc \
	ffmpeg \
	flex \
	fonts-mplus \
	gfortran \
	git \
	graphicsmagick \
	language-pack-ja \
	language-pack-ja-base \
	less \
	libatlas-base-dev \
	libavcodec-dev \
	libavformat-dev \
	libcap-dev \
	libgraphicsmagick1-dev \
	libgtk2.0-dev \
	libjpeg-dev \
	liblapack-dev \
	libopenexr-dev \
	libpam0g-dev \
	libpng-dev \
	libsm6 \
	libssl-dev \
	libswscale-dev \
	libwebp-dev \
	libx11-dev \
	libxext6 \
	libxfixes-dev \
	libxml2-dev \
	libxrandr-dev \
	libxrender-dev \
	locales \
	nasm \
	openssh-server \
	pkg-config \
	python-pip-whl \
	python3-dev \
	python3-numpy \
	python3-pip \
	python3-pkg-resources \
	python3-setuptools \
	python3-tk \
	python3-venv \
	tzdata \
	uuid-runtime \
	vim \
	vlc \
	wget \
	x11-xserver-utils \
	xauth \
	xautolock \
	xfce4 \
	xfce4-terminal \
	xfce4-xkb-plugin \
	xinit \
	xorgxrdp \
	xprintidle \
	xrdp \
	xserver-xorg \
	xsltproc \
	zip \
	cuda \
	libcublas-11-7 \
	libcublas-dev-11-7 \
	libcudnn8 \
	libcudnn8-dev \
	liblapack-dev \
	libopenblas-dev \
	nvidia-cuda-toolkit \
	&& rm -rf /var/lib/apt/lists/*

# configure
RUN locale-gen ja_JP.UTF-8
ENV LANG=ja_JP.UTF-8
ENV TZ=Asia/Tokyo
ENV GTK_IM_MODULE=fcitx \
    QT_IM_MODULE=fcitx \
    XMODIFIERS=@im=fcitx \
    DefaultIMModule=fcitx
# add user
ARG DOCKER_UID=1000
ARG DOCKER_USER=docker
ARG DOCKER_PASSWORD=docker
RUN useradd -m \
  --uid ${DOCKER_UID} --groups sudo --shell /bin/bash ${DOCKER_USER} \
  && echo ${DOCKER_USER}:${DOCKER_PASSWORD} | chpasswd

WORKDIR /home/${DOCKER_USER}
RUN chown -R ${DOCKER_USER} ./
USER ${DOCKER_USER}

# Install FACE01
# RUN { \
#   cat << EOS >> .bashrc \
#   export PATH="/usr/local/cuda/bin/:$PATH" \
#   export LD_LIBRARY_PATH="/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" \
#   export PATH=/usr/local/cuda/bin${PATH:+:${PATH}} \
#   export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} \
# EOS}
# RUN source .bashrc

RUN mkdir /home/${DOCKER_USER}/FACE01_SAMPLE
WORKDIR /home/${DOCKER_USER}/FACE01_SAMPLE/
COPY ./face01lib /home/${DOCKER_USER}/FACE01_SAMPLE/face01lib
COPY ./output /home/${DOCKER_USER}/FACE01_SAMPLE/output
COPY ./noFace /home/${DOCKER_USER}/FACE01_SAMPLE/noFace
COPY ./images /home/${DOCKER_USER}/FACE01_SAMPLE/images
COPY ./requirements.txt /home/${DOCKER_USER}/FACE01_SAMPLE/
COPY ./config.ini /home/${DOCKER_USER}/FACE01_SAMPLE/
COPY ./FACE01.py /home/${DOCKER_USER}/FACE01_SAMPLE/
COPY ./CALL_FACE01.py /home/${DOCKER_USER}/FACE01_SAMPLE/
COPY ./npKnown.npz /home/${DOCKER_USER}/FACE01_SAMPLE/
COPY ./dlib-19.24.tar.bz2 /home/${DOCKER_USER}/FACE01_SAMPLE/
COPY ./SystemCheckLock /home/${DOCKER_USER}/FACE01_SAMPLE/
COPY ./*.mp4 /home/${DOCKER_USER}/FACE01_SAMPLE/
RUN echo ${DOCKER_PASSWORD} | sudo -S chown -R ${DOCKER_USER} /home/${DOCKER_USER}/FACE01_SAMPLE
COPY ./priset_face_images /home/${DOCKER_USER}/FACE01_SAMPLE/priset_face_images

RUN python3 -m venv /home/${DOCKER_USER}/FACE01_SAMPLE/ \
	&& . /home/${DOCKER_USER}/FACE01_SAMPLE/bin/activate \
	&& python3 -m pip install -U pip \
	&& python3 -m pip install -U setuptools \
	&& python3 -m pip install -U wheel \
	&& python3 -m pip install -r requirements.txt \
	&& tar -jxvf /home/${DOCKER_USER}/FACE01_SAMPLE/dlib-19.24.tar.bz2 \
	&& cd dlib-19.24 \
	&& python3 setup.py install --clean \
	&& cd ../

CMD startxfce4
# RUN python3 ./CALL_FACE01.py

# `--clean` see bellow
# [Have you done sudo python3 setup.py install --clean yet?](https://github.com/davisking/dlib/issues/1686#issuecomment-471509357)