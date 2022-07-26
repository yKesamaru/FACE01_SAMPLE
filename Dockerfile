FROM nvidia/cuda:11.0.3-base-ubuntu20.04
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute

# FROM ubuntu:20.04
LABEL maintainer="yKesamaru <y.kesamaru@tokai-kaoninsho.com>"

ENV DEBIAN_FRONTEND noninteractive

ADD sources.list /etc/apt/
# RUN sed -i "s/# deb-src/deb-src/g" /etc/apt/sources.list
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

# 日本語
RUN locale-gen ja_JP.UTF-8
ENV LANG=ja_JP.UTF-8

# タイムゾーン
ENV TZ=Asia/Tokyo

# 日本語入力
ENV GTK_IM_MODULE=fcitx \
    QT_IM_MODULE=fcitx \
    XMODIFIERS=@im=fcitx \
    DefaultIMModule=fcitx

# docker内で使うユーザを作成する。
# ホストと同じUIDにする。
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
WORKDIR /home/${DOCKER_USER}/FACE01_SAMPLE
ADD ./* ./

RUN python3 -m venv .
RUN source bin/activate

RUN python3 -m pip cache remove dlib
RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools
RUN python3 -m pip install -U wheel
RUN python3 -m pip install -r requirements.txt

RUN tar -jxvf dlib-19.24.tar.bz2
# `--clean` see bellow
# [Have you done sudo python3 setup.py install --clean yet?](https://github.com/davisking/dlib/issues/1686#issuecomment-471509357)
WORKDIR dlib-19.24
RUN python3 setup.py install --clean
WORKDIR /home/${DOCKER_USER}/FACE01_SAMPLE

CMD startxfce4
# RUN python3 ./CALL_FACE01.py
