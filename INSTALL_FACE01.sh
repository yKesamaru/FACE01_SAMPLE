#!/usr/bin/env bash
set -Ceux -o pipefail

# -----------------------------------------------------------------
# FACE01 SETUP INSTALLER
# THIS IS *ONLY* USE FOR UBUNTU *20.04*
# -----------------------------------------------------------------

sudo su
apt update && sudo apt upgrade -y
apt install -y build-essential
apt install -y cmake
apt install -y ffmpeg
apt install -y fonts-mplus
apt install -y git
apt install -y libavcodec-dev
apt install -y libavformat-dev
apt install -y libswscale-dev
apt install -y libx11-dev
apt install -y python3-dev
apt install -y python3-tk
apt install -y python3-venv
apt install -y wget

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

apt update && sudo apt upgrade -y
ubuntu-drivers autoinstall
apt install -y cuda
apt install -y libcublas
apt install -y libcudnn8
apt install -y libcudnn8-dev
apt install -y liblapack-dev
apt install -y libopenblas-dev
apt install -y nvidia-cuda-toolkit

apt autoremove -y

exit

cat << EOS
export PATH="/usr/local/cuda/bin/:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64/:$LD_LIBRARY_PATH"
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOS
source ~/.bashrc

mkdir -p ~/bin/DIST/
cd ~/bin/DIST/

python3 -m venv ./
source bin/activate

pip cache remove dlib
pip install -U pip
pip install -U wheel
pip install -U setuptools
pip install -r requirements.txt

wget http://dlib.net/files/dlib-19.24.tar.bz2
tar -jxvf dlib-19.24.tar.bz2
cd dlib-19.24
# `--clean` see bellow
# [Have you done sudo python3 setup.py install --clean yet?](https://github.com/davisking/dlib/issues/1686#issuecomment-471509357)
python3 setup.py install --clean

cd ../

git clone 

python CALL_FACE01.py

