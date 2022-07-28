
First, you have to clone FACE01_SAMPLE repository.
```bash
git clone https://github.com/yKesamaru/FACE01_SAMPLE.git
```
# To build FACE01 docker image with nvidia-docker2 package
## To make image
```bash
cd FACE01_SAMPLE
docker build -t face01_gui:1.4.03 -f docker/Dockerfile_xfce4 docker/ --network host
```
## To make image including xfce4
```bash
cd FACE01_SAMPLE
docker build -t face01_cui:1.4.03 -f docker/Dockerfile_console docker/ --network host
```
# To build FACE01 docker image without nvidia-docker2 package
```bash
cd FACE01_SAMPLE
docker build -t face01_no_gpu:1.4.03 -f docker/Dockerfile_no_gpu docker/ --network host
```
# Check the completed image.
```bash
docker images
```