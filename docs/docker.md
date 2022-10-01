# To install Docker
See [here](Install_docker.md).


# Pull Docker image
The easiest way to use Docker is to pull the image.
![](img/PASTE_IMAGE_2022-08-29-07-43-12.png)

***NOTE***
If you are using a PC with NVIDIA GPU installed, the Docker Image you should pull is `face01_gpu:1.4.10`.
If not, the Docker Image you should pull is `face01_no_gpu:1.4.10`.


## When using `face01_gpu:1.4.10`
```bash
# Download Docker Image
docker pull tokaikaoninsho/face01_gpu:1.4.10

# Check `REPOSITORY`, `TAG`, and `IMAGE ID` of the downloaded Docker Image. (IMPORTANT!)
docker images

# Run Docker
docker run --rm -it \
  --gpus all -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix/:/tmp/.X11-unix: tokaikaoninsho/face01_gpu:1.4.10 
```


### **NOTE**
If you get an error as a result of entering the above command, try the following command.

```bash
docker run --rm -it \
  --gpus all -e DISPLAY=$DISPLAY \
  --device /dev/video0:/dev/video0:mwr \
  -v /tmp/.X11-unix/:/tmp/.X11-unix: tokaikaoninsho/face01_gpu:1.4.10 
```

***Don't forget activate*** the Python virtual environment (activate venv).
```bash
# Activate venv (IMPORTANT!)
. bin/activate
```


Let's try `simple.py` in example folder.
```bash
python example/simple.py
```


## When using `face01_no_gpu:1.4.10`
```bash
# Download Docker Image
docker pull tokaikaoninsho/face01_no_gpu:1.4.10

# Check `REPOSITORY`, `TAG`, and `IMAGE ID` of the downloaded Docker Image. (IMPORTANT!)
docker images

# Run Docker
docker run --rm -it -v /tmp/.X11-unix/:/tmp/.X11-unix: tokaikaoninsho/face01_no_gpu:1.4.10 
```


### **NOTE**
If you get an error as a result of entering the above command, try the following command.
```bash
docker run --rm -it \
  -e DISPLAY=$DISPLAY \
  --device /dev/video0:/dev/video0:mwr \
  -v /tmp/.X11-unix/:/tmp/.X11-unix: tokaikaoninsho/face01_no_gpu:1.4.10 
```

***Don't forget activate*** the Python virtual environment (activate venv).
```bash
# Activate venv (IMPORTANT!)
. bin/activate
```


Let's try `simple.py` in example folder.
```bash
python example/simple.py
```


# Build Docker image
If you want to build the Docker Image yourself, See [here](build_docker_image.md).

