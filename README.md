![Logo](https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/images/g1320.png)

![GitHub commit activity](https://img.shields.io/github/commit-activity/y/yKesamaru/FACE01_SAMPLE)
![](https://img.shields.io/badge/Release-v1.4.09-blue)

![](https://github.com/yKesamaru/FACE01_SAMPLE/blob/master/docs/img/ROMAN_HOLIDAY.GIF?raw=true)
```bash
# result
Audrey Hepburn 
         Anti spoof              real 
         Anti spoof score        100.0 %
         similarity              99.1% 
         coordinate              (123, 390, 334, 179) 
         time                    2022,08,09,04,19,35,552949 
         output                  output/Audrey Hepburn_2022,08,09,04,19,35,556237_0.39.png 
 -------
 ```
 

# About FACE01
FACE01 is a **face recognition library** that integrates various functions and can be called from **Python**.


- `Real-time face recognition` is possible from face datas of **more than 10,000 people**
- Super high-speed face coordinate output function
- Face image saving function with date and time information
- You can set to modify output frame image
- Centralized management of functions by configuration file
- You can choose input protocol ex. RTSP, HTTP and USB
- You can call individual methods or call a set of functions
- You can use many function for `face-recognition` and `Image-processing` (See [Useful FACE01 library](https://ykesamaru.github.io/FACE01_SAMPLE/))
- ...and many others!


# Update
- v1.4.09
  - More faster
  - Some bugs fix
  - Add examples


# Installation
See [here](docs/Installation.md).

# If you want using Docker
See [here](docs/docker.md).

# Configuration
See [here](docs/config_ini.md).

# Example
There are some example files in the example folder.
Let's try `example/simple.py` here.
`python3 example/simple.py`
See [simple.py](./example/simple.py) to refer the entire code.

## Result
```bash

[2022-09-27 19:20:48,174] [face01lib.load_priset_image] [simple.py] [INFO] Loading npKnown.npz
菅義偉 
         similarity              99.1% 
         coordinate              (138, 240, 275, 104) 
         time                    2022,09,27,19,20,49,835926 
         output                   
 -------

麻生太郎 
         similarity              99.6% 
         coordinate              (125, 558, 261, 422) 
         time                    2022,09,27,19,20,49,835926 
         output                   
 -------
 ...

```

# Note
This repository contains FACE01 SAMPLE for UBUNTU 20.04.
If you are a Windows user, please use this on WSLg or Docker.
This sample can be used until December 2022.


# Acknowledgments
I would like to acknowledgments those who have published such wonderful libraries and models.
1. [dlib](https://github.com/davisking/dlib) /  davisking 
2. [face_recognition](https://github.com/ageitgey/face_recognition) /  ageitgey 
3. [mediapipe](https://github.com/google/mediapipe) / google
4. [open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/anti-spoof-mn3) /  openvinotoolkit 
5. [light-weight-face-anti-spoofing](https://github.com/kprokofi/light-weight-face-anti-spoofing) /  kprokofi 
6. [openvino2tensorflow](https://github.com/PINTO0309/openvino2tensorflow) / Katsuya Hyodo (PINTO0309) 
7. [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/191_anti-spoof-mn3) / Katsuya Hyodo (PINTO0309) 
8. [FaceDetection-Anti-Spoof-Demo](https://github.com/Kazuhito00/FaceDetection-Anti-Spoof-Demo) / KazuhitoTakahashi (Kazuhito00) 