![Logo](images/g1320.png)
# FACE01 SAMPLE
This repository contains FACE01 SAMPLE for UBUNTU 20.04.
If you are a Windows user, please use on WSL2.
This sample can be used until December 2022.

# New function!
***Anti spoof*** function was added to FACE01 in v1.4.03.

# About FACE01
FACE01 is a face recognition library that integrates various functions and can be called from Python.
You can call individual methods or call a set of functions.
See [docs/functions](docs/functions.md).

# Installation
```bash
wget https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/INSTALL_FACE01.sh
chmod +x INSTALL_FACE01.sh
bash -c ./INSTALL_FACE01.sh
```

NOTE: THIS IS *ONLY* USE FOR UBUNTU *20.04*.
If you are a Windows user, please use on `WSL2`.
If you are using another Linux distribution, use `Docker`, `Boxes`, or `lxd`.

Alternatively, refer to INSTALL_FACE01.sh and install manually, or use docker.

## If you want using Docker
### To install Docker
See [here](docs/install_docker.md).
You can choose from `To build Docker` or `Pull Docker image`.
#### To build Docker
See [here](docs/to_build_docker_image.md)

#### Pull Docker image
```bash
docker pull tokaikaoninsho/face01_gpu:1.4.03
```

- Digest
    - sha256:01c94cc3b60bab1846b4fcfb44f1fefa7afcfeac809109b0ec30a2ad867f0475
- OS/ARCH
  - linux/amd64
- Compressed Size
  - 8.85 GB

### Start FACE01 example
#### Dockerfile_gpu
This docker image is build with dockerfile named 'Dockerfile_gpu'.
```bash
# Launch Docker image
docker run --rm -it --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix: face01_gpu:1.4.03
# Enter the Python virtual environment
docker@e85311b5908e:~/FACE01_SAMPLE$ . bin/activate
(FACE01_SAMPLE) docker@e85311b5908e:~/FACE01_SAMPLE$ 
# Launch FACE01_SAMPLE
(FACE01_SAMPLE) docker@e85311b5908e:~/FACE01_SAMPLE$ python CALL_FACE01.py
```

![](img/PASTE_IMAGE_2022-07-20-07-00-03.png)
#### Dockerfile_no_gpu

```bash
docker run --rm -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix: face01_no_gpu:1.4.03

# Configuration
Edit `config.ini` file to configure FACE01.
If you want to use the http protocol as the input source, replace the `movie =` part of the `config.ini` file with` movie = http: // <IP> / cgi-bin / <parameter> `. If you want to store the authentication information, enter `user =" ", passwd =" "` in the `config.ini` file as above.
See [docs/config.ini](docs/config_ini.md) for details.

# Example
`python CALL_FACE01.py`
See [CALL_FACE01.py](CALL_FACE01.py) to refer the entire code.
## If you want to extract only face coordinates
Set `headless = True` on `config.ini`.

## Import FACE01
```python
import cProfile as pr
import PySimpleGUI as sg
import cv2
import time
from face01lib.video_capture import VidCap 
from memory_profiler import profile  # If you want to use @profile()
from sys import exit

import FACE01 as fg
```
## Set the number of playback frames
If you just want to try FACE01 a bit, you can limit the number of frames it loads.
```python
exec_times: int = 50
ALL_FRAME = exec_times
```
```python
next_frame_gen_obj = VidCap().frame_generator(fg.args_dict)
def extract_face_locations(exec_times):
    profile_HANDLING_FRAME_TIME_FRONT = time.perf_counter()
    i: int = 0
    for i in range(exec_times):
        i += 1
        if i >= exec_times:
            break
        next_frame = next_frame_gen_obj.__next__()
        frame_datas_array = fg.Core().frame_pre_processing(fg.logger, fg.args_dict,next_frame)
        for frame_datas in frame_datas_array:
            for face_location in frame_datas["face_location_list"]:
                print(face_location)
    
    print('Finish profiling')
    profile_HANDLING_FRAME_TIME_REAR = time.perf_counter()
    profile_HANDLING_FRAME_TIME = (profile_HANDLING_FRAME_TIME_REAR - profile_HANDLING_FRAME_TIME_FRONT) 
    print(f'Predetermined number of frames: {ALL_FRAME}')
    print(f'Number of frames processed: {i}')
    print(f'Total processing time: {round(profile_HANDLING_FRAME_TIME , 3)}[seconds]')
    print(f'Per frame: {round(profile_HANDLING_FRAME_TIME / i, 3)}[seconds]')
pr.run('extract_face_locations(exec_times)', 'restats')
```
## Result
```bash
...
(113, 522, 270, 365)
(124, 198, 281, 41)
(115, 525, 274, 366)
(114, 528, 276, 366)
(124, 199, 283, 40)
(115, 528, 276, 367)
(122, 200, 283, 39)
Finish profiling
Predetermined number of frames: 50
Number of frames processed: 50
Total processing time: 2.353[second]
Per frame: 0.047[second]
```
The face images are output to the `/output/` folder.
![](img/PASTE_IMAGE_2022-07-20-07-36-26.png)
You can see the profile result.
`snakeviz restats`
```bash
snakeviz restats
```
![](img/PASTE_IMAGE_2022-07-20-07-23-21.png)

## If you want to face recognition
Set `headless = True` on `config.ini`.
```python
def common_main(exec_times):
    profile_HANDLING_FRAME_TIME_FRONT = time.perf_counter()
    event = ''
    while True:
        try:
            frame_datas_array = fg.main_process().__next__()
        except Exception as e:
            print(e)
            exit(0)
        exec_times = exec_times - 1
        if  exec_times <= 0:
            break
        else:
            print(f'exec_times: {exec_times}')
            if fg.args_dict["headless"] == False:
                event, _ = window.read(timeout = 1)
                if event == sg.WIN_CLOSED:
                    print("The window was closed manually")
                    break
            for frame_datas in frame_datas_array:
                if "face_location_list" in frame_datas:
                    img, face_location_list, overlay, person_data_list = \
                        frame_datas['img'], frame_datas["face_location_list"], frame_datas["overlay"], frame_datas['person_data_list']
                    for person_data in person_data_list:
                        name, pict, date,  location, percentage_and_symbol = \
                            person_data['name'], person_data['pict'], person_data['date'],  person_data['location'], person_data['percentage_and_symbol']
                        if not name == 'Unknown':
                            print(
                                name, "\n",
                                "\t", "similarity\t", percentage_and_symbol, "\n",
                                "\t", "coordinate\t", location, "\n",
                                "\t", "time\t", date, "\n",
                                "\t", "output\t", pict, "\n",
                                "-------\n"
                            )
                    if fg.args_dict["headless"] == False:
                        imgbytes = cv2.imencode(".png", img)[1].tobytes()
                        window["display"].update(data = imgbytes)
        if fg.args_dict["headless"] == False:
            if event =='terminate':
                break
    if fg.args_dict["headless"] == False:
        window.close()
    
    profile_HANDLING_FRAME_TIME_REAR = time.perf_counter()
    profile_HANDLING_FRAME_TIME = (profile_HANDLING_FRAME_TIME_REAR - profile_HANDLING_FRAME_TIME_FRONT) 
    print(f'Predetermined number of frames: {ALL_FRAME}')
    print(f'Number of frames processed: {ALL_FRAME - exec_times}')
    print(f'Total processing time: {round(profile_HANDLING_FRAME_TIME , 3)}[seconds]')
    print(f'Per frame: {round(profile_HANDLING_FRAME_TIME / (ALL_FRAME - exec_times), 3)}[seconds]')
pr.run('common_main(exec_times)', 'restats')
```

## Result
```bash
...
exec_times: 2
麻生太郎 
         similarity      99.1% 
         coordinate      (114, 528, 276, 366) 
         time    2022,07,20,07,14,56,229442 
         output  output/麻生太郎_2022,07,20,07,14,56,254172_0.39.png 
 -------

菅義偉 
         similarity      99.3% 
         coordinate      (124, 199, 283, 40) 
         time    2022,07,20,07,14,56,229442 
         output  output/麻生太郎_2022,07,20,07,14,56,254172_0.39.png 
 -------

exec_times: 1
麻生太郎 
         similarity      99.3% 
         coordinate      (115, 528, 276, 367) 
         time    2022,07,20,07,14,56,340726 
         output   
 -------

菅義偉 
         similarity      99.3% 
         coordinate      (122, 200, 283, 39) 
         time    2022,07,20,07,14,56,340726 
         output   
 -------

Predetermined number of frames: 50
Number of frames processed: 50
Total processing time: 5.994[seconds]
Per frame: 0.12[seconds]
```
You can see the profile result.
`snakeviz restats`
```bash
snakeviz restats
```
![](img/PASTE_IMAGE_2022-07-20-07-20-01.png)

## If you want to display GUI window.
The processing speed will be slower, but you can use the GUI to display the window.
Set `headless = False` on `config.ini`.
```python
# PySimpleGUI layout
sg.theme('LightGray')
if fg.args_dict["headless"] == False:
    layout = [
        [sg.Image(filename='', key='display', pad=(0,0))],
        [sg.Button('terminate', key='terminate', pad=(0,10), expand_x=True)]
    ]
    window = sg.Window(
        'FACE01 EXAMPLE', layout, alpha_channel = 1, margins=(10, 10),
        location=(0,0), modal = True, titlebar_icon="./images/g1320.png", icon="./images/g1320.png"
    )

# @profile()
def common_main(exec_times):
    profile_HANDLING_FRAME_TIME_FRONT = time.perf_counter()
    event = ''
    while True:
        try:
            frame_datas_array = fg.main_process().__next__()
        except Exception as e:
            print(e)
            exit(0)
        exec_times = exec_times - 1
        if  exec_times <= 0:
            break
        else:
            print(f'exec_times: {exec_times}')
            if fg.args_dict["headless"] == False:
                event, _ = window.read(timeout = 1)
                if event == sg.WIN_CLOSED:
                    print("The window was closed manually")
                    break
            for frame_datas in frame_datas_array:
                if "face_location_list" in frame_datas:
                    img, face_location_list, overlay, person_data_list = \
                        frame_datas['img'], frame_datas["face_location_list"], frame_datas["overlay"], frame_datas['person_data_list']
                    for person_data in person_data_list:
                        if len(person_data) == 0:
                            continue
                        name, pict, date,  location, percentage_and_symbol = \
                            person_data['name'], person_data['pict'], person_data['date'],  person_data['location'], person_data['percentage_and_symbol']
                        # ELE: Equally Likely Events
                        if not name == 'Unknown':
                            result, score, ELE = Core_obj.return_anti_spoof(frame_datas['img'], person_data["location"])
                            if ELE is False:
                                print(
                                    name, "\n",
                                    "\t", "Anti spoof\t\t", result, "\n",
                                    "\t", "Anti spoof score\t", round(score * 100, 2), "%\n",
                                    "\t", "similarity\t\t", percentage_and_symbol, "\n",
                                    "\t", "coordinate\t\t", location, "\n",
                                    "\t", "time\t\t\t", date, "\n",
                                    "\t", "output\t\t\t", pict, "\n",
                                    "-------\n"
                                )
                    if fg.args_dict["headless"] == False:
                        imgbytes = cv2.imencode(".png", img)[1].tobytes()
                        window["display"].update(data = imgbytes)
        if fg.args_dict["headless"] == False:
            if event =='terminate':
                break
    if fg.args_dict["headless"] == False:
        window.close()
    
    profile_HANDLING_FRAME_TIME_REAR = time.perf_counter()
    profile_HANDLING_FRAME_TIME = (profile_HANDLING_FRAME_TIME_REAR - profile_HANDLING_FRAME_TIME_FRONT) 
    print(f'Predetermined number of frames: {ALL_FRAME}')
    print(f'Number of frames processed: {ALL_FRAME - exec_times}')
    print(f'Total processing time: {round(profile_HANDLING_FRAME_TIME , 3)}[seconds]')
    print(f'Per frame: {round(profile_HANDLING_FRAME_TIME / (ALL_FRAME - exec_times), 3)}[seconds]')
pr.run('common_main(exec_times)', 'restats')
```
## Result
```bash
...
exec_times: 1
麻生太郎 
         Anti spoof              not_spoof 
         Anti spoof score        89.0 %
         similarity              99.2% 
         coordinate              (113, 529, 277, 365) 
         time                    2022,07,24,19,42,02,574734 
         output                   
 -------

菅義偉 
         Anti spoof              spoof 
         Anti spoof score        89.0 %
         similarity              99.3% 
         coordinate              (122, 200, 283, 39) 
         time                    2022,07,24,19,42,02,574734 
         output                  output/菅義偉_2022,07,24,19,42,02,610419_0.34.png 
 -------

Predetermined number of frames: 50
Number of frames processed: 50
Total processing time: 10.654[seconds]
Per frame: 0.213[seconds]
```
![FACE01_GUI](https://user-images.githubusercontent.com/93259837/180339656-7ef7baea-480f-4d78-b29b-e8e12bc85189.gif)
![](img/PASTE_IMAGE_2022-07-24-19-43-44.png)

# References
1. [dlib](https://github.com/davisking/dlib)
2. [face_recognition](https://github.com/ageitgey/face_recognition)
3. [mediapipe](https://github.com/google/mediapipe)
4. [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/191_anti-spoof-mn3)

