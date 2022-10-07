![Logo](https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/images/g1320.png)

![GitHub commit activity](https://img.shields.io/github/commit-activity/y/yKesamaru/FACE01_SAMPLE)
![](https://img.shields.io/badge/Release-v1.4.08-blue)

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
 
# FACE01 SAMPLE
This repository contains FACE01 SAMPLE for UBUNTU 20.04.
If you are a Windows user, please use this on WSLg or Docker.
This sample can be used until December 2022.

# About FACE01
FACE01 is a face recognition library that integrates various functions and can be called from Python.
You can call individual methods or call a set of functions.
- High-speed face coordinate output function
- Face image saving function with date and time information
- Output modified image
- High-speed face recognition is possible from face data of more than 10,000 people
- Centralized management of functions by configuration file
- ...and many others!


See [docs/functions](https://github.com/yKesamaru/FACE01_SAMPLE/blob/master/docs/functions.md).

# Update
- v1.4.08
  - Fix memory leak bug.


# Installation
```bash
wget https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/INSTALL_FACE01.sh
chmod +x INSTALL_FACE01.sh
bash -c ./INSTALL_FACE01.sh
```

NOTE: THIS IS *ONLY* USE FOR UBUNTU *20.04*.
If you are a Windows user, please use on `WSLg` or `Docker`.
If you are using another Linux distribution, use `Docker`, `Boxes`, or `lxd`.

Alternatively, refer to INSTALL_FACE01.sh and install manually.

# If you want using Docker
See [here](https://github.com/yKesamaru/FACE01_SAMPLE/blob/master/docs/docker.md).

# Configuration
Edit `config.ini` file to configure FACE01.
If you use docker face01_gpu, you can modify the config.ini with `vim`.
If you want to use the http protocol as the input source, replace the `movie =` part of the `config.ini` file with ` movie = http: // <IP> / cgi-bin / <parameter> `. If you want to store the authentication information, enter `user =" ", passwd =" "` in the `config.ini` file as above.
See [docs/config.ini](https://github.com/yKesamaru/FACE01_SAMPLE/blob/master/docs/config_ini.md) for details.

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
![](https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/img/PASTE_IMAGE_2022-07-20-07-36-26.png)
You can see the profile result.
`snakeviz restats`
```bash
snakeviz restats
```
![](https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/img/PASTE_IMAGE_2022-07-20-07-23-21.png)

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
![](https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/img/PASTE_IMAGE_2022-07-20-07-20-01.png)

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

def common_main(exec_times):
    profile_HANDLING_FRAME_TIME_FRONT = time.perf_counter()
    event = ''
    while True:
        try:
            frame_datas_array = main_process().__next__()
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            exit(0)
        exec_times = exec_times - 1
        if  exec_times <= 0:
            break
        else:
            print(f'exec_times: {exec_times}')
            if args_dict["headless"] == False:
                event, _ = window.read(timeout = 1)
                if event == sg.WIN_CLOSED:
                    print("The window was closed manually")
                    break
            for frame_datas in frame_datas_array:
                if "face_location_list" in frame_datas:
                    img = frame_datas['img']
                    person_data_list = frame_datas['person_data_list']
                    
                    for person_data in person_data_list:
                        if person_data == {}:
                            continue

                        name = person_data['name']
                        pict = person_data['pict']
                        date = person_data['date']
                        location = person_data['location']
                        percentage_and_symbol = person_data['percentage_and_symbol']

                        spoof_or_real, score, ELE = \
                            Core_obj.return_anti_spoof(img, location)
                        # ELE: Equally Likely Events
                        if not name == 'Unknown':
                            # Bug fix
                            if args_dict["anti_spoof"] == True:
                                if ELE == False and spoof_or_real == 'real':
                                    print(
                                        name, "\n",
                                        "\t", "Anti spoof\t\t", spoof_or_real, "\n",
                                        "\t", "Anti spoof score\t", round(score * 100, 2), "%\n",
                                        "\t", "similarity\t\t", percentage_and_symbol, "\n",
                                        "\t", "coordinate\t\t", location, "\n",
                                        "\t", "time\t\t\t", date, "\n",
                                        "\t", "output\t\t\t", pict, "\n",
                                        "-------\n"
                                    )
                            else:
                                if ELE == False:
                                    print(
                                        name, "\n",
                                        "\t", "similarity\t\t", percentage_and_symbol, "\n",
                                        "\t", "coordinate\t\t", location, "\n",
                                        "\t", "time\t\t\t", date, "\n",
                                        "\t", "output\t\t\t", pict, "\n",
                                        "-------\n"
                                    )
                    if args_dict["headless"] == False:
                        imgbytes = cv2.imencode(".png", img)[1].tobytes()
                        window["display"].update(data = imgbytes)
        if args_dict["headless"] == False:
            if event =='terminate':
                break
    if args_dict["headless"] == False:
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
         Anti spoof              real 
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
![](https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/img/PASTE_IMAGE_2022-07-24-19-43-44.png)

# References
1. [dlib](https://github.com/davisking/dlib)
2. [face_recognition](https://github.com/ageitgey/face_recognition)
3. [mediapipe](https://github.com/google/mediapipe)
4. [open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/anti-spoof-mn3)
5. [light-weight-face-anti-spoofing](https://github.com/kprokofi/light-weight-face-anti-spoofing)
6. [openvino2tensorflow](https://github.com/PINTO0309/openvino2tensorflow)
7. [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/191_anti-spoof-mn3)
8. [FaceDetection-Anti-Spoof-Demo](https://github.com/Kazuhito00/FaceDetection-Anti-Spoof-Demo)