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


See [Useful FACE01 library](https://ykesamaru.github.io/FACE01_SAMPLE/).

# Update
- v1.4.09
  - More faster
  - Some bugs fix
  - Add examples


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
There are some example files in the example folder.
Let's try `example/simple.py` here.
`python3 example/simple.py`
See [simple.py](example/simple.py) to refer the entire code.

## If you want to get only face recognition
Set `headless = True` on `config.ini`.

## Import FACE01 library
```python
from face01lib.Core import Core
from face01lib.Initialize import Initialize
```
## Set the number of playback frames
If you just want to try FACE01 a bit, you can limit the number of frames it loads.

```python
if __name__ == '__main__':
    main(exec_times = 5)
```

```python

def main(exec_times: int = 50):

    # Initialize
    CONFIG: Dict =  Initialize().initialize()

    # Make generator
    gen = Core().common_process(CONFIG)

    # Repeat 'exec_times' times
    for i in range(1, exec_times):

        # Call __next__() from the generator object
        frame_datas_array = gen.__next__()

        for frame_datas in frame_datas_array:
            
            for person_data in frame_datas['person_data_list']:
                if not person_data['name'] == 'Unknown':
                    print(
                        person_data['name'], "\n",
                        "\t", "similarity\t\t", person_data['percentage_and_symbol'], "\n",
                        "\t", "coordinate\t\t", person_data['location'], "\n",
                        "\t", "time\t\t\t", person_data['date'], "\n",
                        "\t", "output\t\t\t", person_data['pict'], "\n",
                        "-------\n"
                    )
```

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

The face images are output to the `/output/` folder.
![](https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/img/PASTE_IMAGE_2022-07-20-07-36-26.png)

Try `example/benchmark_CUI.py` in the same examples folder.
You can see the profile result.
Your browser will automatically display a very cool and easy to use graph using `snakeviz`.

```bash
snakeviz restats
```

![](https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/img/PASTE_IMAGE_2022-07-20-07-23-21.png)

## If you want to display GUI window
Want to display in a cool GUI window?
Try `example/benchmark_GUI_window.py`.

*Don't forget* to set `headless=True` on `config.ini`.

```python
import cv2
import PySimpleGUI as sg

from face01lib.Core import Core
from face01lib.Initialize import Initialize


def main(exec_times: int = 50):

    # Initialize
    CONFIG: Dict =  Initialize().initialize()


    if CONFIG["headless"] == True:
        print("""
        For this example, set config.ini as follows.
            > [MAIN] 
            > headless = False
        """)
        exit()


    # Make PySimpleGUI layout
    sg.theme('LightGray')
    layout = [
        [sg.Image(filename='', key='display', pad=(0,0))],
        [sg.Button('terminate', key='terminate', pad=(0,10), expand_x=True)]
    ]
    window = sg.Window(
        'FACE01 EXAMPLE',
        layout, alpha_channel = 1,
        margins=(10, 10),
        location=(0, 0),
        modal = True,
        titlebar_icon="./images/g1320.png",
        icon="./images/g1320.png"
    )


    gen = Core().common_process(CONFIG)
    

    # Repeat 'exec_times' times
    for i in range(1, exec_times):

        # Call __next__() from the generator object
        frame_datas_array = gen.__next__()

        event, _ = window.read(timeout = 1)

        if event == sg.WIN_CLOSED:
            print("The window was closed manually")
            break

        for frame_datas in frame_datas_array:
            
            for person_data in frame_datas['person_data_list']:
                if not person_data['name'] == 'Unknown':
                    print(
                        person_data['name'], "\n",
                        "\t", "similarity\t\t", person_data['percentage_and_symbol'], "\n",
                        "\t", "coordinate\t\t", person_data['location'], "\n",
                        "\t", "time\t\t\t", person_data['date'], "\n",
                        "\t", "output\t\t\t", person_data['pict'], "\n",
                        "-------\n"
                    )
            
                imgbytes = cv2.imencode(".png", frame_datas['img'])[1].tobytes()
                window["display"].update(data = imgbytes)
            
        if event =='terminate':
            break
    window.close()


if __name__ == '__main__':
    main(exec_times = 50)
```

All the code can be get [here](example/display_GUI_window.py).

## Result
```bash
...
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

...

```

![FACE01_GUI](https://user-images.githubusercontent.com/93259837/180339656-7ef7baea-480f-4d78-b29b-e8e12bc85189.gif)
![](https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/img/PASTE_IMAGE_2022-07-24-19-43-44.png)

# References
I would like to thank those who have published such wonderful libraries and models.
1. [dlib](https://github.com/davisking/dlib) /  davisking 
2. [face_recognition](https://github.com/ageitgey/face_recognition) /  ageitgey 
3. [mediapipe](https://github.com/google/mediapipe) / google
4. [open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/anti-spoof-mn3) /  openvinotoolkit 
5. [light-weight-face-anti-spoofing](https://github.com/kprokofi/light-weight-face-anti-spoofing) /  kprokofi 
6. [openvino2tensorflow](https://github.com/PINTO0309/openvino2tensorflow) / Katsuya Hyodo (PINTO0309) 
7. [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/191_anti-spoof-mn3) / Katsuya Hyodo (PINTO0309) 
8. [FaceDetection-Anti-Spoof-Demo](https://github.com/Kazuhito00/FaceDetection-Anti-Spoof-Demo) / KazuhitoTakahashi (Kazuhito00) 