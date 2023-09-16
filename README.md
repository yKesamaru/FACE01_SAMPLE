<div align="center">

<img src="https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/images/g1320.png" width="200px">

⚡️ **SUPER HIGH SPEED RECOGNITION**  
⚡️ **USEFUL MANY METHODS**  
⚡️ **RICH AND COMPREHENSIVE DOCUMENTATION**  
FACE01 -- LET'S START !  

___

![GitHub commit activity](https://img.shields.io/github/commit-activity/y/yKesamaru/FACE01_SAMPLE) ![](https://img.shields.io/badge/Release-v2.1.05-blue) ![](https://img.shields.io/badge/Python-%3E%3D3.8-blue) ![](https://img.shields.io/github/deployments/yKesamaru/FACE01_SAMPLE/github-pages)
[![Open in Visual Studio Code](https://img.shields.io/static/v1?logo=visualstudiocode&label=&message=Open%20in%20Visual%20Studio%20Code&labelColor=2c2c32&color=007acc&logoColor=007acc)](https://open.vscode.dev/yKesamaru/FACE01_SAMPLE)

![](https://github.com/yKesamaru/FACE01_SAMPLE/blob/master/docs/img/ROMAN_HOLIDAY.GIF?raw=true)

</div>

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

---

📖 TOC
1. [About FACE01](#about-face01)
2. [ℹ️: Note](#ℹ️-note)
3. [Install](#install)
   1. [INSTALL\_FACE01.sh](#install_face01sh)
   2. [Docker](#docker)
4. [Example](#example)
5. [Document](#document)
6. [Configuration](#configuration)
7. [Update](#update)
8. [Note](#note)
9. [Acknowledgments](#acknowledgments)
10. [References](#references)

# About FACE01
---
✨ FACE01 is a **face recognition library** that integrates various functions and can be called from **Python**.

- 🎉 EfficientNetV2 Arcface Model is now available ! 
  - `EfficientNetV2 Arcface Model`は日本人の顔認証に特化したモデルです。
- `Real-time face recognition` is possible from face datas of **more than 10,000 people**
- Super high-speed face coordinate output function
- Face image saving function with date and time information
- You can set to modify output frame image
- Centralized management of functions by configuration file
- You can choose input protocol ex. RTSP, HTTP and USB
- You can use many function for `face-recognition` and `Image-processing` (See [Useful FACE01 library](https://ykesamaru.github.io/FACE01_SAMPLE/))
- ...and many others!


# ℹ️: Note
> - このリポジトリが提供するファイルは、無料でお使いいただけます。
> 教育機関でご利用の場合、ソースコードを研究・教育にご利用できます。
>   詳しくは[日本のAI教育を支援する、顔認識ライブラリ`FACE01`の提供について](docs/academic.md)をご覧ください。
> - 商用利用にあたっては別途ライセンスが必要です。

# Install
---
Setting up your FACE01 develop environment is really easy !
## INSTALL_FACE01.sh

```bash
wget https://raw.githubusercontent.com/yKesamaru/FACE01_SAMPLE/master/INSTALL_FACE01.sh
chmod +x INSTALL_FACE01.sh
bash -c ./INSTALL_FACE01.sh
```

See [here](docs/Installation.md).
## Docker
🐳 The easiest way to use Docker is to pull the image.  
See [here](docs/docker.md).

If you cannot use Docker by any means, please refer to [here](docs/Installation.md).


# Example
---
There are some example files in the example folder.  
Let's try *step-by-step* examples.  
See [here](docs/example_doc.md).

<div>
<img src="docs/img/benchmark_GUI.png" width="300px" >
<img src="docs/img/distort_barrel.png" width="300px" >
<img src="docs/img/benchmark_GUI_window.png" width="300px" >
<img src="docs/img/20_times.png" width="300px" >
</div>

If you want to see the exhaustive document, see [here](https://ykesamaru.github.io/FACE01_SAMPLE/).


# Document
---
- 🧑‍💻 [Step-by-step to use FACE01 library](https://github.com/yKesamaru/FACE01_SAMPLE/blob/master/docs/example_doc.md#step-by-step-to-use-face01-library)  
  - For beginner

    <img src="docs/img/step-by-step.png" width="400px" >

- 🧑‍💻 [Comprehensive and detailed documentation](https://ykesamaru.github.io/FACE01_SAMPLE/index.html)  
  - Comprehensive resource for intermediates 

    <img src="docs/img/document.png" width="400px" >


# Configuration
---
- Highly flexible, inheritable and easy-to-use configuration file: config.ini
  See [here](docs/config_ini.md).


# Update
---
- 🔖 v2.1.05
  - Add `EfficientNetV2 Arcface Model`


# Note
---
ℹ️
> This repository contains FACE01 SAMPLE for UBUNTU 20.04.  
  If you are a Windows user, please use this on Docker.  
  This sample can be used until December 2023.  


# Acknowledgments
---
📄 I would like to acknowledgments those who have published such wonderful libraries and models.  
1. [dlib](https://github.com/davisking/dlib) /  davisking
2. [face_recognition](https://github.com/ageitgey/face_recognition) /  ageitgey
3. [mediapipe](https://github.com/google/mediapipe) / google
4. [open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/anti-spoof-mn3) /  openvinotoolkit
5. [light-weight-face-anti-spoofing](https://github.com/kprokofi/light-weight-face-anti-spoofing) /  kprokofi
6. [openvino2tensorflow](https://github.com/PINTO0309/openvino2tensorflow) / Katsuya Hyodo (PINTO0309)
7. [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/191_anti-spoof-mn3) / Katsuya Hyodo (PINTO0309)
8. [FaceDetection-Anti-Spoof-Demo](https://github.com/Kazuhito00/FaceDetection-Anti-Spoof-Demo) / KazuhitoTakahashi (Kazuhito00)
9. Some images from [Pakutaso](https://www.pakutaso.com/), [pixabay](https://pixabay.com/ja/)

# References
---
- [Deep Face Recognition A Survey](https://arxiv.org/pdf/1804.06655.pdf)
- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/pdf/2104.00298.pdf)
- [ArcFace: Additive Angular Margin Loss for Deep](https://arxiv.org/pdf/1801.07698.pdf)
- [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/ftp/arxiv/papers/1804/1804.07573.pdf)
- [Dlib Python API](http://dlib.net/python/index.html)
- [Pytorch documentation and Python API](https://pytorch.org/docs/stable/index.html)
- [ONNX documentation](https://onnx.ai/onnx/)
- [教育と著作権](http://www.ic.daito.ac.jp/~mizutani/literacy/copyright.pdf): 水谷正大 著, 大東文化大学 (2021)