# Step-by-step to use FACE01 library
Welcome to FACE01 world!

In this article, I will introduce the necessary knowledge and techniques to create an application that uses face recognition using FACE01 with an example program.

Are you ready?

Let's check the **checks you must pass** as step. 1.

## Checks you must pass
✅
- [x] Basic operation of Python is possible
-  [x] Can perform basic operations of Docker
-  [x] Basic operation of Linux terminal
-  [x] (If using Nvidia GPU) CUDA driver is already installed


Have you checked everything?
OK! Let's get started!


## [Register face images](register_faces.md)
This article describes how to register face images.


## activate virtual python mode
Start the virtual environment using venv of the Python standard library.

```bash
# activate venv
. bin/activate
```

## Check vim installed
The Docker Image comes with vim installed so you can edit `conf.ini`.

```bash
# Check vim installed
which vim
```


## [Simple face recognition](simple.md)
Let' try `simple.py`.
simple.py is an example script for CUI behavior.
Make sure `headless=True` in `conf.ini`.
If `headless = False`, modify the value to `True`.

```python
python example/simple.py
```


## [Display GUI window](display_GUI_win.md)
Want to display in a cool GUI window?
Try `example/benchmark_GUI_window.py`.
So, set the value `headless = False` in `config.ini` as opposed to simple.py.


## [Display 'telop' and 'logo' images which you're company's.](ch_telop.md)
Do you want your window to display your company logo or something?
Of course you can!


## [Want to benchmark?](benchmark_CUI.md)
See [here](benchmark_CUI.md).