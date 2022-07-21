# More functions tutorial
FACE01 have many functions inner `face01lib/'.
This section, we will talk about how to use useable functions in FACE01.

We need to configure in `config.ini`, so should import FACE01.
```python
import FACE01 as fg
```

In this tutorial, for testing functions, we have to input movie file. So every example import VidCap class.
You will get video frames object which is declared in `config.ini` file, and `VidCap().frame_generator()` needs augment `fg.args_dict`.
```python
from face01lib.video_capture import VidCap
next_frame_gen_obj = VidCap().frame_generator(fg.args_dict)
```
For getting frames, we have to call `__next__`.
```python
while True:
    next_frame = next_frame_gen_obj.__next__()
```

# api
This class is modified from [face_recognition](https://github.com/ageitgey/face_recognition) by ageitgey. And model data from [dlib](https://github.com/davisking/dlib) by davisking. We will not to use 68 face model but also 5 face model. In FACE01 repository, not exist 68 face model and using it's code.
See to refer `Core.return_face_location_list` example.
```python
from face01lib.api import Dlib_api
Dlib_api_obj = Dlib_api()
```
## face_locations
Returns an array of bounding boxes of faces in a frame.
```python
face_list = Dlib_api().face_locations(img, number_of_times_to_upsample, model, )
``` 
### example
```python
for i in range(exec_times):
    next_frame = next_frame_gen_obj.__next__()
    if model == 'cnn':
        print( [Dlib_api_obj._trim_css_to_bounds(Dlib_api_obj._rect_to_css(face.rect), next_frame.shape) for face in Dlib_api_obj._raw_face_locations(next_frame, number_of_times_to_upsample, model)])
    else:
        print( [Dlib_api_obj._trim_css_to_bounds(Dlib_api_obj._rect_to_css(face), next_frame.shape) for face in Dlib_api_obj._raw_face_locations(next_frame, number_of_times_to_upsample, model)])
```
### result
```bash
...
[(145, 177, 259, 63), (108, 521, 272, 357)]
[(134, 177, 248, 63), (92, 521, 256, 357)]
[(125, 199, 261, 62), (108, 521, 272, 357)]
[(125, 199, 261, 62), (92, 521, 256, 357)]
[(125, 199, 261, 62), (92, 521, 256, 357)]
[(138, 185, 275, 49), (92, 521, 256, 357)]
```
Whole example code is [here](example/../../example/api_face_locations.py)

# Core
Import Core class.
```python
from face01lib.Core import Core
```
## return_face_location_list
Return face location list. This function is much faster against `api.face_locations`.
```python
set_width = fg.args_dict['set_width']
set_height = fg.args_dict['set_height']
for i in range(exec_times):
    next_frame = next_frame_gen_obj.__next__()
    print(Core().return_face_location_list(next_frame, set_width, set_height,0, 0.4))
```
### result
```bash
...
[(135, 191, 281, 45), (91, 530, 257, 364)]
[(139, 192, 275, 56), (115, 527, 268, 374)]
[(134, 196, 274, 56)]
[(136, 199, 276, 59)]
[(131, 202, 277, 56), (121, 528, 280, 369)]
[(134, 200, 278, 56)]
[(137, 198, 277, 58)]
[(133, 199, 275, 57)]
[(132, 199, 274, 57), (117, 527, 269, 375)]
[(134, 197, 273, 58), (103, 529, 263, 369)]
```
Whole example code is [here](../example/Core_return_face_location_list.py).