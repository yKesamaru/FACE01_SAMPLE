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

# `Dlib_api` class
This class is modified from [face_recognition](https://github.com/ageitgey/face_recognition) by ageitgey. And model data from [dlib](https://github.com/davisking/dlib) by davisking. We will not to use 68 face model but also 5 face model. In FACE01 repository, not exist 68 face model and using it's code.
See to refer `Core.return_face_location_list` example.
```python
from face01lib.api import Dlib_api
Dlib_api_obj = Dlib_api()
```
## face_locations
Returns an array of bounding boxes of faces in a frame.
```python
face_list = Dlib_api().face_locations(img, number_of_times_to_upsample, model)
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
Whole example code is [here](example/../../example/api_face_locations.py).

# `Core` class
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

# `load_priset_image`
This function loads face images in `priset_face_images` folder, and make npKnown.npz file.


# `return_anti_spoof`
This method returns valuable `spoof_or_real`, `score` and `ELE`.

In general, the results inferred from the trained model are not clearly divided into 1 and 0. For this reason, FACE01 incorporates the concept of `ELE: Equally Likely Events`. `score` originally presents two numbers between 0 ~ 1. At this time, the difference between the two numbers is set to 0.4, and the combination of numbers with a difference of 0.4 or less is considered to be "similarly certain"(=Equally Likely Events). FACE01 expresses this as ELE. That is, if the difference between the two numbers is LESS 0.4, it is not possible to determine whether it is `spoof` or` not spoof`.
## example
```python
from face01lib.Core import Core
spoof_or_real, score, ELE = Core().return_anti_spoof(frame_datas['img'], person_data["location"])
if ELE is False:
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
```
Whole code is [here](../CALL_FACE01.py).

# `VidCap` class
This class is included in `video_capture`.
To import, see bellow.
```python
from face01lib.video_capture import VidCap
```

## `resize_frame`
Return numpy array of resized image.
```python
resized_frame = VidCap().resize_frame(set_width, set_height, original_frame)
```
Whole code is [here](../example/resize_frame.py).

## `return_movie_property`
Return property fps, height and width of input movie data.
```python
set_width,fps,height,width,set_height = VidCap().return_movie_property(set_width, vcap)
```
result
```bash
...
set_width:  750 
 set_height: 421 
 fps:  30 
```
Whole code is [here](../example/return_movie_property.py).