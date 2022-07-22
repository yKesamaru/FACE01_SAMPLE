# Functions tutorial
This section talk about `main_process` function.
More function's information is described [here](doc/../more_functions.md).
To use FACE01, it is enough to read this tutorial.

First, we have to import FACE01.
```python
import FACE01 as fg
```
## `main_process`
It is defined as follows in the `FACE01.py` file.
```python
def main_process():
    frame_datas_array = Core().frame_pre_processing(logger, args_dict, frame_generator_obj.__next__())
    face_encodings, frame_datas_array = Core().face_encoding_process(logger, args_dict, frame_datas_array)
    frame_datas_array = Core().frame_post_processing(logger, args_dict, face_encodings, frame_datas_array, GLOBAL_MEMORY)
    yield frame_datas_array
```
There are 3 part of functions.
1.  Core().frame_pre_processing(logger, args_dict, frame_generator_obj.__next__())
2.  Core().face_encoding_process(logger, args_dict, frame_datas_array)
3.  Core().frame_post_processing(logger, args_dict, face_encodings, frame_datas_array, GLOBAL_MEMORY)

`Core` is the class declared in `face01lib.Core`.
To import `Core` class, see `frame_pre_processing` section.

`main_process` method is the generator, return generator object.
```python
while True:
  frame_datas_array = fg.main_process().__next__()
  for frame_datas in frame_datas_array:
    if "face_location_list" in frame_datas:
        img, face_location_list, overlay, person_data_list = \
            frame_datas['img'], frame_datas["face_location_list"], frame_datas["overlay"], frame_datas['person_data_list']
        for person_data in person_data_list:
            name, pict, date,  location, percentage_and_symbol = \
                person_data['name'], person_data['pict'], person_data['date'],  person_data['location'], person_data['percentage_and_symbol']
```
Whole code is [CALL_FACE01.py](../CALL_FACE01.py).


### `frame_pre_processing`
```python
fg.Core().frame_pre_processing(fg.logger, fg.args_dict,next_frame)
```
will return `Dict` contains variables `img`, `face_location_list`, `overlay`,`person_data_list`.
- `img`
  - `numpy.ndarray`
- `face_location_list` variable is the
  -  `List[tuple]`
- `overlay`
  - `numpy.ndarray`
- `person_data_list`
  - `List[Dict{name:'', pict: '', date: '', location: tuple, percentage_and_symbol: ''}]`

In `face_location_list` or `person_data_list.location` contains coordinates of faces. If you want only coordinates of faces, this function is useful.

### `face_encoding_process`
If you want to get names who are in movie frames, you need call methods `frame_pre_processing`, `face_encoding_process`, and `face_encoding_process`.
The above three methods are grouped in the `main_process`.
`face_encoding_process` and `frame_post_processing` are few opportunities to use alone.
```python
fg.Core().face_encoding_process(fg.logger, fg.args_dict, fg.frame_datas_array)
```
will return `face_encodings`, `frame_datas_array`.
- `face_encodings`
  - numpy.ndarray[list, ...]
- `frame_datas_array`
  - [{'img': 'no-data_img', 'face_location_list': [...], 'overlay': array([], dtype=float64), 'person_data_list': [...]}]

### `frame_post_processing`
```python
 fg.Core().frame_post_processing(fg.logger,fg.args_dict, face_encodings, frame_datas_array, fg.GLOBAL_MEMORY)
 ```
will return `frame_datas_array`.
- [{'img': 'no-data_img', 'face_location_list': [...], 'overlay': array([], dtype=float64), 'person_data_list': [...]}]

## `person_data_list`
Above example, returned value `person_data_list` is very important.
Person_data_list contains some values as bellow.
- name
- pict
  - Saved faces image file path.
- date
- location
  - Coordinate of a face in a frame.
- percentage_and_symbol
  - Similarity of face in a frame represented by '%'.

### example
```python
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
```
will result 
```bash
菅義偉 
         similarity      99.3% 
         coordinate      (127, 182, 276, 33) 
         time    2022,07,22,11,05,01,426742 
         output  output/菅義偉_2022,07,22,11,05,01,459014_0.34.png 
 -------

麻生太郎 
         similarity      99.4% 
         coordinate      (122, 535, 281, 376) 
         time    2022,07,22,11,05,01,426742 
         output  output/菅義偉_2022,07,22,11,05,01,459014_0.34.png 
 -------
```
Whole code is described `CALL_FACE01.py`.