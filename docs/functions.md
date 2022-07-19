# Functions
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
### `face_encoding_process`
### `frame_post_processing`