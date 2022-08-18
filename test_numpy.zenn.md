# numpy
## shape
> 各次元ごとの要素数
> https://deepage.net/features/numpy-shape.html

## ndim
> 次元数
> https://deepage.net/features/numpy-axis.html#ndarray%E3%81%AE%E6%AC%A1%E5%85%83%E6%95%B0ndim%E3%81%A8%E3%81%AF%E4%BD%95%E3%81%8B

## axis
> 座標軸。どの軸かを指定するための方法として、axisはshapeのインデックスに対応する。

## NumPy配列のスライシング
> 軸(axis)の番号が若いものから順に指定していきます。次元ごとには[:,:,:]のようにカンマ,で区切ります。
> https://deepage.net/features/numpy-slicing.html#%E5%A4%9A%E6%AC%A1%E5%85%83%E3%81%B8%E3%81%AE%E6%8B%A1%E5%BC%B5

# numpyのスライスの挙動
```python
import numpy as np
start_stop = (0,3,0,3)
```
## 1次元のスライス
```python
dim1 = np.arange(10, dtype=uint8)
print(f'dim1: \n{dim1}\n')
dim1_slice = dim1[
    start_stop[0]:start_stop[1]
]
print(f'dim1_slice: \n{dim1_slice}\n')
```
```bash
dim1: 
[0 1 2 3 4 5 6 7 8 9]

dim1_slice: 
[0 1 2]
```
## 2次元のスライス
```python
dim2 = np.arange(16, dtype=uint8).reshape(4,4)
print(f'dim2: \n{dim2}\n')
dim2_slice = dim2[
    start_stop[0]:start_stop[1],
    start_stop[2]:start_stop[3]
]
print(f'dim2_slice: \n{dim2_slice}\n')
```
```bash
dim2: 
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]

dim2_slice: 
[[ 0  1  2]
 [ 4  5  6]
 [ 8  9 10]]
 ```

## 3次元のスライス
![](img/PASTE_IMAGE_2022-08-16-12-17-38.png)
軸2方向を省略
```python
dim3 = np.arange(48, dtype=uint8).reshape(4,4,3)

print(f'dim3: \n{dim3}\n')
dim3_slice_1 = dim3[
    start_stop[0]:start_stop[1],
    start_stop[2]:start_stop[3]
]
print(f'dim3_slice_1: \n{dim3_slice_1}\n')
```
```bash
dim3: 
[[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]
  [ 9 10 11]]

 [[12 13 14]
  [15 16 17]
  [18 19 20]
  [21 22 23]]

 [[24 25 26]
  [27 28 29]
  [30 31 32]
  [33 34 35]]

 [[36 37 38]
  [39 40 41]
  [42 43 44]
  [45 46 47]]]

dim3_slice_1: 
[[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]]

 [[12 13 14]
  [15 16 17]
  [18 19 20]]

 [[24 25 26]
  [27 28 29]
  [30 31 32]]]
  ```
### 軸2方向を適当に指定
```python
dim3_slice_2 = dim3[
    start_stop[0]:start_stop[1],
    start_stop[2]:start_stop[3],
    0:1
]
print(f'dim3_slice_2: \n{dim3_slice_2}\n')
```
```bash
dim3_slice_2: 
[[[ 0]
  [ 3]
  [ 6]]

 [[12]
  [15]
  [18]]

 [[24]
  [27]
  [30]]]
```
### 軸2方向を全指定
```python
dim3_slice_3 = dim3[
    start_stop[0]:start_stop[1],
    start_stop[2]:start_stop[3],
    0:dim3.shape[2]
]
print(f'dim3_slice_3: \n{dim3_slice_3}\n')
```
```bash
dim3_slice_3: 
[[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]]

 [[12 13 14]
  [15 16 17]
  [18 19 20]]

 [[24 25 26]
  [27 28 29]
  [30 31 32]]]
```

# cppファイル作成
## compile.py
```python
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = \
    [Pybind11Extension(
        "test_numpy",
        ["./test_numpy.cpp"]
    )]

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules = ext_modules
)
```
## compile
```bash
python compile.py build_ext --inplace
```

# numpy
```bash
 terms  terms-Desks  ~/bin/FACE01  python
Python 3.8.10 (default, Jun 22 2022, 20:18:18) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> x = np.array([[[1],[2],[3]], [[4],[5],[6]]])
>>> x
array([[[1],
        [2],
        [3]],

       [[4],
        [5],
        [6]]])
>>> x.shape
(2, 3, 1)
>>> x.ndim
3
>>> 
```