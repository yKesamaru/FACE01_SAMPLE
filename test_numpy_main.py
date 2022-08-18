import numpy as np
start_stop = (0,2,0,2)

dim1 = np.arange(10)

print(f'dim1: \n{dim1}\n')
dim1_slice = dim1[
    start_stop[0]:start_stop[1]
]
print(f'dim1_slice: \n{dim1_slice}\n')
print(f'dim1.ndim: {dim1.ndim}\n')
print(f'dim1.shape: {dim1.shape}\n')

x = np.array([[[1],[2],[3]], [[4],[5],[6]]])

x_slice = x[
    start_stop[0]:start_stop[1]
]
print(f'x.slice: \n{x_slice}\n')
print(f'x.ndim: {x.ndim}\n')
print(f'x.shape: {x.shape}\n')

# dim2 = np.arange(9, dtype=np.uint8).reshape(3,3)
dim2 = np.arange(16, dtype=np.uint8).reshape(4,4)
print(f'dim2: \n{dim2}\n')
print(f'dim2.ndim: {dim2.ndim}\n')

dim2_slice_1 = dim2[
    start_stop[0]:start_stop[1],
    start_stop[2]:start_stop[3]
]
print(f'dim2_slice_1: \n{dim2_slice_1}\n')
print(f'dim2_slice_1.ndim: {dim2_slice_1.ndim}\n')

dim2_slice_2 = dim2[
    1:2,
    3:4
]
print(f'dim2_slice_2: \n{dim2_slice_2}\n')
print(f'dim2_slice_2.ndim: {dim2_slice_2.ndim}\n')

dim3 = np.arange(48, dtype=np.uint8).reshape(4,4,3)
# print(f'dim3: \n{dim3}\n')

# dim3_slice_1 = dim3[
#     start_stop[0]:start_stop[1],
#     start_stop[2]:start_stop[3]
# ]
# print(f'dim3_slice_1: \n{dim3_slice_1}\n')

# dim3_slice_2 = dim3[
#     start_stop[0]:start_stop[1],
#     start_stop[2]:start_stop[3],
#     0:1
# ]
# print(f'dim3_slice_2: \n{dim3_slice_2}\n')

# dim3_slice_3 = dim3[
#     start_stop[0]:start_stop[1],
#     start_stop[2]:start_stop[3],
#     0:dim3.shape[2]
# ]
# print(f'dim3_slice_3: \n{dim3_slice_3}\n')

# from face01lib.test_numpy import Test_numpy
# return_dim2 = Test_numpy().test_numpy(dim2, start_stop)
# # return_dim3 = Test_numpy().test_numpy(dim3, start_stop)
# # print(f'return_dim3: \n{return_dim3}\n')
# print(f'return_dim2: \n{return_dim2}\n')
