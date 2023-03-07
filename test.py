# import numpy as np
# from joblib import Parallel, delayed
#
# dim1 = 2
# dim2 = 2
# dim3 = 2
#
# x = np.zeros((dim1, dim2, dim3))
#
# def process_element(ii):
#     x[ii,:,:] = np.ones((dim2, dim3))*ii
#
# Parallel(n_jobs=-1)(delayed(process_element)(ii) for ii in range(dim1))
#
# print(x)
#
# for ii in range(dim1):
#     x[ii,:,:] = np.ones((dim2,dim3)) * ii
#
# print(x)

import numpy as np
from joblib import Parallel, delayed

dim1 = 2
dim2 = 2
dim3 = 2

x = np.zeros((dim1, dim2, dim3))

def process_element(ii):
    y = np.zeros((dim2,dim3))
    y = np.ones((dim2,dim3)) * ii
    return y

x_parallel = Parallel(n_jobs=-1)(delayed(process_element)(ii) for ii in range(dim1))
x = np.array(x_parallel)

print(x)