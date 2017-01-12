cimport numpy as np
import numpy as np
cimport cython
from cython cimport floating


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.embedsignature(True)
cdef count_mat_gt_radii_impl(double[:, :] D, double[:] radius, unsigned short[:,:] ret) nogil:
    cdef:
        unsigned int i=0, j=0, k=0
        short tmp=0
        double r
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            tmp = 0
	    r = radius[j]
            for k in range(D.shape[0]):
                tmp += (D[k,i] > r)
            ret[i,j] = tmp
            if tmp == 0:
                break
    return

def count_mat_gt_radii(D, r, ret):
    return count_mat_gt_radii_impl(D, r, ret)
#     return count_mat_gt_radii_impl[np.ndarray[float,2], np.ndarray[float,1], np.ndarray[short,2]](D, r, ret)
#     # if D.dtype == np.float32:
#     # else:
#     #     return count_mat_gt_radii_impl[np.ndarray[double,2],np.ndarray[double,1],np.ndarray[short,2]](D, r, ret)