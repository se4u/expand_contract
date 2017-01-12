cimport numpy as np
from numpy cimport ndarray
import numpy as np
cimport cython
from cython cimport floating


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.embedsignature(True)
cdef unsigned short[:, :] count_mat_gt_radii_impl(double[:, :] D, double[:] radius, unsigned short[:,:] ret) nogil:
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
    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
cdef ndarray[np.uint16_t, ndim=2] layercake_impl(ndarray[np.uint32_t, ndim=2] indices, int s1, int s2):
    cdef ndarray[np.uint16_t,ndim=2] ret = np.zeros((s1, s2), dtype='uint16', order='F')
    cdef unsigned short one = 1
    cdef int i, j, k
    cdef int is0=indices.shape[0], is1=indices.shape[1]
    cdef unsigned int beg, rmdr
    for i in range(is0):
        for j in range(is1):
            beg = indices[i, j]
            for k in range(beg):
                ret[i, k] += one
            # rmdr = beg%10
            # for k in range(rmdr):
            #     ret[i, k] += one
            # if beg >= 10:
            #     for k in range(rmdr, beg, 10): # , s2
            #         ret[i, k+0] += one # 0
            #         ret[i, k+1] += one # 1
            #         ret[i, k+2] += one # 2
            #         ret[i, k+3] += one # 3
            #         ret[i, k+4] += one # 4
            #         ret[i, k+5] += one # 5
            #         ret[i, k+6] += one # 6
            #         ret[i, k+7] += one # 7
            #         ret[i, k+8] += one # 8
            #         ret[i, k+9] += one # 9

    return ret

def count_mat_gt_radii(D, r, ret):
    return count_mat_gt_radii_impl(D, r, ret)

def layercake(D, sched):
    indices = np.searchsorted(list(sched), D.T).astype('uint32')
    return layercake_impl(indices, indices.shape[0], len(sched))

#     return count_mat_gt_radii_impl[np.ndarray[float,2], np.ndarray[float,1], np.ndarray[short,2]](D, r, ret)
#     # if D.dtype == np.float32:
#     # else:
#     #     return count_mat_gt_radii_impl[np.ndarray[double,2],np.ndarray[double,1],np.ndarray[short,2]](D, r, ret)
