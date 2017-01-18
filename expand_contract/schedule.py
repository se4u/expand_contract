#!/usr/bin/env python
'''
| Filename    : schedule.py
| Description : A schedule of radius values
| Author      : Pushpendre Rastogi
| Created     : Wed Jan 11 18:33:46 2017 (-0500)
| Last-Updated: Wed Jan 18 15:41:03 2017 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 13
'''
import numpy as np


def create_linear_list(distmat, size):
    mean = distmat[:].mean()
    stdev = np.sqrt(distmat[:].var())
    return np.linspace(mean - 2 * stdev, mean + 2 * stdev, num=size)


def Schedule(stype='linear', length=10, auto_config=None, keep_all=True, lst=None):
    '''
    --- Example ---
    import numpy as np
    print list(Schedule(auto_config=np.array([[.1, .2], [.3, .4]])))
    '''
    if auto_config is None:
        lst = lst
    else:
        assert isinstance(auto_config, np.ndarray)
        if keep_all:
            lst = np.concatenate(([-1], np.unique(auto_config)))
            assert (len(lst) == 1) or (lst[0] <= lst[1])
        elif stype == 'linear':
            lst = create_linear_list(auto_config, length)
        else:
            raise NotImplementedError(stype)
    return lst
