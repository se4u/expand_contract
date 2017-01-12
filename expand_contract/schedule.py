#!/usr/bin/env python
'''
| Filename    : schedule.py
| Description : A schedule of radius values
| Author      : Pushpendre Rastogi
| Created     : Wed Jan 11 18:33:46 2017 (-0500)
| Last-Updated: Wed Jan 11 19:07:28 2017 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 5
'''
import numpy as np


class Schedule(object):
    '''
    --- Example ---
    import numpy as np
    print list(Schedule(auto_config=np.array([[.1, .2], [.3, .4]])))
    '''
    @staticmethod
    def create_linear_list(distmat, size):
        mean = distmat[:].mean()
        stdev = np.sqrt(distmat[:].var())
        return np.linspace(mean - 2 * stdev, mean + 2 * stdev, num=size)

    def __init__(self, stype='linear', length=10, auto_config=None, **kwargs):
        self.length = length
        self.stype = stype
        if auto_config is None:
            self.lst = kwargs['lst']
        else:
            assert isinstance(auto_config, np.ndarray)
            if stype != 'linear':
                raise NotImplementedError(stype)
            self.lst = self.create_linear_list(auto_config, length)
        return

    def __len__(self):
        return len(self.lst)

    def __iter__(self):
        return iter(self.lst)

    def __getitem__(self, idx):
        return self.lst[idx]
