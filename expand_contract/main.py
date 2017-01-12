#!/usr/bin/env python
'''
| Filename    : main.py
| Description :
| Author      : Pushpendre Rastogi
| Created     : Wed Jan 11 19:08:18 2017 (-0500)
| Last-Updated: Wed Jan 11 19:27:24 2017 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 16
'''
from distance_computer import l2distance
from schedule import Schedule
import numpy as np


def expand_contract_and_count(seed, point, D=None, sched=None):
    if D is None:
        D = l2distance(seed, point)
    if sched is None:
        sched = Schedule(auto_config=D)
    ret = np.empty((point.shape[0], len(sched)), dtype='uint16')
    for idx, radius in enumerate(sched):
        ret[:, idx] = (D > radius).sum(axis=0)
    return ret


if __name__ == '__main__':
    seed = np.random.rand(1000, 100).astype('float32')
    point = np.random.rand(300000, 100).astype('float32')
    D = l2distance(seed, point)
    sched = Schedule(length=100, auto_config=D)
    ret = expand_contract_and_count(seed, point, sched=sched, D=D)
    print ret.shape, ret
