#!/usr/bin/env python
'''
| Filename    : main.py
| Description :
| Author      : Pushpendre Rastogi
| Created     : Wed Jan 11 19:08:18 2017 (-0500)
| Last-Updated: Mon Jan 16 03:18:42 2017 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 50
'''
from distance_computer import l2distance
from schedule import Schedule
import numpy as np
import itertools
from rasengan import tictoc

def expand_contract_and_count(seed, point, sched, D):
    # APPROACH 1
    # ret = np.zeros((point.shape[0], len(sched)), dtype='uint16')
    # from fast_count import count_mat_gt_radii
    # count_mat_gt_radii(D, np.array(list(sched)), ret)

    # APPROACH 2
    # ret = np.zeros((point.shape[0], len(sched)), dtype='uint16')
    # for idx, radius in enumerate(sched):
    #     ret[:, idx] = (D > radius).sum(axis=0)

    # APPROACH 3
    # from fast_count import layercake
    # ret = layercake(D, sched)

    # Approach 4
    with tictoc('Step 1'):
        M_tilde = np.searchsorted(sched, D)
    with tictoc('Step 2'):
        M_tilde.sort(axis=1)
    assert M_tilde[0, 0] < M_tilde[0, 1]
    def cmp_fnc(i, ip):
        for j in xrange(M_tilde.shape[1]):
            mij = M_tilde[i,j]
            mipj = M_tilde[ip, j]
            if mij > mipj:
                return -1
            elif mij < mipj:
                return 1
        return 0
    with tictoc('Step 3'):
        ranking = sorted(xrange(D.shape[0]), cmp=cmp_fnc)
    return ranking


if __name__ == '__main__':
    seed = np.random.rand(1000, 100)
    point = np.random.rand(30000, 100)
    D = l2distance(point, seed)
    sched = Schedule(auto_config=D, keep_all=True)
    ret = expand_contract_and_count(seed, point, sched, D)
