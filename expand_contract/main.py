#!/usr/bin/env python
'''
| Filename    : main.py
| Description :
| Author      : Pushpendre Rastogi
| Created     : Wed Jan 11 19:08:18 2017 (-0500)
| Last-Updated: Mon Jan 16 03:23:48 2017 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 52
'''
from distance_computer import l2distance
from schedule import Schedule
import numpy as np
import itertools
import contextlib
import time

@contextlib.contextmanager
def tictoc(msg):
    t = time.time()
    print "Started", msg
    yield
    print "\nCompleted", msg, "in %0.1fs" % (time.time() - t)



def inflation_ranking(seed, point, sched, D):
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
    ranking = inflation_ranking(seed, point, sched, D)
    print 'Finished ranking points'
