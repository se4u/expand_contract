#!/usr/bin/env python
'''
| Filename    : main.py
| Description :
| Author      : Pushpendre Rastogi
| Created     : Wed Jan 11 19:08:18 2017 (-0500)
| Last-Updated: Tue Jan 17 16:51:05 2017 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 70
'''
from distance_computer import l2distance
from schedule import Schedule
import numpy as np
import contextlib
import time


@contextlib.contextmanager
def tictoc(msg):
    t = time.time()
    print "Started", msg
    yield
    print "\nCompleted", msg, "in %0.1fs" % (time.time() - t)


def arrange_by_index(M, I):
    M2 = M.copy()
    for i in xrange(M.shape[0]):
        M2[i] = M[i][I[i]]
    return M2

def inflation_ranking(sched, D, seed_labels):
    assert all(e in [-1, 1] for e in seed_labels)
    assert sched[0] < 0
    with tictoc('Step 1'):
        M = np.searchsorted(sched, D)
    with tictoc('Step 2'):
        # NOTE: use mergesort since it is a stable sort in numpy.
        # docs.scipy.org/doc/numpy/reference/generated/numpy.sort.html
        I_tilde = np.argsort(M, axis=1, kind='mergesort')
        M_tilde = arrange_by_index(M, I_tilde)
        G_tilde = (np.diff(M_tilde, axis=1)==0)
    assert M_tilde[0, 0] <= M_tilde[0, 1]
    def cmp_fnc(i, ip):
        for j in xrange(M_tilde.shape[1]):
            mij, mipj = M_tilde[i,j], M_tilde[ip, j]
            if mipj < mij:
                nbr_influence = seed_labels[I_tilde[ip, j]]
                return (1 if nbr_influence < 0 else -1)
            elif mipj > mij:
                nbr_influence = seed_labels[I_tilde[i, j]]
                return (1 if nbr_influence > 0 else -1)
            elif mij == mipj:
                lij = seed_labels[I_tilde[i, j]]
                lipj = seed_labels[I_tilde[ip, j]]
                if lij != lipj:
                    return (1 if lij > lipj else -1)
        return 0
    with tictoc('Step 3'):
        ranking = sorted(xrange(D.shape[0]), cmp=cmp_fnc)
    return ranking


def test1(args):
    point = np.random.rand(args.P, args.D)
    seed = np.random.rand(args.S, args.D)
    seed_labels = (np.rint(np.random.rand(args.S)) - 0.5)*2
    D = l2distance(point, seed)
    sched = Schedule(auto_config=D, keep_all=True)
    return sched, D, seed_labels

def test2(_):
    sched = np.array([-1, 1, 5, 9])
    D = np.array([[1, 1, 1, 1],
                  [5, 5, 1, 9],
                  [9, 1, 5, 5],
                  [5, 5, 9, 1],
                  [1, 9, 5, 5]], dtype='double')
    seed_labels = np.array([1, 1, -1, -1])
    return sched, D, seed_labels

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('--seed', default=0, type=int, help='Default={0}')
    arg_parser.add_argument('--S', default=1000, type=int)
    arg_parser.add_argument('--P', default=30, type=int)
    arg_parser.add_argument('--D', default=100, type=int)
    args=arg_parser.parse_args()
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    ranking = inflation_ranking(*test2())
    true_ranking = '(13/31)(0)(24/42)'
    print 'Finished ranking points', ranking, true_ranking
