#!/usr/bin/env python
'''
| Filename    : main.py
| Description :
| Author      : Pushpendre Rastogi
| Created     : Wed Jan 11 19:08:18 2017 (-0500)
| Last-Updated: Sat Jan 21 01:45:58 2017 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 152
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
    assert np.all(D >= 0)
    with tictoc('Step 1'):
        M = np.searchsorted(sched, D)
        # Potentially paralle code.
        # from joblib import Parallel, delayed
        # Parallel(n_jobs=10)(delayed(np.searchsorted)(sched, D[i])
        #                     for i in range(D.shape[0]))
    with tictoc('Step 2'):
        # NOTE: use mergesort since it is a stable sort in numpy.
        # docs.scipy.org/doc/numpy/reference/generated/numpy.sort.html
        I_tilde = np.argsort(M, axis=1, kind='mergesort')
        M_tilde = arrange_by_index(M, I_tilde)
        M_tilde_max = M_tilde.max(axis=1)
        M_tilde_min = M_tilde.min(axis=1)
        G_tilde = (np.diff(M_tilde, axis=1)==0)
    assert (M_tilde.shape[1] == 1) or (M_tilde[0, 0] <= M_tilde[0, 1])
    def cmp_fnc(i, ip):
        def get_nbr_influence(row_idx, j):
            backtrack_cursor = j
            while j>0 and G_tilde[row_idx, j-1]:
                j -= 1
                backtrack_cursor = j
            nbr_influence = seed_labels[I_tilde[row_idx, backtrack_cursor]]
            if backtrack_cursor == G_tilde.shape[1]:
                return nbr_influence
            while G_tilde[row_idx, backtrack_cursor]:
                nbr_influence +=  seed_labels[I_tilde[row_idx, backtrack_cursor+1]]
                if backtrack_cursor == G_tilde.shape[1] - 1:
                    break
                else:
                    backtrack_cursor += 1
            return nbr_influence

        i_cursor, ip_cursor = 0, 0
        S = M_tilde.shape[1]
        while min(i_cursor, ip_cursor) < S:
            new_ip_cursor, new_i_cursor = ip_cursor, i_cursor
            ip_id_cursor, i_id_cursor = ip_cursor, i_cursor
            if i_cursor >= S:
                i_id_cursor = S-1
                new_ip_cursor = ip_cursor + 1
            elif ip_cursor >= S:
                ip_id_cursor = S-1
                new_i_cursor = i_cursor + 1
            elif M_tilde[ip, ip_cursor] >= M_tilde[i, i_cursor]:
                new_i_cursor = i_cursor + 1
            elif M_tilde[i, i_cursor] >= M_tilde[ip, ip_cursor]:
                new_ip_cursor = ip_cursor + 1
            else:
                raise Exception("IllegalState")
            mij, mipj = M_tilde[i,i_id_cursor], M_tilde[ip, ip_id_cursor]
            if mipj < mij:
                nbr_influence = get_nbr_influence(ip, ip_id_cursor)
                if nbr_influence != 0:
                    return (1 if nbr_influence < 0 else -1)
            elif mipj > mij:
                nbr_influence = get_nbr_influence(i, i_id_cursor)
                if nbr_influence != 0:
                    return (1 if nbr_influence > 0 else -1)
            elif mij == mipj:
                i_nbr_influence = get_nbr_influence(i, i_id_cursor)
                ip_nbr_influence = get_nbr_influence(ip, ip_id_cursor)
                if i_nbr_influence != ip_nbr_influence:
                    return (1 if i_nbr_influence > ip_nbr_influence else -1)
            ip_cursor, i_cursor = new_ip_cursor, new_i_cursor
            pass # Close while loop: while min(i_cursor, ip_cursor) < S
        # Handle the special case, that the radius of i/ip from all the
        # seeds is lesser than the radius of ip/i
        if (M_tilde_max[i] < M_tilde_min[ip]
            or M_tilde_min[i] > M_tilde_max[ip]):
            i_nbr_influence = get_nbr_influence(i, 0)
            ip_nbr_influence = get_nbr_influence(ip, 0)
            if i_nbr_influence != ip_nbr_influence:
                return (1 if i_nbr_influence > ip_nbr_influence else -1)
        return 0
    with tictoc('Step 3'):
        ranking = sorted(xrange(D.shape[0]), cmp=cmp_fnc)
    return ranking


def test1():
    import random
    random.seed(0)
    np.random.seed(0)
    point = np.random.rand(30, 100)
    seed = np.random.rand(1000, 100)
    seed_labels = (np.rint(np.random.rand(1000)) - 0.5)*2
    D = l2distance(point, seed)
    sched = Schedule(auto_config=D, keep_all=True)
    return sched, D, seed_labels

def test2():
    sched = np.array([-1, 1, 5, 9])
    D = np.array([[1, 1,1,  1],
                  [5, 1,5,  9],
                  [9, 5,1,  5],
                  [5, 9,5,  1],
                  [1, 5,9,  5]], dtype='double')
    seed_labels = np.array([1, -1, 1, -1])
    return sched, D, seed_labels


def test3():
    seeds = np.array([[0, -1], [0, 1]], dtype = 'double')
    seed_labels = np.array([1, -1])
    pts = np.array([[0, 0], [2, 1]], dtype = 'double')
    D = l2distance(pts, seeds)
    sched = np.concatenate([[-1],sorted(set(D.flatten()))])
    return  sched, D, seed_labels

def test4():
    seeds = np.array([[-3, 2], [3, 2], [3, -2]], dtype = 'double')
    seed_labels = np.array([-1, 1, 1])
    pts = np.array([[-1, -3], [0, 1]], dtype = 'double')
    D = l2distance(pts, seeds)
    sched = np.concatenate([[-1],sorted(set(D.flatten()))])
    return  sched, D, seed_labels

if __name__ == '__main__':
    assert inflation_ranking(*test1()) == [
        22, 8, 29, 10, 25, 3, 24, 28, 6, 4, 17, 15, 7, 2, 19,
        13, 5, 14, 9, 12, 11, 0, 21, 16, 1, 26, 23, 20, 27, 18]
    assert inflation_ranking(*test2()) in [[1,3,0,4,2],
                                          [1,3,0,2,4],
                                          [3,1, 0,2,4],
                                          [3,1,0,4,2]]
    assert inflation_ranking(*test3()) == [1, 0]
    assert inflation_ranking(*test4()) == [1, 0]
