#!/usr/bin/env python
'''
| Filename    : distance_computer.py
| Description : Methods to compute distance between seeds and points
| Author      : Pushpendre Rastogi
| Created     : Wed Jan 11 18:18:07 2017 (-0500)
| Last-Updated: Wed Jan 11 19:24:17 2017 (-0500)
|           By: Pushpendre Rastogi
|     Update #: 7
'''
import numpy as np


def l2distance(seed, point):
    '''
    seed  : An M x d matrix
    point : An N x d matrix
    return : an MxN matrix of l2 distance between seeds and points
    --- Example ---
    from numpy import array
    l2distance(array([[1, 0], [0, 1], [1, 1]], dtype='float64'),
               array([[2, 0], [2, 2]], dtype='float64'))
    array([[1, 2.23], [2.23, 2.23], [1.414, 1.414]])
    '''
    assert seed.shape[1] == point.shape[1]
    seed_mag = np.square(np.linalg.norm(seed, axis=1))
    point_mag = np.square(np.linalg.norm(point, axis=1))
    two_n_XY = np.dot((-2 * seed), point.T)
    two_n_XY += seed_mag[:, None]
    two_n_XY += point_mag[None, :]
    return np.sqrt(two_n_XY, out=two_n_XY)
