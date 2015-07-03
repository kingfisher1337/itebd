# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:25:54 2015

@author: michael
"""

import numpy as np

def rotate_right(a):
    return a.swapaxes(0,1).swapaxes(1,2).swapaxes(2,3)

def rotate_180(a):
    return a.swapaxes(0,2).swapaxes(1,3)

def rotate_left(a):
    return a.swapaxes(0,3).swapaxes(1,3).swapaxes(2,3)

class PeriodicArray(object):
    def __init__(self, shape):
        if type(shape) == int:
            shape = (shape,)
        self.shape = shape
        self.size = np.prod(np.array(shape))
        self.memory = [None]*self.size
        self.strides = np.concatenate([np.cumprod(shape[1:][::-1])[::-1], [1]])
    def __getitem__(self, i):
        return self.memory[np.dot(np.mod(i, self.shape), self.strides)]
    def __setitem__(self, i, x):
        self.memory[np.dot(np.mod(i, self.shape), self.strides)] = x
    
    def copy(self):
        c = PeriodicArray(self.shape)
        for j in xrange(len(self.memory)):
            c.memory[j] = self.memory[j]
        return c



def build_lattice_lookup_table(nns, minsize):
    d = len(nns)    # number of dimensions
    m = len(nns[0]) # number of distinct sites
    
    # determine periodicity of unit cell:
    period_len = [1]*d
    for direction in xrange(d):
        i = 0
        while nns[direction][i] != 0:
            i = nns[direction][i]
            period_len[direction] += 1
    
    # determine the required lengths of the lut
    lut_len = map(lambda j: minsize[j] if minsize[j] % period_len[j] == 0 else (minsize[j]/period_len[j]+1)*period_len[j], xrange(d))
    
    # finally build lut
    lut = np.ndarray([m] + lut_len, dtype=int)
    lut_len = tuple(lut_len)
    for s in xrange(m):
        for x in np.ndindex(lut_len):
            t = s
            for j in xrange(d):
                for _ in xrange(x[j]):
                    t = nns[j][t]
            lut[s][x] = t
    
    return lut




