import numpy as np
from numpy import dot
from numpy import tensordot as tdot
from numpy.linalg import svd
from numpy.linalg import qr
from copy import copy
import peps
from time import time
from util import PeriodicArray
import sys

class OneSiteEnvironment:
    def __init__(self, e):
        self.e = e.reshape(e.size)
        
    def contract(self, a):
        return dot(self.e, a.reshape(self.e.size))

class BondEnvironment:
    """
    An instance of this class consists of 6 tensors e1, ..., e6, which
    represent the environment around a bond::
        
        +----e1----e2----+
        |     |     |    |
        e6--          --e3
        |     |     |    |
        +----e5----e4----+
    """
    def __init__(self, e1, e2, e3, e4, e5, e6):
        self.e1, self.e2, self.e3, self.e4, self.e5, self.e6 = e1, e2, e3, e4, e5, e6
        self.chi12 = self.e1.shape[2]
        self.chi23 = self.e2.shape[2]
        self.chi34 = self.e3.shape[2]
        self.chi45 = self.e4.shape[2]
        self.chi56 = self.e5.shape[2]
        self.chi61 = self.e6.shape[2]
    
    def contract(self, a, b=None):
        """
        Contracts the two site environment together with the inner two sites::
        
               |   |
            ---a---b---
               |   |
        
        If b is none, this method returns the environment of b.
        """
        tmp = tdot(self.e6, self.e1, [2,0])
        tmp = tdot(tmp, a, [[1,2],[0,3]])
        tmp = tdot(tmp, self.e5, [[0,3],[2,1]])
        tmp = tdot(self.e4, tmp, [2,2])
        tmp2 = tdot(self.e2, self.e3, [2,0])
        tmp = tdot(tmp2, tmp, [[0,3],[2,0]])
        if b is None:
            return tmp
        n = np.prod(tmp.shape)
        return dot(tmp.reshape(n), b.reshape(n))

class CTMRGDummyTester:
    def test(self, e):
        return False

class CTMRGObservableTester:
    def __init__(self, a, o, err=1e-6, x=0, y=0, verbose=False):
        self.__D = a.shape[1]**8
        self.a = peps.make_double_layer(a).reshape(self.__D)
        self.b = peps.make_double_layer(a, o=o).reshape(self.__D)
        self.targeterr = err
        self.err = 2*err
        self.__x = x
        self.__y = y
        self.__verbose = verbose
        self.__vals = np.array([1e10] * 10)
        self.__cnt = 0
    def test(self, e):
        e = e.get_site_environment(self.__x, self.__y)
        nval = e.contract(self.b) / e.contract(self.a)
        self.__cnt += 1
        self.__vals[self.__cnt % len(self.__vals)] = nval
        return (np.abs(self.__vals - np.mean(self.__vals)) < self.err).all()
    def get_value(self):
        return self.__vals[self.__cnt % len(self.__vals)]

class CTMRGEnergyTester:
    def __init__(self, a, b, o1, o2a, o2b, c1, c2, err=1e-6, verbose=False):
        self.__A = peps.make_double_layer(a)
        self.__B = peps.make_double_layer(b)
        self.__O1 = peps.make_double_layer(a, o=o1)
        self.__O2a = peps.make_double_layer(a, o=o2a)
        self.__O2b = peps.make_double_layer(a, o=o2b)
        self.__c1 = c1
        self.__c2 = c2
        self.__err = err
        self.__vals = np.array([1e10] * 10)
        self.__cnt = 0
        self.__verbose = verbose
    def test(self, e):
        e1 = e.get_site_environment()
        e2 = e.get_bond_environment()
        x1 = e1.contract(self.__O1) / e1.contract(self.__A)        
        x2 = e2.contract(self.__O2a, self.__O2b) / e2.contract(self.__A, self.__B)
        nval = self.__c1 * x1 + self.__c2 * x2
        self.__cnt += 1
        self.__vals[self.__cnt % len(self.__vals)] = nval
        return (np.abs(self.__vals - np.mean(self.__vals)) < self.__err).all()
    def get_value(self):
        return self.__vals[self.__cnt % len(self.__vals)]

class CTMRGGenericTester:
    def __init__(self, f, err=1e-6, verbose=False):
        self.__f = f
        self.__err = err
        self.__verbose = verbose
        self.__period = 10
        self.__vals = np.array([1e10] * self.__period)
        self.__cnt = 0
    def test(self, e):
        nval = self.__f(e)
        curerr = np.max(np.abs(1 - nval / self.__vals))
        self.__cnt += 1
        self.__vals[self.__cnt % self.__period] = nval
        #if self.__verbose:
        #    print "[CTMRGGenericTester.test] {:e} +/- {:e}".format(nval, np.max(curerr))
        return curerr < self.__err
    def get_value(self):
        return self.__vals[self.__cnt % len(self.__vals)]



class CTMEnvironment:
    def __init__(self, lut, c1, c2, c3, c4, t1, t2, t3, t4):
        self.lut = lut
        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
        self.t1, self.t2, self.t3, self.t4 = t1, t2, t3, t4
        
    def get_site_environment(self, site=0):
        lut = self.lut[site]
        x1 = tdot(self.c1[lut[-1,-1]], self.t1[lut[0,-1]], [1,0])
        x2 = tdot(self.c2[lut[+1,-1]], self.t2[lut[+1,0]], [1,0])
        x3 = tdot(self.c3[lut[+1,+1]], self.t3[lut[0,+1]], [1,0])
        x4 = tdot(self.c4[lut[-1,+1]], self.t4[lut[-1,0]], [1,0])
        return OneSiteEnvironment(tdot(tdot(x1, x2, [2,0]), tdot(x3, x4, [2,0]), [[0,3],[3,0]]))

    def get_bond_environment(self):
        return self.get_bond_environment_horizontal()

    def get_bond_environment_horizontal(self, site=0):
        """
        Returns the environment of (site = j)::
        
                 |         |
            ----a(j)---a(j+(1,0))---
                 |         |
         
        as::
        
            +----e1----e2----+
            |     |     |    |
            e6--          --e3
            |     |     |    |
            +----e5----e4----+
        """
        lut = self.lut[site]
        e1 = self.t1[lut[0,-1]]
        e2 = self.t1[lut[1,-1]]
        e3 = tdot(tdot(self.c2[lut[2,-1]], self.t2[lut[2,0]], [1,0]), self.c3[lut[2,1]], [2,0])
        e4 = self.t3[lut[1,1]]
        e5 = self.t3[lut[0,1]]
        e6 = tdot(tdot(self.c4[lut[-1,1]], self.t4[lut[-1,0]], [1,0]), self.c1[lut[-1,-1]], [2,0])
        return BondEnvironment(e1, e2, e3, e4, e5, e6)
        
    def get_bond_environment_vertical(self, site=0):
        """
        Returns the environment of (site = j)::
        
                   |
            ------a(j)------
                   |
            ---a(j+(0,1))---
                   |
          
        as::
        
            +----e6---+
            |    |    |
            e5--   --e1
            |         |
            e4--   --e2
            |    |    |
            +----e3---+
        """
        lut = self.lut[site]
        e1 = self.t2[lut[1,0]]
        e2 = self.t2[lut[1,1]]
        e3 = tdot(tdot(self.c3[lut[1,2]], self.t3[lut[0,2]], [1,0]), self.c4[lut[-1,2]], [2,0])
        e4 = self.t4[lut[-1,1]]
        e5 = self.t4[lut[-1,0]]
        e6 = tdot(tdot(self.c1[lut[-1,-1]], self.t1[lut[0,-1]], [1,0]), self.c2[lut[1,-1]], [2,0])
        return BondEnvironment(e1, e2, e3, e4, e5, e6)

def _contract_big_corner(c, t1, t4, a):
    """
    Returns::
    
        c---t1---
        |    |
        t4---a---
        |    |
    """
    chi0, D3, chi1 = t4.shape
    chi2, D0, chi3 = t1.shape
    D1, D2 = a.shape[1:3]
    tmp = dot(t4.reshape(chi0*D3, chi1), dot(c, t1.reshape(chi2, D0*chi3)))
    tmp = tmp.reshape(chi0, D3, D0, chi3).transpose([0,3,1,2]).reshape(chi0*chi3, D3*D0)
    a = a.transpose([3,0,1,2]).reshape(D3*D0, D1*D2)
    return dot(tmp, a).reshape(chi0, chi3, D1, D2).transpose([0,3,1,2]).reshape(chi0*D2, chi3*D1)

def _renormalise_corner1(c, t, p):
    """
    Returns::
        
         c---t---
         |   |
        +-----+
        |  p  |
        +-----+
           |
    """
    chi1 = c.shape[0]
    chi2, D, chi3 = t.shape
    tmp = dot(c, t.reshape(chi2, D*chi3))
    tmp = tmp.reshape(chi1*D, chi3)
    tmp = dot(p.T, tmp)
    tmp /= np.max(np.abs(tmp))
    return tmp

def _renormalise_corner2(c, t, p):
    """
    Returns::
    
           |
        +-----+
        |  p  |
        +-----+
         |   |
         c---t---
    """
    chi0, D, chi1 = t.shape
    chi2 = c.shape[1]
    tmp = dot(t.reshape(chi0*D, chi1), c)
    tmp = tmp.reshape(chi0, D, chi2)
    tmp = tmp.transpose([0,2,1])
    tmp = tmp.reshape(chi0, chi2*D)
    tmp = dot(tmp, p.T)
    tmp /= np.max(np.abs(tmp))
    return tmp

def _renormalise_row_transfer_tensor(t, a, p1, p2):
    """
    Returns::
    
           |
        +------+
        |  p1  |
        +------+
          |  |
          t--a---
          |  |
        +------+
        |  p2  |
        +------+
           |
    """
    chi1, D3, chi2 = t.shape
    D0, D1, D2 = a.shape[:3]
    chi0 = p2.shape[1]
    chi3 = p1.shape[0]
    tmp = p2.reshape(chi1, D2*chi0).T
    tmp = dot(tmp, t.reshape(chi1, D3*chi2))
    tmp = tmp.reshape(D2, chi0, D3, chi2)
    tmp = tmp.transpose([1,3,0,2])
    tmp = tmp.reshape(chi0*chi2, D2*D3)
    tmp = dot(tmp, a.reshape(D0*D1, D2*D3).T)
    tmp = tmp.reshape(chi0, chi2, D0, D1)
    tmp = tmp.transpose([0,3,1,2])
    tmp = tmp.reshape(chi0*D1, chi2*D0)
    tmp = dot(tmp, p1.T)
    tmp = tmp.reshape(chi0, D1, chi3)
    tmp /= np.max(np.abs(tmp))
    return tmp

def _ctmrg_step_horizontal(a, e, chi):
    n = len(a[0])
    p1 = [None] * n
    p2 = [None] * n
    p3 = [None] * n
    p4 = [None] * n
    
    for j in xrange(n):
        lut = e.lut[j]
        tmp = _contract_big_corner(e.c1[j], e.t1[lut[1,0]], e.t4[lut[0,1]], a[0][lut[1,1]])
        tmp2 = _contract_big_corner(e.c2[lut[3,0]], e.t2[lut[3,1]], e.t1[lut[2,0]], a[1][lut[2,1]])
        tmp = dot(tmp, tmp2)
        r1 = qr(tmp.T)[1]
        l2 = qr(tmp)[1]
        
        tmp = _contract_big_corner(e.c3[lut[3,3]], e.t3[lut[2,3]], e.t2[lut[3,2]], a[2][lut[2,2]])
        tmp2 = _contract_big_corner(e.c4[lut[0,3]], e.t4[lut[0,2]], e.t3[lut[1,3]], a[3][lut[1,2]])
        tmp = dot(tmp, tmp2)
        r2 = qr(tmp)[1]
        l1 = qr(tmp.T)[1]
        
        u, s, v = svd(dot(r1, r2.T))
        chi2 = np.min([np.count_nonzero(s), chi])
        u = u[:,:chi2]
        s = 1.0 / np.sqrt(s[:chi2])
        v = v[:chi2]
        
        u2, s2, v2 = svd(dot(l1, l2.T))
        chi2 = np.min([np.count_nonzero(s2), chi])
        u2 = u2[:,:chi2]
        s2 = 1.0 / np.sqrt(s2[:chi2])
        v2 = v2[:chi2]
        
        p1[lut[0,1]] = dot((u.conj()*s).T, r1) # <==> s u.H r1
        p2[lut[0,1]] = dot(r2.T, v.conj().T)*s # <==> r2.T v.H s
        p3[lut[3,2]] = dot((u2.conj()*s2).T, l1) # <==> s2 u2.H l1
        p4[lut[3,2]] = dot(l2.T, v2.conj().T)*s2 # <==> l2.T v2.H s2
        
    c1_new = [None] * n
    c2_new = [None] * n
    c3_new = [None] * n
    c4_new = [None] * n
    t2_new = [None] * n
    t4_new = [None] * n
    
    for j in xrange(n):
        lut = e.lut[j]
        c1_new[lut[1,0]] = _renormalise_corner1(e.c1[j], e.t1[lut[1,0]], p2[j])
        c4_new[lut[1,0]] = _renormalise_corner2(e.c4[j], e.t3[lut[1,0]], p1[lut[0,-1]])
        t4_new[lut[1,0]] = _renormalise_row_transfer_tensor(e.t4[j], a[0][lut[1,0]], p1[lut[0,-1]], p2[j])
        c3_new[lut[-1,0]] = _renormalise_corner1(e.c3[j], e.t3[lut[-1,0]], p4[j])
        c2_new[lut[-1,0]] = _renormalise_corner2(e.c2[j], e.t1[lut[-1,0]], p3[lut[0,1]])
        t2_new[lut[-1,0]] = _renormalise_row_transfer_tensor(e.t2[j], a[2][lut[-1,0]], p3[lut[0,1]], p4[j])
    
    e.c1 = c1_new
    e.c2 = c2_new
    e.c3 = c3_new
    e.c4 = c4_new
    e.t2 = t2_new
    e.t4 = t4_new

def _ctmrg_step_vertical(a, e, chi):
    n = len(a[0])
    p1 = [None] * n
    p2 = [None] * n
    p3 = [None] * n
    p4 = [None] * n
    
    for j in xrange(n):
        lut = e.lut[j]
        tmp = _contract_big_corner(e.c2[j], e.t2[lut[0,1]], e.t1[lut[-1,0]], a[1][lut[-1,1]])
        tmp2 = _contract_big_corner(e.c3[lut[0,3]], e.t3[lut[-1,3]], e.t2[lut[0,2]], a[2][lut[-1,2]])
        tmp = dot(tmp, tmp2)
        r1 = qr(tmp.T)[1]
        l2 = qr(tmp)[1]
        
        tmp = _contract_big_corner(e.c4[lut[-3,3]], e.t4[lut[-3,2]], e.t3[lut[-2,3]], a[3][lut[-2,2]])
        tmp2 = _contract_big_corner(e.c1[lut[-3,0]], e.t1[lut[-2,0]], e.t4[lut[-3,1]], a[0][lut[-2,1]])
        tmp = dot(tmp, tmp2)
        r2 = qr(tmp)[1]
        l1 = qr(tmp.T)[1]
        
        u, s, v = svd(dot(r1, r2.T))
        chi2 = np.min([np.count_nonzero(s), chi])
        u = u[:,:chi2]
        s = 1.0 / np.sqrt(s[:chi2])
        v = v[:chi2]
        
        u2, s2, v2 = svd(dot(l1, l2.T))
        chi2 = np.min([np.count_nonzero(s2), chi])
        u2 = u2[:,:chi2]
        s2 = 1.0 / np.sqrt(s2[:chi2])
        v2 = v2[:chi2]
        
        p1[lut[-1,0]] = dot((u.conj()*s).T, r1) # <==> s u.H r1
        p2[lut[-1,0]] = dot(r2.T, v.conj().T)*s # <==> r2.T v.H s
        p3[lut[-2,3]] = dot((u2.conj()*s2).T, l1) # <==> s2 u2.H l1
        p4[lut[-2,3]] = dot(l2.T, v2.conj().T)*s2 # <==> l2.T v2.H s2
        
    c1_new = [None] * n
    c2_new = [None] * n
    c3_new = [None] * n
    c4_new = [None] * n
    t1_new = [None] * n
    t3_new = [None] * n
    
    for j in xrange(n):
        lut = e.lut[j]
        c2_new[lut[0,1]] = _renormalise_corner1(e.c2[j], e.t2[lut[0,1]], p2[j])
        c1_new[lut[0,1]] = _renormalise_corner2(e.c1[j], e.t4[lut[0,1]], p1[lut[1,0]])
        t1_new[lut[0,1]] = _renormalise_row_transfer_tensor(e.t1[j], a[1][lut[0,1]], p1[lut[1,0]], p2[j])
        c4_new[lut[0,-1]] = _renormalise_corner1(e.c4[j], e.t4[lut[0,-1]], p4[j])
        c3_new[lut[0,-1]] = _renormalise_corner2(e.c3[j], e.t2[lut[0,-1]], p3[lut[-1,0]])
        t3_new[lut[0,-1]] = _renormalise_row_transfer_tensor(e.t3[j], a[3][lut[0,-1]], p3[lut[-1,0]], p4[j])
    
    e.c1 = c1_new
    e.c2 = c2_new
    e.c3 = c3_new
    e.c4 = c4_new
    e.t1 = t1_new
    e.t3 = t3_new

def _ctmrg_step(a0, a1, a2, a3, c1, c2, c3, c4, t1, t2, t3, t4, lut, chi, dx):
    n = len(a0)
    p1 = [None] * n
    p2 = [None] * n
    
    dx1 = dx[0]
    dx2 = dx[1]
    dy1 = dx2
    dy2 = -dx1
    
    for j in xrange(n):
        L = lut[j]
        tmp = _contract_big_corner(c1[j], t1[L[-dx1,-dx2]], t4[L[dy1,dy2]], a0[L[-dx1+dy1,-dx2+dy2]])
        tmp2 = _contract_big_corner(c2[L[-3*dx1,-3*dx2]], t2[L[-3*dx1+dy1,-3*dx2+dy2]], t1[L[-2*dx1,-2*dx2]], a1[L[-2*dx1+dy1,-2*dx2+dy2]])
        r1 = qr(dot(tmp, tmp2).T)[1]
        
        tmp = _contract_big_corner(c3[L[-3*dx1+3*dy1,-3*dx2+3*dy2]], t3[L[-2*dx1+3*dy1,-2*dx2+3*dy2]], t2[L[-3*dx1+2*dy1,-3*dx2+2*dy2]], a2[L[-2*dx1+2*dy1,-2*dx2+2*dy2]])
        tmp2 = _contract_big_corner(c4[L[3*dy1,3*dy2]], t4[L[2*dy1,2*dy2]], t3[L[-dx1+3*dy1,-dx2+3*dy2]], a3[L[dx1+2*dy1,dx2+2*dy2]])
        r2 = qr(dot(tmp, tmp2))[1]
        
        u, s, v = svd(dot(r1, r2.T))
        chi2 = np.min([np.count_nonzero(s), chi])
        u = u[:,:chi2]
        s = 1.0 / np.sqrt(s[:chi2])
        v = v[:chi2]
        
        p1[L[dy1,dy2]] = dot((u.conj()*s).T, r1) # <==> s u.H r1
        p2[L[dy1,dy2]] = dot(r2.T, v.conj().T)*s # <==> r2.T v.H s
        
    c1_new = [None] * n
    c4_new = [None] * n
    t4_new = [None] * n
    
    for j in xrange(n):
        L = lut[j]
        c1_new[L[-dx1,-dx2]] = _renormalise_corner1(c1[j], t1[L[-dx1,-dx2]], p2[j])
        c4_new[L[-dx1,-dx2]] = _renormalise_corner2(c4[j], t3[L[-dx1,-dx2]], p1[L[-dy1,-dy2]])
        t4_new[L[-dx1,-dx2]] = _renormalise_row_transfer_tensor(t4[j], a0[L[-dx1,-dx2]], p1[L[-dy1,-dy2]], p2[j])
    
    return c1_new, t4_new, c4_new


def ctmrg(a, lut, chi, env=None, tester=None, max_iterations=100000, verbose=False):
    t0 = time()
    
    n = len(a)
    a0 = a
    a1, a2, a3 = [None]*n, [None]*n, [None]*n
    for j in xrange(n):
        a1[j] = a[j].transpose([1,2,3,0])
        a2[j] = a[j].transpose([2,3,0,1])
        a3[j] = a[j].transpose([3,0,1,2])

    if env == "random":
        env = CTMEnvironment(lut, [None]*n, [None]*n, [None]*n, [None]*n, [None]*n, [None]*n, [None]*n, [None]*n)
        for j in xrange(n):
            env.c1[j] = np.random.rand(chi, chi)
            env.c2[j] = np.random.rand(chi, chi)
            env.c3[j] = np.random.rand(chi, chi)
            env.c4[j] = np.random.rand(chi, chi)
            env.t1[j] = np.random.rand(chi, a0[lut[j,0,+1]].shape[0], chi)
            env.t2[j] = np.random.rand(chi, a0[lut[j,-1,0]].shape[1], chi)
            env.t3[j] = np.random.rand(chi, a0[lut[j,0,-1]].shape[2], chi)
            env.t4[j] = np.random.rand(chi, a0[lut[j,+1,0]].shape[3], chi)
    elif type(env) != CTMEnvironment:
        env = CTMEnvironment(lut, [None]*n, [None]*n, [None]*n, [None]*n, [None]*n, [None]*n, [None]*n, [None]*n)
        D = a0[0].shape[0]
        D2 = D / int(np.sqrt(D))
        delta = np.identity(D / D2)[:D2].reshape(D) # if D2**2 = D holds, delta is just np.identity(D2) (aligned as a vector)
        for j in xrange(n):
            env.t1[j] = np.einsum(a0[j], [3,2,1,0], delta, [3])
            env.t2[j] = np.einsum(a0[j], [0,3,2,1], delta, [3])
            env.t3[j] = np.einsum(a0[j], [1,0,3,2], delta, [3])
            env.t4[j] = np.einsum(a0[j], [2,1,0,3], delta, [3])
            env.c1[j] = np.einsum(env.t1[j], [2,0,1], delta, [2])
            env.c2[j] = np.einsum(env.t1[j], [0,1,2], delta, [2])
            env.c3[j] = np.einsum(env.t3[j], [2,0,1], delta, [2])
            env.c4[j] = np.einsum(env.t3[j], [0,1,2], delta, [2])
    
    converged = False
    it = -1
    
    for it in xrange(max_iterations):
        #_ctmrg_step_horizontal(a, env, chi)
        #_ctmrg_step_vertical(a, env, chi)
        
        env.c1, env.t4, env.c4 = _ctmrg_step(a0, a1, a2, a3, env.c1, env.c2, env.c3, env.c4, env.t1, env.t2, env.t3, env.t4, lut, chi, [-1,  0])
        env.c2, env.t1, env.c1 = _ctmrg_step(a1, a2, a3, a0, env.c2, env.c3, env.c4, env.c1, env.t2, env.t3, env.t4, env.t1, lut, chi, [ 0, -1])
        env.c3, env.t2, env.c2 = _ctmrg_step(a2, a3, a0, a1, env.c3, env.c4, env.c1, env.c2, env.t3, env.t4, env.t1, env.t2, lut, chi, [ 1,  0])
        env.c4, env.t3, env.c3 = _ctmrg_step(a3, a0, a1, a2, env.c4, env.c1, env.c2, env.c3, env.t4, env.t1, env.t2, env.t3, lut, chi, [ 0,  1])
        
        if tester.test(env):
            converged = True
            break

    if not converged:
        sys.stderr.write("[ctmrg] warning: did not converge within {:d} iterations!\n".format(max_iterations))
        
    if verbose:
        print "[ctmrg] needed {:d} iterations and {:f} seconds".format(it+1, time()-t0)
    
    return env




class CTMEnvironment1x1Rotsymm:
    def __init__(self, c, t):
        self.c = c
        self.t = t
    
    def get_site_environment(self):
        tmp = (self.t.T * self.c).T
        tmp = tdot(tmp, tmp, [2,0])
        tmp = tdot(tmp, tmp, [[0,3],[3,0]])
        return OneSiteEnvironment(tmp)
        
    def eval_site(self, a):
        return self.get_site_environment().contract(a)
    
    def get_bond_environment(self):
        e3 = (self.t.T * self.c).T * self.c
        return BondEnvironment(self.t, self.t, e3, self.t, self.t, e3)
    
def get_bond_environment(self, x=0, y=0):
        e1 = e2 = self.t
        e2 = self.t
        e3 = tdot(tdot(self.c, self.t, [1,0]), self.c, [2,0])
        e4 = e5 = self.t
        e6 = tdot(tdot(self.c, self.t, [1,0]), self.c, [2,0])
        return BondEnvironment(e1, e2, e3, e4, e5, e6)

def ctmrg_1x1_rotsymm(a, chi, env=None, max_iterations=1000000, iteration_bunch=1, err=1e-6, verbose=False):

    t0 = time()
    D = a.shape[0]
    
    if type(env) == CTMEnvironment1x1Rotsymm:
        c = env.c
        t = env.t
    elif env == "random":
        c = np.random.rand(chi)
        t = np.ndarray((chi,D,chi))
        for j in xrange(D):
            tmp = np.random.rand(chi,chi)*2-1
            t[:,j,:] = np.dot(tmp, tmp.T)
    else:
        D2 = D / int(np.sqrt(D))
        delta = np.identity(D / D2)[:D2].reshape(D)
        t = np.einsum(a, [3,2,1,0], delta, [3])
        u,c,_ = np.linalg.svd(np.einsum(t, [2,0,1], delta, [2]))
        t = tdot(tdot(u, t, [0,0]), u.conj(), [2,1])
    
    for it in xrange(max_iterations):
        s = c
        tmp = np.fromfunction(lambda j,k,l: t[j,k,l]*c[l], t.shape, dtype=int) # todo: can be done as t*c?
        tmp = tdot(t, tmp, [0,2])
        tmp = tdot(tmp, a, [[0,3],[0,3]]).swapaxes(1,2)
        chi0 = tmp.shape[0]
        u,c,_ = svd(tmp.reshape(chi0*D, chi0*D))
        c = c[:chi]
        u = u[:,:chi]
        c /= c[0]
        
        tmp = tdot(u.reshape(chi0, D, len(c)), t, [0,0])
        tmp = tdot(tmp, a, [[0,2],[2,3]]).reshape(len(c), chi0*D, D)
        t = tdot(tmp, u.conj(), [1,0])
        t /= np.max(np.abs(t))
        
        if it % iteration_bunch == 0:
            curerr = np.sum(np.abs(c[:len(s)] - s)) + np.sum(np.abs(c[len(s):]))
            if curerr < err:
                break

    if verbose:
        print "[ctmrg] needed {:d} iterations and {:f} seconds".format(it+1, time()-t0)    
    
    return CTMEnvironment1x1Rotsymm(c, t)

