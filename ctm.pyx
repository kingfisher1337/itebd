import numpy as np
from numpy import dot
from numpy import tensordot as tdot
from numpy.linalg import svd
from numpy.linalg import qr
from copy import copy
import peps
from time import time
import sys
import os

class OneSiteEnvironment:
    def __init__(self, e):
        self.e = e.reshape(e.size)
        
    def contract(self, a):
        return dot(self.e, a.reshape(self.e.size))
    
    def get_reduced_density_matrix(self, a, b=None):
        if b is None:
            b = a.conj()
        sa = a.shape
        sb = b.shape
        Da4 = np.prod(sa[1:])
        Db4 = np.prod(sb[1:])
        
        rho = self.e.reshape(sa[1], sb[1], sa[2], sb[2], sa[3], sb[3], sa[4], sb[4])
        rho = rho.transpose([0,2,4,6,1,3,5,7]).reshape(Da4, Db4)
        rho = np.dot(a.reshape(sa[0], Da4), rho)
        rho = np.dot(rho, b.reshape(sb[0], Db4).T)
        return rho / np.trace(rho)

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
        tmp = tdot(tmp, a, [[1,2],[3,0]])
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
    def __init__(self, a, o, err=1e-6, site=0, verbose=False):
        self.__D = a.shape[1]**8
        self.a = peps.make_double_layer(a).reshape(self.__D)
        self.b = peps.make_double_layer(a, o=o).reshape(self.__D)
        self.targeterr = err
        self.err = 2*err
        self.__site = site
        self.__verbose = verbose
        self.__vals = np.array([1e10] * 10)
        self.__cnt = 0
    def test(self, e):
        e = e.get_site_environment(self.__site)
        nval = e.contract(self.b) / e.contract(self.a)
        self.__cnt += 1
        self.__vals[self.__cnt % len(self.__vals)] = nval
        return (np.abs(self.__vals - np.mean(self.__vals)) < self.err).all()
    def get_value(self):
        return self.__vals[self.__cnt % len(self.__vals)]

class CTMRGGenericTester:
    def __init__(self, f, err=1e-6, verbose=False):
        self.__f = f
        self.__err = err
        self.__verbose = verbose
        self.__period = 10
        self.__vals = []
        self.__errs = []
        self.__converged = False
    def test(self, e):
        nval = self.__f(e)
        self.__errs.append(np.inf if len(self.__vals) < self.__period else np.abs(1 - nval / self.__vals[-1]))
        if self.__verbose:
            print "[CTMRGGenericTester.test] error is {:e}".format(self.__errs[-1])
        self.__vals.append(nval)
        self.__converged = np.array(map(lambda e: e < self.__err, self.__errs[-self.__period:])).all()
        return self.__converged
    def get_value(self):
        return self.__vals[-1]
    def get_values(self):
        return self.__vals
    def get_errors(self):
        return self.__errs
    def is_converged(self):
        return self.__converged


class CTMRGTester:
    def __init__(self, f, err, abserr, checklen=10, verbose=False):
        self.__f = f
        self.__vals = []
        self.__errs = []
        self.__abserrs = []
        self.__converged = False
        self.__checklen = checklen
        self.__targeterr = err
        self.__targetabserr = abserr
        self.__verbose = verbose
        self.__okrellen = None
        self.__okabslen = None
        
    def test(self, e):
        nval = np.array(self.__f(e))
        n = len(nval)
        
        if self.__okrellen is None:
            self.__okrellen = [0]*n
            self.__okabslen = [0]*n
        
        nerr = np.ndarray((n,))
        nerrabs = np.ndarray((n,))
        nconverged = True
        for j in xrange(n):
            if len(self.__vals) == 0:
                nerr[j] = np.inf
                nerrabs[j] = np.inf
            else:
                nerr[j] = np.abs(nval[j] / self.__vals[-1][j] - 1)
                nerrabs[j] = np.abs(nval[j] - self.__vals[-1][j])
            
            if nerr[j] < self.__targeterr:
                self.__okrellen[j] += 1
            else:
                self.__okrellen[j] = 0
            
            if nerrabs[j] < self.__targetabserr:
                self.__okabslen[j] += 1
            else:
                self.__okabslen[j] = 0
            
            if self.__okrellen[j] < self.__checklen and self.__okabslen[j] < self.__checklen:
                nconverged = False
        
        if self.__verbose:
            print "[CTMRGTester.test] largest relative error is {:e}; largest absolute error is {:e}".format(np.max(nerr), np.max(nerrabs))
        
        self.__vals.append(nval)
        self.__errs.append(nerr)
        self.__abserrs.append(nerrabs)
        self.__converged = nconverged
        
        return self.__converged
    
    def get_value(self, j=-1):
        return self.__vals[j]
    
    def get_values(self):
        return self.__vals
    
    def get_error(self, j=-1):
        return self.__errs[j]
    
    def get_errors(self):
        return self.__errs
    
    def get_abserror(self, j=-1):
        return self.__abserrs[j]
    
    def get_abserrors(self):
        return self.__abserrs
    
    def is_converged(self):
        return self.__converged

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

    def get_bond_environment(self, site=0):
        return self.get_bond_environment_horizontal(site)

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
    
    # equivalent but non-optimised version:
    #tmp = tdot(c, t1, [1,0])
    #tmp = tdot(t4, tmp, [2,0])
    #tmp = tdot(tmp, a, [[2,1],[0,3]])
    #tmp = tmp.transpose([0,3,1,2])
    #s = tmp.shape
    #return tmp.reshape(s[0]*s[1], s[2]*s[3])

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
    tmp = tmp.swapaxes(1,2)
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

def _build_projectors(chi, c1, c2, c3, c4, t1, t2, t3, t4, t5, t6, t7, t8, a1, a2, a3, a4):
    """
    builds projectors for the network::
    
        c1---t1---t2---c2
        |    |    |     |
        t8---a1---a2---t3
        |    |    |     |
        t7---a3---a4---t4
        |    |    |     |
        c4---t6---t5---c3
    """
    tmp = _contract_big_corner(c1, t1, t8, a1)
    tmp2 = _contract_big_corner(c2, t3, t2, a2)
    r1 = qr(dot(tmp, tmp2).T)[1]
    tmp = _contract_big_corner(c3, t5, t4, a4)
    tmp2 = _contract_big_corner(c4, t7, t6, a3)
    r2 = qr(dot(tmp, tmp2))[1]
    
    try:
        tmp = dot(r1, r2.T)
        u, s, v = svd(tmp)
    except np.linalg.LinAlgError:
        np.savetxt("output/svd_mat_pid{:d}.dat".format(os.getpid()), tmp)
        raise
    
    s[s < (s[0]/1e12)] = 0 # keep only the largest 12 orders of magnitude
    chi2 = np.min([np.count_nonzero(s), chi]) # truncate singular values equal to zero
    u = u[:,:chi2]
    s = 1.0 / np.sqrt(s[:chi2])
    v = v[:chi2]
    
    # return s u.H r1, r2.T v.H s
    return dot((u.conj()*s).T, r1), dot(r2.T, v.conj().T)*s

def _ctmrg_step(a0, a1, a2, a3, c1, c2, c3, c4, t1, t2, t3, t4, lut, chi, dx):
    n = len(a0)
    p1 = [None] * n
    p2 = [None] * n
    
    for j in xrange(n):
        L = lut[j]
        x, y = dx
        s00, s10, s20, s30 = j,            L[-x,        -y], L[-2*x,        -2*y], L[-3*x,        -3*y]
        s01, s11, s21, s31 = L[  -y,   x], L[-x-y,     x-y], L[-2*x-y,     x-2*y], L[-3*x-y,     x-3*y]
        s02, s12, s22, s32 = L[-2*y, 2*x], L[-x-2*y, 2*x-y], L[-2*x-2*y, 2*x-2*y], L[-3*x-2*y, 2*x-3*y]
        s03, s13, s23, s33 = L[-3*y, 3*x], L[-x-3*y, 3*x-y], L[-2*x-3*y, 3*x-2*y], L[-3*x-3*y, 3*x-3*y]
        
        p1[s01], p2[s01] = _build_projectors(chi,
            c1[s00], c2[s30], c3[s33], c4[s03],
            t1[s10], t1[s20], t2[s31], t2[s32],
            t3[s23], t3[s13], t4[s02], t4[s01],
            a0[s11], a1[s21], a3[s12], a2[s22])
        
        # note: p1[s01] is the projector to renormalise c4[s02] (to get c4[s12])
        #       p2[s01] is the projector to renormalise c1[s01] (to get c1[s11])
        # note: p1[s01] is of shape chi x (D*chi)
        #       p2[s01] is of shape (D*chi) x chi
        
    c1_new = [None] * n
    c4_new = [None] * n
    t4_new = [None] * n
    
    for j in xrange(n):
        L = lut[j]
        x, y = dx
        s00 = L[y, -x]
        s01, s11 = j, L[-x, -y]
        
        c1_new[s11] = _renormalise_corner1(c1[s01], t1[s11], p2[s01])
        c4_new[s11] = _renormalise_corner2(c4[s01], t3[s11], p1[s00])
        t4_new[s11] = _renormalise_row_transfer_tensor(t4[s01], a0[s11], p1[s00], p2[s01])
        
    return c1_new, t4_new, c4_new


def ctmrg(a, lut, chi, env=None, tester=None, max_iterations=10000, verbose=False):
    t0 = time()
    
    n = len(a)
    a0 = a
    a1 = map(lambda a: a.transpose([1,2,3,0]), a0)
    a2 = map(lambda a: a.transpose([2,3,0,1]), a0)
    a3 = map(lambda a: a.transpose([3,0,1,2]), a0)
    
    # a1, a2 and a3 are the rotated versions of a0
    #       j          m          l          k
    #       |          |          |          |
    #    m--a0--k = l--a1--j = k--a2--m = j--a3--l
    #       |          |          |          |
    #       l          k          j          m

    if env == "random":
        if verbose:
            print "[ctmrg] initialising with random tensors"
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
    elif env == "default" or env is None:
        if verbose:
            print "[ctmrg] initialising with single site tensors"
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
        try:
            env.c1, env.t4, env.c4 = _ctmrg_step(a0, a1, a2, a3, env.c1, env.c2, env.c3, env.c4, env.t1, env.t2, env.t3, env.t4, lut, chi, [-1,  0])
            env.c2, env.t1, env.c1 = _ctmrg_step(a1, a2, a3, a0, env.c2, env.c3, env.c4, env.c1, env.t2, env.t3, env.t4, env.t1, lut, chi, [ 0, -1])
            env.c3, env.t2, env.c2 = _ctmrg_step(a2, a3, a0, a1, env.c3, env.c4, env.c1, env.c2, env.t3, env.t4, env.t1, env.t2, lut, chi, [ 1,  0])
            env.c4, env.t3, env.c3 = _ctmrg_step(a3, a0, a1, a2, env.c4, env.c1, env.c2, env.c3, env.t4, env.t1, env.t2, env.t3, lut, chi, [ 0,  1])
        except np.linalg.LinAlgError:
            print "ctmrg failed in iteration {:d}/{:d}".format(it, max_iterations)
            for j in xrange(n):
                print "a[{:d}] contains nan:".format(j), np.isnan(a0[j]).any()
                print "c1[{:d}] contains nan:".format(j), np.isnan(env.c1[j]).any()
                print "c2[{:d}] contains nan:".format(j), np.isnan(env.c2[j]).any()
                print "c3[{:d}] contains nan:".format(j), np.isnan(env.c3[j]).any()
                print "c4[{:d}] contains nan:".format(j), np.isnan(env.c4[j]).any()
                print "t1[{:d}] contains nan:".format(j), np.isnan(env.t1[j]).any()
                print "t2[{:d}] contains nan:".format(j), np.isnan(env.t2[j]).any()
                print "t3[{:d}] contains nan:".format(j), np.isnan(env.t3[j]).any()
                print "t4[{:d}] contains nan:".format(j), np.isnan(env.t4[j]).any()
            raise
        
        if tester.test(env):
            converged = True
            break

    if not converged:
        sys.stderr.write("[ctmrg] warning: did not converge within {:d} iterations!\n".format(max_iterations))
        
    if verbose:
        print "[ctmrg] needed {:d} iterations and {:f} seconds".format(it+1, time()-t0)
    
    return env


def ctmrg_post_tebd(a, lut, anew, bnew, j, orientation, chi, env):
    e = env
    anew = peps.make_double_layer(anew)
    bnew = peps.make_double_layer(bnew)
    
    if orientation == 0: # horizontal
        L = lut[lut[j,-1,-1]]
        p1, p2 = _build_projectors(chi,
            e.c1[L[0,0]], e.c2[L[3,0]], e.c3[L[3,3]], e.c4[L[0,3]],
            e.t1[L[1,0]], e.t1[L[2,0]], e.t2[L[3,1]], e.t2[L[3,2]],
            e.t3[L[2,3]], e.t3[L[1,3]], e.t4[L[0,2]], e.t4[L[0,1]],
            anew, bnew.transpose([1,2,3,0]), 
            a[L[1,2]].transpose([3,0,1,2]), a[L[2,2]].transpose([2,3,0,1]))
        L = lut[lut[j,-1,-2]]
        p3, p4 = _build_projectors(chi,
            e.c1[L[0,0]], e.c2[L[3,0]], e.c3[L[3,3]], e.c4[L[0,3]],
            e.t1[L[1,0]], e.t1[L[2,0]], e.t2[L[3,1]], e.t2[L[3,2]],
            e.t3[L[2,3]], e.t3[L[1,3]], e.t4[L[0,2]], e.t4[L[0,1]],
            a[L[1,1]], a[L[2,1]].transpose([1,2,3,0]),
            anew.transpose([3,0,1,2]), bnew.transpose([2,3,0,1]))
        e.c1[lut[j,0,-1]] = _renormalise_corner1(e.c1[lut[j,-1,-1]], e.t1[lut[j,0,-1]], p4)
        e.c4[lut[j,0,1]] = _renormalise_corner2(e.c4[lut[j,-1,1]], e.t3[lut[j,0,1]], p1)
        e.t4[j] = _renormalise_row_transfer_tensor(e.t4[lut[j,-1,0]], anew, p3, p2)
        
        L = lut[lut[j,-1,-1]]
        p1, p2 = _build_projectors(chi,
            e.c3[L[3,3]], e.c4[L[0,3]], e.c1[L[0,0]], e.c2[L[3,0]],
            e.t3[L[2,3]], e.t3[L[1,3]], e.t4[L[0,2]], e.t4[L[0,1]],
            e.t1[L[1,0]], e.t1[L[2,0]], e.t2[L[3,1]], e.t2[L[3,2]],
            a[L[2,2]].transpose([2,3,0,1]), a[L[1,2]].transpose([3,0,1,2]), 
            bnew.transpose([1,2,3,0]), anew)
        L = lut[lut[j,-1,-2]]
        p3, p4 = _build_projectors(chi,
            e.c3[L[3,3]], e.c4[L[0,3]], e.c1[L[0,0]], e.c2[L[3,0]],
            e.t3[L[2,3]], e.t3[L[1,3]], e.t4[L[0,2]], e.t4[L[0,1]],
            e.t1[L[1,0]], e.t1[L[2,0]], e.t2[L[3,1]], e.t2[L[3,2]],
            bnew.transpose([2,3,0,1]), anew.transpose([3,0,1,2]),
            a[L[2,1]].transpose([1,2,3,0]), a[L[1,1]])
        e.c3[lut[j,1,1]] = _renormalise_corner1(e.c3[lut[j,2,1]], e.t3[lut[j,1,1]], p2)
        e.c2[lut[j,1,-1]] = _renormalise_corner2(e.c2[lut[j,2,-1]], e.t1[lut[j,1,-1]], p3)
        e.t2[lut[j,1,0]] = _renormalise_row_transfer_tensor(e.t2[lut[j,2,0]], bnew.transpose([2,3,0,1]), p1, p4)
    else: # vertical 
        L = lut[lut[j,-1,-1]]
        p1, p2 = _build_projectors(chi,
            e.c2[L[3,0]], e.c3[L[3,3]], e.c4[L[0,3]], e.c1[L[0,0]],
            e.t2[L[3,1]], e.t2[L[3,2]], e.t3[L[2,3]], e.t3[L[1,3]],
            e.t4[L[0,2]], e.t4[L[0,1]], e.t1[L[1,0]], e.t1[L[2,0]],
            anew, a[L[2,1]].transpose([1,2,3,0]),
            bnew.transpose([3,0,1,2]), a[L[2,2]].transpose([2,3,0,1]))
        L = lut[lut[j,-2,-1]]
        p3, p4 = _build_projectors(chi,
            e.c2[L[3,0]], e.c3[L[3,3]], e.c4[L[0,3]], e.c1[L[0,0]],
            e.t2[L[3,1]], e.t2[L[3,2]], e.t3[L[2,3]], e.t3[L[1,3]],
            e.t4[L[0,2]], e.t4[L[0,1]], e.t1[L[1,0]], e.t1[L[2,0]],
            a[L[1,1]], anew.transpose([1,2,3,0]),
            a[L[1,2]].transpose([3,0,1,2]), bnew.transpose([2,3,0,1]))
        e.c2[lut[j,1,0]] = _renormalise_corner1(e.c2[lut[j,1,-1]], e.t2[lut[j,1,0]], p2)
        e.c1[lut[j,-1,0]] = _renormalise_corner2(e.c1[lut[j,-1,-1]], e.t4[lut[j,-1,0]], p3)
        e.t1[j] = _renormalise_row_transfer_tensor(e.t1[lut[j,0,-1]], anew.transpose([1,2,3,0]), p1, p4)
        
        L = lut[lut[j,-1,-1]]
        p1, p2 = _build_projectors(chi,
            e.c4[L[0,3]], e.c1[L[0,0]], e.c2[L[3,0]], e.c3[L[3,3]],
            e.t4[L[0,2]], e.t4[L[0,1]], e.t1[L[1,0]], e.t1[L[2,0]],
            e.t2[L[3,1]], e.t2[L[3,2]], e.t3[L[2,3]], e.t3[L[1,3]],
            bnew.transpose([3,0,1,2]), anew,
            a[L[2,2]].transpose([2,3,0,1]), a[L[2,1]].transpose([1,2,3,0]))
        L = lut[lut[j,-2,1]]
        p3, p4 = _build_projectors(chi,
            e.c4[L[0,3]], e.c1[L[0,0]], e.c2[L[3,0]], e.c3[L[3,3]],
            e.t4[L[0,2]], e.t4[L[0,1]], e.t1[L[1,0]], e.t1[L[2,0]],
            e.t2[L[3,1]], e.t2[L[3,2]], e.t3[L[2,3]], e.t3[L[1,3]],
            a[L[1,2]].transpose([3,0,1,2]), a[L[1,1]],
            bnew.transpose([2,3,0,1]), anew.transpose([1,2,3,0]))
        e.c4[lut[j,-1,1]] = _renormalise_corner1(e.c4[lut[j,0,-1]], e.t4[lut[j,-1,1]], p4)
        e.c3[lut[j,1,1]] = _renormalise_corner2(e.c3[lut[j,1,2]], e.t3[lut[j,1,1]], p1)
        e.t3[lut[j,0,1]] = _renormalise_row_transfer_tensor(e.t3[lut[j,0,2]], bnew.transpose([3,0,1,2]), p3, p2)

class CTMEnvironment1x1Rotsymm:
    def __init__(self, c, t):
        self.c = c
        self.t = t
    
    def get_site_environment(self, site=0):
        tmp = (self.t.T * self.c).T
        tmp = tdot(tmp, tmp, [2,0])
        tmp = tdot(tmp, tmp, [[0,3],[3,0]])
        return OneSiteEnvironment(tmp)
    
    def get_bond_environment(self, site=0):
        e3 = (self.c * self.t.T).T * self.c
        return BondEnvironment(self.t, self.t, e3, self.t, self.t, e3)

def ctmrg_1x1_rotsymm(a, chi, env=None, tester=None, max_iterations=1000000, iteration_bunch=1, verbose=False):
    t0 = time()
    D = a.shape[0]
    converged = False
    
    if env == "random":
        c = np.random.rand(chi)
        c = np.sort(c)[::-1]
        t = np.random.rand(chi, D, chi)
        t = t + t.swapaxes(0, 2)
        env = CTMEnvironment1x1Rotsymm(c, t)
    elif env == "ones":
        env = CTMEnvironment1x1Rotsymm(np.ones(chi), np.ones((chi, D, chi)))
    elif env is None:
        D2 = D / int(np.sqrt(D))
        delta = np.identity(D / D2)[:D2].reshape(D)
        t = np.einsum(a, [3,2,1,0], delta, [3])
        u,c,_ = np.linalg.svd(np.einsum(t, [2,0,1], delta, [2]))
        t = tdot(tdot(u, t, [0,0]), u.conj(), [2,1])
        env = CTMEnvironment1x1Rotsymm(c, t)
    
    for it in xrange(max_iterations):
        tmp = env.t * env.c
        tmp = tdot(env.t, tmp, [0,2])
        tmp = tdot(tmp, a, [[0,3],[0,3]]).swapaxes(1,2)
        chi0 = tmp.shape[0]
        u,env.c,_ = svd(tmp.reshape(chi0*D, chi0*D))
        env.c = env.c[:chi]
        u = u[:,:chi]
        env.c /= env.c[0]
        
        tmp = tdot(u.reshape(chi0, D, len(env.c)), env.t, [0,0])
        tmp = tdot(tmp, a, [[0,2],[2,3]]).reshape(len(env.c), chi0*D, D)
        env.t = tdot(tmp, u.conj(), [1,0])
        env.t /= np.max(np.abs(env.t))
        
        if it % iteration_bunch == 0:
            if tester.test(env):
                converged = True
                break

    if not converged:
        sys.stderr.write("[ctmrg_1x1_rotsymm] warning: did not converge within {:d} iterations!\n".format(max_iterations))

    if verbose:
        print "[ctmrg_1x1_rotsymm] needed {:d} iterations and {:f} seconds".format(it+1, time()-t0)    
    
    return env

class CTMEnvironment1x1Hermitian:
    def __init__(self, c, t1, t2):
        self.c = c
        self.t1 = t1
        self.t2 = t2
        
    def get_site_environment(self, site=0):
        D, chi = self.t1.shape[1:]
        tmp1 = tdot(self.c, self.t1, [1,0])
        tmp2 = tdot(self.c.conj().T, self.t2.conj(), [1,0])
        tmp3 = tdot(self.c, self.t1.conj(), [1,0])
        tmp4 = tdot(self.c.conj().T, self.t2, [1,0])
        tmp1 = tdot(tmp1, tmp2, [2,0])
        tmp2 = tdot(tmp3, tmp4, [2,0])
        return OneSiteEnvironment(tdot(tmp1, tmp2, [[0,3],[3,0]]))
    
    def get_bond_environment(self, site=0):
        e3 = tdot(tdot(self.c.conj().T, self.t2.conj(), [1,0]), self.c, [2,0])
        e6 = tdot(tdot(self.c.conj().T, self.t2, [1,0]), self.c, [2,0])
        return BondEnvironment(self.t1, self.t1, e3, self.t1, self.t1, e6)

def ctmrg_1x1_hermitian(a, chi, env=None, tester=None, max_iterations=100000, iteration_bunch=1, verbose=False):
    t0 = time()
    D = a.shape[0]
    
    if env == "random" or env is None:
        c = np.random.rand(chi, chi)
        t1 = np.random.rand(chi, D, chi)
        t2 = np.random.rand(chi, D, chi)
        env = CTMEnvironment1x1Hermitian(c, t1, t2)
    elif env == "ones":
        c = np.ones((chi, chi))
        t1 = t2 = np.ones((chi, D, chi))
        env = CTMEnvironment1x1Hermitian(c, t1, t2)
    elif env.c.shape[0] != chi:
        pass # todo: chi-upscaling

    converged = False

    for it in xrange(max_iterations):
        u, s, v = svd(tdot(env.c, env.t1, [1,0]).reshape(chi*D, chi), full_matrices=False)
        env.c = (s * v.T).T
        env.t2 = tdot(tdot(tdot(u.conj().reshape(chi, D, chi), env.t2, [0,0]), a, [[0,2],[2,3]]).reshape(chi, chi*D, D), u, [1,0])
        
        u, s, v = svd(tdot(env.t2, env.c, [2,0]).swapaxes(1,2).reshape(chi, chi*D), full_matrices=False)
        env.c = u * s
        env.t1 = tdot(tdot(tdot(v.reshape(chi, chi, D), env.t1, [1,0]), a, [[2,1],[0,3]]).reshape(chi, chi*D, D), v.conj(), [1,1])
        
        env.c /= np.max(np.abs(env.c))
        env.t1 /= np.max(np.abs(env.t1))
        env.t2 /= np.max(np.abs(env.t2))
        
        if it % iteration_bunch == 0:
            if tester.test(env):
                converged = True
                break
                
    if not converged:
        sys.stderr.write("[ctmrg_1x1_hermitian] warning: did not converge within {:d} iterations!\n".format(max_iterations))

    if verbose:
        print "[ctmrg_1x1_hermitian] needed {:d} iterations and {:f} seconds".format(it+1, time()-t0)    
        
    return env

