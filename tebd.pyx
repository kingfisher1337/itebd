import numpy as np
from numpy import dot
from numpy import tensordot as tdot
from numpy.linalg import qr
from numpy.linalg import svd
from numpy.linalg import eigh
import ctm
from time import time
from util import PeriodicArray
import peps
import sys

def _renormalise_entries(a):
    x = np.max(np.abs(a))
    if x != 0:
        a /= x

def _build_fu_env(D, be, X, Y):
    tmp = tdot(be.e5, be.e6, [2,0]).reshape(be.chi45, D, D, D, D, be.chi61)
    tmp = tdot(tmp, X, [[1,3],[1,2]])
    tmp = tdot(tmp, X.conj(), [[1,2],[1,2]])
    tmp = tdot(tmp, be.e1.reshape(be.chi61, D, D, be.chi12), [[1,2,4],[0,1,2]])
    tmp2 = tdot(be.e2, be.e3, [2,0]).reshape(be.chi12, D, D, D, D, be.chi34)
    tmp2 = tdot(tmp2, Y, [[1,3],[0,1]])
    tmp2 = tdot(tmp2, Y.conj(), [[1,2],[0,1]])
    tmp2 = tdot(tmp2, be.e4.reshape(be.chi34, D, D, be.chi45), [[1,2,4],[0,1,2]])
    return tdot(tmp, tmp2, [[0,3],[0,3]])

def _itebd_step(a, lut, g, j, orientation, env, chi, verbose, cost_err, cost_max_iterations):
    p, D = a[0].shape[:2]
    
    # apply gate to bond
    # --> between j and j+(1,0) if orientation == 0
    # --> between j and j+(0,1) if orientation == 1
    kappa = np.min([p*D, D**3])
    if orientation == 0:
        X, a2 = qr(a[j].transpose([1,3,4,0,2]).reshape(D**3, p*D))
        Y, b2 = qr(a[lut[j,1,0]].transpose([1,2,3,0,4]).reshape(D**3, p*D))
    else:
        X, a2 = qr(a[j].transpose([2,4,1,0,3]).reshape(D**3, p*D))
        Y, b2 = qr(a[lut[j,0,1]].transpose([2,3,4,0,1]).reshape(D**3, p*D))
    X = X.reshape(D, D, D, kappa)
    a2 = a2.reshape(kappa, p, D)
    Y = Y.reshape(D, D, D, kappa)
    b2 = b2.reshape(kappa, p, D)

    a3,tmp,b3 = svd(tdot(tdot(a2, g, [1,0]), b2, [[1,2],[2,1]]).swapaxes(2,3).reshape([p*kappa]*2))
    tmp = np.sqrt(tmp[:D])
    a3 = (a3[:,:D] * tmp).reshape(kappa, p, D)
    b3 = (b3[:D].T * tmp).reshape(kappa, p, D)
    
    if orientation == 0:
        be = env.get_bond_environment_horizontal(j)
    else:
        be = env.get_bond_environment_vertical(j)
    e = _build_fu_env(D, be, X, Y)
    
    # environment gauge fixing (see arXiv:1503.05345v1)
    #e = e.swapaxes(1,2).reshape([kappa**2]*2)
    #e = 0.5 * (e + e.T.conj())
    #s, w = eigh(e)
    #print "[_itebd_step] norm svals: ", s
    #idx = np.nonzero(s.clip(min=0))[0]
    #s = s[idx]
    #w = w[:,idx]
    #w = w * np.sqrt(s)
    #e = dot(w, w.T.conj())
    #e = e.reshape([kappa]*4).swapaxes(1,2)
    
    cost2 = np.inf
    converged = False
    for _ in xrange(cost_max_iterations):
        tmp = tdot(tdot(e, tdot(a2, b2, [2,2]), [[0,2],[0,2]]), g, [[2,3],[0,1]])
        
        S = tdot(tmp, b3.conj(), [[3,1],[1,0]]).reshape(kappa*p*D)
        M = tdot(tdot(e, b3, [2,0]), b3.conj(), [[2,3],[0,1]])
        M = tdot(M, np.identity(p), [[],[]]).transpose([1,5,3,0,4,2]).reshape([kappa*p*D]*2)
        a3vec = np.linalg.lstsq(M, S)[0]
        a3vec /= np.max(np.abs(a3vec))
        a3 = a3vec.reshape(kappa, p, D)
        
        S = tdot(tmp, a3.conj(), [[0,2],[0,1]]).reshape(kappa*p*D)
        M = tdot(tdot(e, a3, [0,0]), a3.conj(), [[0,3],[0,1]])
        M = tdot(M, np.identity(p), [[],[]]).transpose([1,4,3,0,5,2]).reshape([kappa*p*D]*2)
        b3vec = np.linalg.lstsq(M, S)[0]
        b3vec /= np.max(np.abs(b3vec))
        b3 = b3vec.reshape(kappa, p, D)
        
        cost = dot(b3vec.conj(), dot(M, b3vec)) - 2 * np.real(dot(b3vec.conj(), S))
        err = np.abs(cost2 - cost)
        if err < cost_err:
            converged = True
            break
        cost2 = cost
        
    if not converged:
        sys.stderr.write("[_itebd_step] warning: cost function did not converge! cost err is {:e}\n".format(err))
    
    if orientation == 0:
        return tdot(a3, X, [0,3]).swapaxes(1,2), tdot(b3, Y, [0,3]).transpose([0,2,3,4,1])
    else:
        return tdot(a3, X, [0,3]).transpose([0,4,2,1,3]), tdot(b3, Y, [0,3])
        


def itebd(
    a, lut,
    g1, g2, 
    env=None, 
    err=1e-7, tebd_max_iterations=10000,
    ctmrg_chi=16, ctmrg_max_iterations=10000,
    ctmrg_test_fct=None, ctmrg_err=1e-12,
    verbose=False, logfile=None):
    
    t0 = time()
    
    if ctmrg_err > err:
        sys.stderr.write("[itebd] warning: ctmrg_err > itebd_err!\n")
    
    n = len(a)
    A = [None] * n
    vals = np.array([1e100] * 10)
    
    for it in xrange(tebd_max_iterations):
        for j in xrange(n):
            A[j] = peps.make_double_layer(a[j])
        tester = ctm.CTMRGGenericTester(ctmrg_test_fct(a, A), ctmrg_err, verbose)
        env = ctm.ctmrg(A, lut, ctmrg_chi, env, tester, ctmrg_max_iterations, verbose)
        
        nval = tester.get_value()
        valerr = np.max(np.abs(1 - nval / vals))
        vals[it % len(vals)] = nval
        
        if logfile is not None:
            logfile.write("{:.15e} {:.15e} {:f}\n".format(nval, valerr, time()-t0))
            logfile.flush()
            sys.stdout.flush()
        if verbose:
            print "[itebd] testval (it {:d}): {:e} +/- {:e}".format(it, nval, valerr)
        if valerr < err:
            break

        t1 = time()
        
#        for (j, g) in g1:
#            a[j] = tdot(g, a[j], [1,0])
#            a[j] /= np.max(np.abs(a[j]))
#        
#        if len(g1) > 0:
#            for k in xrange(n):
#                A[k] = peps.make_double_layer(a[k])
#            tester = ctm.CTMRGGenericTester(ctmrg_test_fct(a, A), ctmrg_err, verbose)
#            env = ctm.ctmrg(A, lut, ctmrg_chi, env, tester, ctmrg_max_iterations, verbose)
        
        for (j, orientation, g) in g2:
#            for k in xrange(n):
#                A[k] = peps.make_double_layer(a[k])
#            tester = ctm.CTMRGGenericTester(ctmrg_test_fct(a, A), ctmrg_err, verbose)
#            env = ctm.ctmrg(A, lut, ctmrg_chi, env, tester, ctmrg_max_iterations, verbose)
#            a[j], a[lut[j,1,0] if orientation == 0 else lut[j,0,1]] = _itebd_step(
#                a, lut, g, j, orientation, env, ctmrg_chi, verbose, 1e-12, 100000)
            
            anew, bnew = _itebd_step(a, lut, g, j, orientation, env, ctmrg_chi, verbose, 1e-12, 100000)
            
            ctm.ctmrg_post_tebd(A, lut, anew, bnew, j, orientation, ctmrg_chi, env)
            a[j] = anew
            A[j] = peps.make_double_layer(anew)
            j2 = lut[j,1,0] if orientation == 0 else lut[j,0,1]
            a[j2] = bnew
            A[j2] = peps.make_double_layer(bnew)
            
        for (j, g) in g1:
            a[j] = tdot(g, a[j], [1,0])
            a[j] /= np.max(np.abs(a[j]))
            
        #for j in xrange(n):
        #    a[j] = tdot(g1, a[j], [1,0])
        #    a[j] /= np.max(np.abs(a[j]))
        
        #for j in xrange(n):
        #    for orientation in [0,1]:
        #        for k in xrange(n):
        #            A[k] = peps.make_double_layer(a[k])
        #        tester = ctm.CTMRGGenericTester(ctmrg_test_fct(a, A), ctmrg_err, verbose)
        #        env = ctm.ctmrg(A, lut, ctmrg_chi, env, tester, ctmrg_max_iterations, verbose)
        #        _itebd_step(a, lut, g2, j, orientation, env, ctmrg_chi, verbose, 1e-12, 100000)
        
        #for j in xrange(n):
        #    a[j] = tdot(g1, a[j], [1,0])
        #    a[j] /= np.max(np.abs(a[j]))
        
        if verbose:
            print "[itebd] full update needed {:f} seconds".format(time()-t1)
        

    if verbose:
        print "[itebd] needed {:f} seconds".format(time()-t0)

    return a, env
    
