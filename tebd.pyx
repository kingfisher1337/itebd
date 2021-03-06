import numpy as np
from numpy import dot
from numpy import tensordot as tdot
from numpy.linalg import qr
from numpy.linalg import svd
from numpy.linalg import eigh
import ctm
from time import time
import peps
import sys
from scipy.optimize import minimize

def _build_fu_env(be, X, Y):
    D1, D5, D6, k1 = X.shape
    D2, D3, D4, k2 = Y.shape
    tmp = tdot(be.e5, be.e6, [2,0]).reshape(be.chi45, D5, D5, D6, D6, be.chi61)
    tmp = tdot(tmp, X, [[1,3],[1,2]])
    tmp = tdot(tmp, X.conj(), [[1,2],[1,2]])
    tmp = tdot(tmp, be.e1.reshape(be.chi61, D1, D1, be.chi12), [[1,2,4],[0,1,2]])
    tmp2 = tdot(be.e2, be.e3, [2,0]).reshape(be.chi12, D2, D2, D3, D3, be.chi34)
    tmp2 = tdot(tmp2, Y, [[1,3],[0,1]])
    tmp2 = tdot(tmp2, Y.conj(), [[1,2],[0,1]])
    tmp2 = tdot(tmp2, be.e4.reshape(be.chi34, D4, D4, be.chi45), [[1,2,4],[0,1,2]])
    return tdot(tmp, tmp2, [[0,3],[3,0]])

def _my_pinv(a, ratio=1e12):
    u, s, v = svd(a)
    s[s < (s[0]/ratio)] = 0 # keep only the largest 12 orders of magnitude
    k = np.count_nonzero(s) # truncate singular values equal to zero
    return dot(v.T.conj() * (1.0 / s), u.T.conj())

def _fix_env_local_gauge(e):
    # --> PRB 90, 064425 (2014) or arXiv:1503.05345v1
    # take the positive approximant
    
    #I = np.identity(e.shape[0])
    #return e, I, I, I, I
    
    k = e.shape[0]
    e = e.swapaxes(1,2).reshape([k**2]*2)
    e = 0.5 * (e + e.T.conj())
    s, w = eigh(e)
    idx = np.nonzero(s.clip(min=0))[0]
    k2 = len(idx)
    s = s[idx]
    w = w[:,idx]
    w = w * np.sqrt(s)
    
    #I = np.identity(k)
    #return np.dot(w, w.T.conj()).reshape(k, k, k, k).swapaxes(1, 2), I, I, I, I
    
    # and fix the local gauge degrees of freedom
    w = w.reshape(k, k, k2)
    q, r = qr(w.transpose([0,2,1]).reshape(k*k2, k))
    _, l = qr(w.transpose([1,2,0]).reshape(k*k2, k))
    #rinv = np.linalg.pinv(r)
    #linv = np.linalg.pinv(l)
    rinv = np.linalg.pinv(r)
    linv = np.linalg.pinv(l)
    w = dot(linv.T, q.reshape(k, k2*k)).reshape(k, k2, k).swapaxes(1, 2).reshape(k*k, k2)
    e = dot(w, w.T.conj()).reshape(k, k, k, k).swapaxes(1, 2)
    return e, r, l, rinv, linv

def _fix_bond_gauge(a, b):
    k, p, D = a.shape
    q1, r1 = qr(a.reshape(k*p, D))
    q2, r2 = qr(b.reshape(k*p, D))
    u, s, v = svd(dot(r1, r2.T))
    s = np.sqrt(s)
    a = (dot(q1, u) * s).reshape(k, p, D)
    b = (dot(q2, v.T) * s).reshape(k, p, D)
    return a, b

def _full_update(e, a2, b2, g, a3, b3, gauge):
    cost_err = 1e-15
    cost_max_iterations = 100
    
    kappa, p, D = a2.shape
    if gauge:
        e, r, l, rinv, linv = _fix_env_local_gauge(e)
        a2 = dot(l, a2.reshape(kappa, p*D)).reshape(kappa, p, D)
        b2 = dot(r, b2.reshape(kappa, p*D)).reshape(kappa, p, D)
        a3 = dot(l, a3.reshape(kappa, p*D)).reshape(kappa, p, D)
        b3 = dot(r, b3.reshape(kappa, p*D)).reshape(kappa, p, D)
    
    cost = []
    err = []
    abserr = []
    converged = False
    for it in xrange(cost_max_iterations):
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
        
        ncost = dot(b3vec.conj(), dot(M, b3vec)) - 2 * np.real(dot(b3vec.conj(), S))
        nerr = np.inf if len(cost) == 0 else np.abs(1.0 - ncost / cost[-1])
        nabserr = np.inf if len(cost) == 0 else np.abs(ncost - cost[-1])
        cost.append(ncost)
        err.append(nerr)
        abserr.append(err)
        
        if len(err) >= 10 and nerr < cost_err:
            converged = True
            print "[_itebd_bond_update] needed {:d} ALS sweeps; error is {:e}".format(it, err[-1])
            break
    
    if not converged:
        sys.stderr.write("[_full_update] warning: cost function did not converge within {:d} iterations! cost relerr is {:e}\n".format(cost_max_iterations, nerr))
    
    if gauge:
        a3 = dot(linv, a3.reshape(kappa, p*D)).reshape(kappa, p, D)
        b3 = dot(rinv, b3.reshape(kappa, p*D)).reshape(kappa, p, D)
    
    a3 /= np.max(np.abs(a3))
    b3 /= np.max(np.abs(b3))
    
    a3, b3 = _fix_bond_gauge(a3, b3)
    
    return a3, b3

def _gradient_full_update(e, a2, b2, g, a, b):
    k, p, D = a.shape
    size = k*p*D
    
    e = e.swapaxes(1, 2).reshape(k**2, k**2)
    ab2 = tdot(a2, b2, [2,2]).swapaxes(1, 2).reshape(k**2, p**2)
    e2 = dot(e.T, ab2).reshape(k**2*p**2)

    def cost(x):
        a3, b3 = np.split(x, 2)
        a3, b3 = a3.reshape(k, p, D), b3.reshape(k, p, D)
        ab3 = tdot(a3, b3, [2,2]).swapaxes(1, 2).reshape(k**2, p**2)
        z = dot(dot(e.T, ab3).reshape(k**2*p**2), ab3.conj().reshape(k**2*p**2))
        y = dot(e2, ab3.conj().reshape(k**2*p**2))
        return z - 2 * y
    
    res = minimize(cost, np.concatenate([a.reshape(size), b.reshape(size)]), method="BFGS", options={"gtol":1e-10, "disp":True})
    a, b = np.split(res.x, 2)
    return a.reshape(k, p, D), b.reshape(k, p, D)

def _itebd_step(a, lut, g, j, orientation, env, mode):
    p, D = a[0].shape[:2]
    
    # apply gate to bond
    # --> between j and j+(1,0) if orientation == 0
    # --> between j and j+(0,1) if orientation == 1
    
    # modes allowed:
    # "fu" --> full update using ALS (alternatig least square) sweeps (default)
    # "su" --> simple update
    # "gfu" --> gradient full update
    
    if orientation == 0:
        X, a2 = qr(a[j].transpose([1,3,4,0,2]).reshape(D**3, p*D))
        Y, b2 = qr(a[lut[j,1,0]].transpose([1,2,3,0,4]).reshape(D**3, p*D))
    else:
        X, a2 = qr(a[j].transpose([2,4,1,0,3]).reshape(D**3, p*D))
        Y, b2 = qr(a[lut[j,0,1]].transpose([2,3,4,0,1]).reshape(D**3, p*D))
    
    kappa = np.min([p*D, D**3])
    X = X.reshape(D, D, D, kappa)
    a2 = a2.reshape(kappa, p, D)
    Y = Y.reshape(D, D, D, kappa)
    b2 = b2.reshape(kappa, p, D)

    # simple update for initialising a3 and b3
    a3,tmp,b3 = svd(tdot(tdot(a2, g, [1,0]), b2, [[1,2],[2,1]]).swapaxes(2,3).reshape([p*kappa]*2))
    tmp = np.sqrt(tmp[:D])
    a3 = (a3[:,:D] * tmp).reshape(kappa, p, D)
    b3 = (b3[:D].T * tmp).reshape(kappa, p, D)
    
    if mode != "su":
        if orientation == 0:
            be = env.get_bond_environment_horizontal(j)
        else:
            be = env.get_bond_environment_vertical(j)
        e = _build_fu_env(be, X, Y)
    
    if mode == "fu":
        a3, b3 = _full_update(e, a2, b2, g, a3, b3, True)
    elif mode == "funogauge":
        a3, b3 = _full_update(e, a2, b2, g, a3, b3, False)
    elif mode == "gfu":
        a3, b3 = _gradient_full_update(e, a2, b2, g, a3, b3)
    
    if orientation == 0:
        return tdot(a3, X, [0,3]).swapaxes(1,2), tdot(b3, Y, [0,3]).transpose([0,2,3,4,1])
    else:
        return tdot(a3, X, [0,3]).transpose([0,4,2,1,3]), tdot(b3, Y, [0,3])


def _apply_one_body_gate(a, g):
    s = a.shape
    a = dot(g, a.reshape(s[0], s[1]*s[2]*s[3]*s[4])).reshape(s)
    return a / np.max(np.abs(a))

class CTMRGEnvContractor:
    def __init__(self, lut, chi, test_fct, relerr, abserr, max_iterations_per_update=1000, ctmrg_verbose=False, tester_verbose=False, tester_checklen=10, plotonfail=False, e="random"):
        self.lut = lut
        self.chi = chi
        self.e = e
        self.test_fct = test_fct
        self.relerr = relerr
        self.abserr = abserr
        self.max_iterations_per_update = max_iterations_per_update
        self.ctmrg_verbose = ctmrg_verbose
        self.tester_verbose = tester_verbose
        self.tester_checklen = tester_checklen
        self.test_values = None
        self.plotonfail = plotonfail
    
    def update(self, a):
        A = map(peps.make_double_layer, a)
        tester = ctm.CTMRGTester(self.test_fct(a, A), self.relerr, self.abserr, self.tester_checklen, self.tester_verbose)
        self.e = ctm.ctmrg(A, self.lut, self.chi, self.e, tester, self.max_iterations_per_update, verbose=self.ctmrg_verbose)
        self.test_values = tester.get_value()
        
        if self.plotonfail and not tester.is_converged():
            self.plot_ctmrg_stat(tester.get_values(), tester.get_errors(), tester.get_abserrors())
    
    def get_environment(self):
        return self.e
    
    def get_test_function(self):
        return self.test_fct
    
    def get_test_values(self):
        return self.test_values
    
    def plot_ctmrg_stat(self, vals, relerrs, abserrs):
        import matplotlib.pyplot as plt
        plt.subplot(311)
        for j in xrange(len(vals[0])):
            plt.plot(map(lambda v: v[j], vals), label=str(j))
        plt.subplot(312)
        for j in xrange(len(relerrs[0])):
            plt.plot(map(lambda v: v[j], relerrs), label=str(j))
        plt.legend(loc="best")
        plt.yscale("log")
        plt.subplot(313)
        for j in xrange(len(abserrs[0])):
            plt.plot(map(lambda v: v[j], abserrs), label=str(j))
        plt.legend(loc="best")
        plt.yscale("log")
        plt.show()
    
    def clone(self):
        ec = CTMRGEnvContractor(self.lut, self.chi, self.test_fct, self.relerr, self.abserr, self.max_iterations_per_update, self.ctmrg_verbose, self.tester_verbose, self.tester_checklen, self.plotonfail, self.e.clone())
        ec.test_values = np.copy(self.test_values)
        return ec

class CTMRGEnvContractorFactory:
    def __init__(self, lut, chi, test_fct, relerr, abserr, max_iterations_per_update=1000, ctmrg_verbose=False, tester_verbose=False, tester_checklen=10, plotonfail=False, e="random"):
        self.lut = lut
        self.chi = chi
        self.test_fct = test_fct
        self.relerr = relerr
        self.abserr = abserr
        self.max_iterations_per_update = max_iterations_per_update
        self.ctmrg_verbose = ctmrg_verbose
        self.tester_verbose = tester_verbose
        self.tester_checklen = tester_checklen
        self.plotonfail = plotonfail
        if e is None or type(e) is str:
            self.e = e
        else:
            self.e = e.clone()
    def create(self):
        return CTMRGEnvContractor(self.lut, self.chi, self.test_fct, self.relerr, self.abserr, self.max_iterations_per_update, self.ctmrg_verbose, self.tester_verbose, self.tester_checklen, self.plotonfail, self.e)

def _apply_two_body_gate_su_helper(a, b, g, x1, x2, x3, x4, x5, x6, x7):
    D = x7.shape[0]
    sa, sb = a.shape, b.shape
    
    a2 = a * x6
    a2 = a2.swapaxes(1,4) * x1
    a2 = a2.swapaxes(3,4) * x5
    a2 = a2.swapaxes(2,4) * x7
    b2 = b.swapaxes(1,4) * x2
    b2 = b2.swapaxes(2,4) * x3
    b2 = b2.swapaxes(3,4) * x4
    b2 = b2.swapaxes(0,1)
    
    a2 = a2.reshape(sa[0]*sa[4]*sa[3]*sa[1], sa[2])
    b2 = b2.reshape(sb[4], sb[0]*sb[1]*sb[2]*sb[3])
    ab = np.dot(a2, b2)
    ab = ab.reshape(sa[0], sa[4]*sa[3]*sa[1], sb[0], sb[1]*sb[2]*sb[3])
    ab = ab.transpose([1,3,0,2]).reshape(sa[4]*sa[3]*sa[1]*sb[1]*sb[2]*sb[3], sa[0]*sb[0])
    ab = np.dot(ab, g.reshape([sa[0]*sb[0]]*2))
    ab = ab.reshape(sa[4]*sa[3]*sa[1], sb[1]*sb[2]*sb[3], sa[0], sb[0]).transpose([2,0,3,1]).reshape(sa[0]*sa[4]*sa[3]*sa[1], sb[0]*sb[1]*sb[2]*sb[3])
    
    u,s,v = svd(ab)
    s = s[:D]
    a2 = u[:,:D].reshape(sa[0], sa[4], sa[3], sa[1], sa[2])
    a2 = a2.swapaxes(2,4) * (1.0 / x5)
    a2 = a2.swapaxes(3,4) * (1.0 / x1)
    a2 = a2.swapaxes(1,4) * (1.0 / x6)
    b2 = v[:D].reshape(sb[4], sb[0], sb[1], sb[2], sb[3])
    b2 = b2.swapaxes(0,1) * (1.0 / x4)
    b2 = b2.swapaxes(3,4) * (1.0 / x3)
    b2 = b2.swapaxes(2,4) * (1.0 / x2)
    b2 = b2.swapaxes(1,4)
    
    a2 /= np.max(np.abs(a2))
    b2 /= np.max(np.abs(b2))
    s /= s[0]
    
    return a2, b2, s

def _apply_two_body_gate_su(a, x, y, lut, g, j, orientation):
    lut = lut[j]
    if orientation == 0: # apply gate on sites j and j+(1,0)
        a[j], a[lut[1,0]], x[j] = _apply_two_body_gate_su_helper(
            a[j], a[lut[1,0]], g,
            y[lut[0,-1]], y[lut[1,-1]], x[lut[1,0]], y[lut[1,0]], y[j], x[lut[-1,0]], x[j])
    else: # apply gate on sites j and j+(0,1)
        a[j], a[lut[0,1]], y[j] = _apply_two_body_gate_su_helper(
            a[j].transpose([0,2,3,4,1]), a[lut[0,1]].transpose([0,2,3,4,1]), g,
            x[j], x[lut[0,1]], y[lut[0,1]], x[lut[-1,1]], x[lut[-1,0]], y[lut[0,-1]], y[j])
        a[j] = a[j].transpose([0,4,1,2,3])
        a[lut[0,1]] = a[lut[0,1]].transpose([0,4,1,2,3])

def itebd_v2(a, lut, t0, dt, tmax, gate_callback, env_contractor, log_dir, simulation_name, backup_interval, mode="fu", log_interval=1):
    
    walltime0 = time()
    max_iterations = int(tmax / dt)
    g1pre, g2, g1post = gate_callback(dt)
    
    if not log_dir.endswith("/"):
        log_dir += "/"
    f = open(log_dir + simulation_name + "_itebd.dat", "a")
    
    if mode == "su":
        x = [np.ones(a[j].shape[2]) for j in xrange(len(a))]
        y = [np.ones(a[j].shape[3]) for j in xrange(len(a))]
        b = [None]*len(a)
        
        for it in xrange(max_iterations):
            for (j, g) in g1pre:
                a[j] = _apply_one_body_gate(a[j], g)
            
            for (j, orientation, g) in g2:
                _apply_two_body_gate_su(a, x, y, lut, g, j, orientation)
                
            for (j, g) in g1post:
                a[j] = _apply_one_body_gate(a[j], g)
            
            if (it+1) % log_interval == 0:
                try:
                    for j in xrange(len(a)):
                        b[j] = a[j] * np.sqrt(x[lut[j,-1,0]])
                        b[j] = b[j].swapaxes(1,4) * np.sqrt(y[lut[j,0,-1]])
                        b[j] = b[j].swapaxes(2,4) * np.sqrt(x[j])
                        b[j] = b[j].swapaxes(3,4) * np.sqrt(y[j])
                        b[j] = b[j].transpose([0,2,3,4,1])
                    env_contractor.update(b)
                except np.linalg.LinAlgError:
                    print "[itebd] environment contractor update failed at observable evaluation!"
                    f.close()
                    return
                
                t = t0 + (it+1)*dt
                walltime = time() - walltime0
                
                f.write("{:.15e} {:f}".format(t, walltime))
                for z in env_contractor.get_test_values():
                    f.write(" {:.15e}".format(z))
                f.write("\n")
                f.flush()
                
            if (it+1) % backup_interval == 0:
                for j in xrange(len(a)):
                    b[j] = a[j] * np.sqrt(x[lut[j,-1,0]])
                    b[j] = b[j].swapaxes(1,4) * np.sqrt(y[lut[j,0,-1]])
                    b[j] = b[j].swapaxes(2,4) * np.sqrt(x[j])
                    b[j] = b[j].swapaxes(3,4) * np.sqrt(y[j])
                    b[j] = b[j].transpose([0,2,3,4,1])
                    
                t = t0 + (it+1)*dt
                peps.save(b, lut, log_dir + simulation_name + "_state_t={0:011.6f}.peps".format(t))
                print "saved peps at t={0:011.6f}".format(t)
                
            sys.stdout.flush()
            
        for j in xrange(len(a)):
            b[j] = a[j] * np.sqrt(x[lut[j,-1,0]])
            b[j] = b[j].swapaxes(1,4) * np.sqrt(y[lut[j,0,-1]])
            b[j] = b[j].swapaxes(2,4) * np.sqrt(x[j])
            b[j] = b[j].swapaxes(3,4) * np.sqrt(y[j])
            b[j] = b[j].transpose([0,2,3,4,1])
        
    else:
        try:
            env_contractor.update(a)
        except np.linalg.LinAlgError:
            print "[itebd] environment contractor update failed on initial update!"
            f.close()
            return
        
        walltime = time() - walltime0
        f.write("{:.15e} {:f}".format(t0, walltime))
        for x in env_contractor.get_test_values():
            f.write(" {:.15e}".format(x))
        f.write("\n")
        f.flush()
        
        for it in xrange(max_iterations):
        
            for (j, g) in g1pre:
                a[j] = _apply_one_body_gate(a[j], g)
            
            for (j, orientation, g) in g2:
                try:
                    env_contractor.update(a)
                except np.linalg.LinAlgError:
                    print "[itebd] environment contractor update failed!"
                    f.close()
                    return
                    
                
                if orientation == 0:
                    a[j], a[lut[j,1,0]] = _itebd_step(a, lut, g, j, orientation, env_contractor.get_environment(), mode)
                else:
                    a[j], a[lut[j,0,1]] = _itebd_step(a, lut, g, j, orientation, env_contractor.get_environment(), mode)
            
            for (j, g) in g1post:
                a[j] = _apply_one_body_gate(a[j], g)
        
            try:
                env_contractor.update(a)
            except np.linalg.LinAlgError:
                print "[itebd] environment contractor update failed at observable evaluation!"
                f.close()
                return
            
            t = t0 + (it+1)*dt
            walltime = time() - walltime0
            
            f.write("{:.15e} {:f}".format(t, walltime))
            for x in env_contractor.get_test_values():
                f.write(" {:.15e}".format(x))
            f.write("\n")
            f.flush()
            
            if (it+1) % backup_interval == 0:
                peps.save(a, lut, log_dir + simulation_name + "_state_t={0:011.6f}.peps".format(t))
                print "saved peps at t={0:011.6f}".format(t)
            
            sys.stdout.flush()
            
        f.close()
        
        if max_iterations % backup_interval != 0:
            peps.save(a, lut, log_dir + simulation_name + "_state_t={0:011.6f}.peps".format(t0 + tmax))
            print "saved peps at t={0:011.6f}".format(tmax)

        return a





"""
def _peps_to_vec(b, size):
    return np.concatenate(map(lambda c: c.reshape(size), b))

def _vec_to_peps(x, n, shape):
    return map(lambda c: c.reshape(shape), np.split(x, n))

#from cython.parallel import prange
#from cython.parallel import threadid
#cimport openmp
import multiprocessing as mp

def _polish_cost_fct_helper(params):
    n, shape, lut, chi, test_fct, relerr, abserr, x, dx, jmin, jmax, evalf, Eidx = params
    ec = CTMRGEnvContractor(lut, chi, test_fct, relerr, abserr)
    bestvals = [1e100] * (Eidx+1 if Eidx > 0 else -Eidx)
    grad = np.empty(jmax - jmin + 2 if evalf else jmax - jmin + 1)
    for j in xrange(jmin, jmax+1):
        y = np.copy(x)
        y[j] += dx
        ec.update(_vec_to_peps(y, n, shape))
        testvals = ec.get_test_values()
        grad[j - jmin] = testvals[Eidx]
        #if testvals[Eidx] < bestvals[Eidx]:
        #    bestvals = testvals
    if evalf:
        ec.update(_vec_to_peps(x, n, shape))
        testvals = ec.get_test_values()
        grad[-1] = testvals[Eidx]
        #if testvals[Eidx] < bestvals[Eidx]:
        #    bestvals = testvals
    
    print grad[-1]
    return grad, bestvals

#def polish(a, lut, env_contractor, energy_idx=-1, pepsfilename=None):
def polish(a, lut, chi, test_fct, relerr=1e-12, abserr=1e-15, energy_idx=-1, pepsfilename=None, numworkers=1):
    cdef int m
    cdef int num_threads

    t0 = time()
    shape = a[0].shape
    size = a[0].size
    n = len(a)
    m = n * size
    dx = 1.4901161193847656e-08
    
    def cost_fct(x):
        numfctevals = len(x) + 1 # number of function evaluations
        m = numfctevals / numworkers # number of function evaluations per worker
        if m*numworkers < numfctevals:
            m += 1
        
        paramlist = [(n, shape, lut, chi, test_fct, relerr, abserr, x, dx, j*m, min((j+1)*m-1, len(x)-1), j+1 == numworkers, energy_idx) for j in xrange(numworkers)]
        print map(lambda z: (z[9], z[10], z[11]) , paramlist)
        pool = mp.Pool(processes=numworkers)
        res = pool.map(_polish_cost_fct_helper, paramlist)
        
        grad_and_f = np.concatenate(map(lambda z: z[0], res))
        grad = grad_and_f[:-1]
        E = grad_and_f[-1]
        
        return E, grad
        
    res = minimize(cost_fct, _peps_to_vec(a, size), jac=True, method="BFGS", options={"disp":True})
    print "[polish] minimize message:", res.message
    
    return _vec_to_peps(res.x, n, shape)
"""
