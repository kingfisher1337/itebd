import numpy as np
import tebd
import gates
import sys
import peps
import util
import os

from time import time
t0 = time()

D = int(sys.argv[1])
chi = int(sys.argv[2])
h = float(sys.argv[3])
tau = float(sys.argv[4])
tebderr = float(sys.argv[5])
maxiterations = int(sys.argv[6])
statefile = sys.argv[7]
fast_full_update = "-ffu" in sys.argv
trotter_second_order = "-trotter2" in sys.argv

basepath = "output_tfi_fixpoint/"
basepath_varpeps = "output_varpeps_tfi/"

sys.stdout = open(basepath + "log_tfi_gs_2d_D={:d}_chi={:d}_h={:f}_tau={:.0e}.txt".format(D, chi, h, tau), "a")
sys.stderr = open(basepath + "err_tfi_gs_2d_D={:d}_chi={:d}_h={:f}_tau={:.0e}.txt".format(D, chi, h, tau), "a")

def test_fct(a, A):
    n = len(a)
    X = map(lambda b: peps.make_double_layer(b, o=gates.sigmax), a)
    Z = map(lambda b: peps.make_double_layer(b, o=gates.sigmaz), a)
    AT = map(lambda B: B.transpose([1,2,3,0]), A)
    ZT = map(lambda B: B.transpose([1,2,3,0]), Z)
    
    def test_fct_impl(e):
        mx = 0
        mz = 0
        czz = 0
        S = 0 # some pseudo entanglement entropy
        for j in xrange(n):
            e1 = e.get_site_environment(j)
            norm = e1.contract(A[j])
            mx += e1.contract(X[j]) / norm
            mz += e1.contract(Z[j]) / norm
            
            e2 = e.get_bond_environment_horizontal(j)
            norm = e2.contract(A[j], A[lut[j,1,0]])
            czz += e2.contract(Z[j], Z[lut[j,1,0]]) / norm
            
            e2 = e.get_bond_environment_vertical(j)
            norm = e2.contract(AT[j], AT[lut[j,0,1]])
            czz += e2.contract(ZT[j], ZT[lut[j,0,1]]) / norm
            
            w1 = np.dot(e.c1[j], e.c2[lut[j,1,0]])
            w2 = np.dot(e.c3[lut[j,1,1]], e.c4[lut[j,0,1]])
            xi = np.linalg.svd(np.dot(w1, w2))[1]
            xi = xi[np.nonzero(xi)]
            S += np.dot(xi**2, np.log(xi))
            
        E = (-czz - h*mx) / n
        mz /= n
        return [mz, S, E]
        
    return test_fct_impl

if trotter_second_order:
    g1 = gates.exp_sigmax(0.5*tau*h)
else:
    g1 = gates.exp_sigmax(tau*h)
g1 = [(0, g1), (1, g1)]
g2 = gates.exp_sigmaz_sigmaz(tau)
g2 = [(0, 0, g2), (0, 1, g2), (1, 0, g2), (1, 1, g2)]

#a, nns = peps.load(basepath_varpeps + statefile)
#lut = util.build_lattice_lookup_table(nns, [4,4])
a = peps.load(basepath_varpeps + statefile)[0][0]
a = [a, a]
lut = util.build_lattice_lookup_table([[1,0],[1,0]], [4,4])

logfilename = "imtimeev{:s}_E_D={:d}_chi={:d}_h={:e}_tau={:.0e}.dat".format("_2nd" if trotter_second_order else "", D, chi, h, tau)
logfile = open(basepath + logfilename, "a")
a, env = tebd.itebd(a, lut, g1, g2, "random", err=tebderr, tebd_max_iterations=maxiterations, ctmrg_chi=chi, ctmrg_test_fct=test_fct, verbose=True, logfile=logfile, fast_full_update=fast_full_update, apply_g1_twice=trotter_second_order)
logfile.close()

peps.save(a, lut, basepath + "state_D={:d}_chi={:d}_h={:e}_tau={:.0e}.peps".format(D, chi, h, tau))


