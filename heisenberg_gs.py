import numpy as np

import tebd
import gates
import peps
import util

import sys
import os

J = int(sys.argv[1])
D = int(sys.argv[2])
chi = int(sys.argv[3])
tau = float(sys.argv[4])
maxiterations = int(sys.argv[5])
statefile = sys.argv[6]

def test_fct(lut):
    def test_fct_impl(a, A):
        n = len(a)
        X = [None] * n
        Y = [None] * n
        Z = [None] * n
        for j in xrange(n):
            X[j] = peps.make_double_layer(a[j], o=gates.sigmax)
            Y[j] = peps.make_double_layer(a[j], o=gates.sigmay)
            Z[j] = peps.make_double_layer(a[j], o=gates.sigmaz)
        def test_fct_impl2(e):
            cxx = 0
            cyy = 0
            czz = 0
            for j in xrange(n):
                e2 = e.get_bond_environment_horizontal(j)
                norm = e2.contract(A[j], A[lut[j,1,0]])
                cxx += e2.contract(X[j], X[lut[j,1,0]]) / norm
                cyy += e2.contract(Y[j], Y[lut[j,1,0]]) / norm
                czz += e2.contract(Z[j], Z[lut[j,1,0]]) / norm
                e2 = e.get_bond_environment_vertical(j)
                norm = e2.contract(A[j].transpose([1,2,3,0]), A[lut[j,0,1]].transpose([1,2,3,0]))
                cxx += e2.contract(X[j].transpose([1,2,3,0]), X[lut[j,0,1]].transpose([1,2,3,0])) / norm
                cyy += e2.contract(Y[j].transpose([1,2,3,0]), Y[lut[j,0,1]].transpose([1,2,3,0])) / norm
                czz += e2.contract(Z[j].transpose([1,2,3,0]), Z[lut[j,0,1]].transpose([1,2,3,0])) / norm
            return np.real(J * (cxx + cyy + czz) / n)
        return test_fct_impl2
    return test_fct_impl

basepath = "output_heisenberg/"
if os.path.isfile(basepath + statefile):
    a, nns = peps.load(basepath + statefile, dtype=complex)
    lut = util.build_lattice_lookup_table(nns, [4,4])
else:
    np.random.seed(523451109)
#    a = [peps.get_state_random_rotsymm(2, D) + 1j*peps.get_state_random_rotsymm(2, D)] * 2
    
    #a = [peps.get_state_fm0(D)] * 2
    #for j in xrange(2):
    #    a[j] += 1e-2 * peps.get_state_random(2, D)
    
    a = [peps.get_state_ising(0.1)] * 2
    
    lut = util.build_lattice_lookup_table([[1,0],[1,0]], [4,4])

g1 = []
expxx = gates.exp_sigmax_sigmax(-J*tau)
expyy = gates.exp_sigmay_sigmay(-J*tau)
expzz = gates.exp_sigmaz_sigmaz(-J*tau)
g2 = [(0, 0, expxx), (0, 1, expxx), (1, 0, expxx), (1, 1, expxx),
      (0, 0, expyy), (0, 1, expyy), (1, 0, expyy), (1, 1, expyy),
      (0, 0, expzz), (0, 1, expzz), (1, 0, expzz), (1, 1, expzz)]

logfilename = "imtimeev_E_J={:d}_D={:d}_chi={:d}_tau={:.0e}.dat".format(J, D, chi, tau)
logfile = open(basepath + logfilename, "a")
a, env = tebd.itebd(a, lut, g1, g2, "random", err=1e-8, tebd_max_iterations=maxiterations, ctmrg_chi=chi, ctmrg_test_fct=test_fct(lut), verbose=True, logfile=logfile)
logfile.close()

peps.save(a, lut, basepath + "state_J={:d}_D={:d}_chi={:d}_tau={:.0e}.peps".format(J, D, chi, tau))

