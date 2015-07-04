import numpy as np

import tebd
import gates
import peps
import util

import sys
import os

J = -1
D = int(sys.argv[1])
chi = int(sys.argv[2])
tau = float(sys.argv[3])
maxiterations = int(sys.argv[4])
statefile = sys.argv[5]

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
                norm = e2.contract(A[j], A[lut[j,0,1]])
                cxx += e2.contract(X[j], X[lut[j,0,1]]) / norm
                cyy += e2.contract(Y[j], Y[lut[j,0,1]]) / norm
                czz += e2.contract(Z[j], Z[lut[j,0,1]]) / norm
            return np.real(J * (cxx + cyy + czz) / (2*n))
        return test_fct_impl2
    return test_fct_impl

basepath = "output_heisenberg/"
if os.path.isfile(basepath + statefile):
    a, nns = peps.load(basepath + statefile)
    lut = util.build_lattice_lookup_table(nns, [4,4])
else:
    #a = [peps.get_state_random_rotsymm(2, D, complex)] * 2
    a = [peps.get_state_random_rotsymm(2, D) + 1j*peps.get_state_random_rotsymm(2, D)] * 2
    lut = util.build_lattice_lookup_table([[1,0],[1,0]], [4,4])

g1 = []
expxx = gates.exp_sigmax_sigmax(-J*tau)
expyy = gates.exp_sigmay_sigmay(-J*tau)
expzz = gates.exp_sigmaz_sigmaz(-J*tau)
g2 = [(0, 0, expxx), (0, 1, expxx), (1, 0, expxx), (1, 1, expxx),
      (0, 0, expyy), (0, 1, expyy), (1, 0, expyy), (1, 1, expyy),
      (0, 0, expzz), (0, 1, expzz), (1, 0, expzz), (1, 1, expzz)]

logfilename = "imtimeev_E_D={:d}_chi={:d}_tau={:.0e}.dat".format(D, chi, tau)
logfile = open(basepath + logfilename, "a")
a, env = tebd.itebd(a, lut, g1, g2, "random", err=1e-8, tebd_max_iterations=maxiterations, ctmrg_chi=chi, ctmrg_test_fct=test_fct(lut), verbose=True, logfile=logfile)
logfile.close()

peps.save(a, lut, basepath + "state_D={:d}_chi={:d}_tau={:.0e}.peps".format(D, chi, tau))

