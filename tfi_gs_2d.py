import numpy as np
import tebd
import gates
import sys
import peps
from util import PeriodicArray
import util
import ctm
import os

from time import time
t0 = time()

def test_fct(h, lut):
    def test_fct_impl(a, A):
        n = len(a)
        X = [None] * n
        Z = [None] * n
        for j in xrange(n):
            X[j] = peps.make_double_layer(a[j], o=gates.sigmax)
            Z[j] = peps.make_double_layer(a[j], o=gates.sigmaz)
        def test_fct_impl2(e):
            mx = 0
            czz = 0
            for j in xrange(n):
                e1 = e.get_site_environment(j)
                mx += e1.contract(X[j]) / e1.contract(A[j])
                e2 = e.get_bond_environment_horizontal(j)
                czz += e2.contract(Z[j], Z[lut[j,1,0]]) / e2.contract(A[j], A[lut[j,1,0]])
                e2 = e.get_bond_environment_vertical(j)
                czz += e2.contract(Z[j], Z[lut[j,0,1]]) / e2.contract(A[j], A[lut[j,0,1]])
            return (-czz - h*mx) / n
        return test_fct_impl2
    return test_fct_impl

D = int(sys.argv[1])
chi = int(sys.argv[2])
h = float(sys.argv[3])
tau = float(sys.argv[4])
tebderr = float(sys.argv[5])
maxiterations = int(sys.argv[6])
statefile = sys.argv[7]

basepath = "output_tfi/"

if os.path.isfile(basepath + statefile):
    a, nns = peps.load(basepath + statefile)
    lut = util.build_lattice_lookup_table(nns, [4,4])
else:
    sys.stderr.write("no file \"{:s}\" found! starting new calculation".format(basepath + statefile))
    a = [None]*2
    for j in xrange(len(a)):
        #a[j] = peps.get_state_ising(1.8)
        a[j] = peps.get_state_random_rotsymm(2, D)
    lut = util.build_lattice_lookup_table([[1,0],[1,0]], [4,4])

g1 = gates.exp_sigmax(0.5*tau*h)
g1 = [(0, g1), (1, g1)]
g2 = gates.exp_sigmaz_sigmaz(tau)
g2 = [(0, 0, g2), (0, 1, g2), (1, 0, g2), (1, 1, g2)]

logfilename = "imtimeev_E_D={:d}_chi={:d}_h={:e}_tau={:e}.dat".format(D, chi, h, tau)
logfile = open(basepath + logfilename, "a")
a, env = tebd.itebd(a, lut, g1, g2, "random", err=tebderr, tebd_max_iterations=maxiterations, ctmrg_chi=chi, ctmrg_test_fct=test_fct(h, lut), verbose=True, logfile=logfile)
logfile.close()

peps.save(a, lut, basepath + "state_D={:d}_chi={:d}_h={:e}_tau={:e}.peps".format(D, chi, h, tau))

e = env.get_site_environment()
mz = np.abs(e.contract(peps.make_double_layer(a[0], o=gates.sigmaz)) / e.contract(peps.make_double_layer(a[0])))
E = -h * e.contract(peps.make_double_layer(a[0], o=gates.sigmax)) / e.contract(peps.make_double_layer(a[0]))
e = env.get_bond_environment()
Za = peps.make_double_layer(a[0], o=gates.sigmaz)
Zb = peps.make_double_layer(a[1], o=gates.sigmaz)
A = peps.make_double_layer(a[0])
B = peps.make_double_layer(a[1])
E += -2*e.contract(Za, Zb) / e.contract(A, B)
f = open(basepath + "h_mz_D={:d}_chi={:d}.dat".format(D, chi), "a")
f.write("{:.15e} {:.15e} {:.15e} {:f} {:d} {:d} {:e} {:d}\n".format(h, mz, E, time()-t0, D, chi, tau, maxiterations))
f.close()

