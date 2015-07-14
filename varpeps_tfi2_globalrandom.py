import ctm
import peps
import numpy as np
import gates
import util
import scipy.optimize
import matplotlib.pyplot as plt
import os
import sys
from time import time

t0 = time()

chi = 20
p = 2
D = 2
h = float(sys.argv[1])
lut = util.build_lattice_lookup_table([[0],[0]], [4,4])
num_params = 12

sys.stdout = open("output_varpeps_tfi/log_varpeps_tfi2_globalrandom_h={:f}.txt".format(h), "a")
sys.stderr = open("output_varpeps_tfi/err_varpeps_tfi2_globalrandom_h={:f}.txt".format(h), "a")

def get_symm_tensor(c):
    A = np.ndarray([2]*5)
    for j in [0,1]:
        A[j,0,0,0,0] = c[0+6*j]
        A[j,0,0,0,1] = A[j,0,0,1,0] = A[j,0,1,0,0] = A[j,1,0,0,0] = c[1+6*j]
        A[j,0,0,1,1] = A[j,0,1,1,0] = A[j,1,1,0,0] = A[j,1,0,0,1] = c[2+6*j]
        A[j,0,1,0,1] = A[j,1,0,1,0] = c[3+6*j]
        A[j,0,1,1,1] = A[j,1,1,1,0] = A[j,1,1,0,1] = A[j,1,0,1,1] = c[4+6*j]
        A[j,1,1,1,1] = c[5+6*j]
    return A

def test_fct(a, A):
    X = peps.make_double_layer(a, o=gates.sigmax)
    Z = peps.make_double_layer(a, o=gates.sigmaz)
    def test_fct_impl(e):
        e1 = e.get_site_environment()
        mx = e1.contract(X) / e1.contract(A)
        e2 = e.get_bond_environment()
        czz = e2.contract(Z, Z) / e2.contract(A, A)
        return -2*czz - h*mx
    return test_fct_impl

env = None
def get_energy(a, returnmz=False):
    global env
    a = get_symm_tensor(a)
    A = peps.make_double_layer(a)
    
    f = test_fct(a, A)
    tester = ctm.CTMRGGenericTester(f, 1e-12)
    env = ctm.ctmrg_1x1_hermitian(A, chi, env, tester)
    
    E = tester.get_value() if tester.is_converged() else 1e10
    
    if returnmz:
        e = env.get_site_environment()
        Z = peps.make_double_layer(a, o=gates.sigmaz)
        mz = e.contract(Z) / e.contract(A)
        return E, mz
    
    return E


num_tries = int(sys.argv[2]) if len(sys.argv) >= 3 else 100

if os.path.isfile("output_varpeps_tfi/params_D=2_h={:f}_globalrandom.dat".format(h)):
    a0_best = np.loadtxt("output_varpeps_tfi/params_D=2_h={:f}_globalrandom.dat".format(h))
    Emin = get_energy(a0_best)
    print "starting global ground state search with energy {:.15e}".format(Emin)
    sys.stdout.flush()
else:
    Emin = 0
    a0_best = None

found_new_min = False

for j in xrange(num_tries):
    print "try number {:d}".format(j)
    sys.stdout.flush()
    a0 = np.random.rand(num_params)
    res = scipy.optimize.minimize(get_energy, a0)
    a0 = res.x
    E, mz = get_energy(a0, True)
    if E < Emin:
        Emin = E
        a0_best = np.copy(a0)
        found_new_min = True
        print "found new minimal energy {:.15e} at try {:d} (mz = {:.15e})".format(E, j+1, mz)
        sys.stdout.flush()

if found_new_min:
    peps.save([get_symm_tensor(a0_best)], lut, "output_varpeps_tfi/state_D=2_h={:f}_globalrandom.peps".format(h))
    np.savetxt("output_varpeps_tfi/params_D=2_h={:f}_globalrandom.dat".format(h), a0_best)

print "needed {:f} seconds".format(time() - t0)

