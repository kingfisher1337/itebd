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

chi = 30
p = 2
D = 3
h = float(sys.argv[1])
lut = util.build_lattice_lookup_table([[0],[0]], [4,4])
num_params = 42

sys.stdout = open("output_varpeps_tfi/log_varpeps_tfi3_globalrandom_h={:f}.txt".format(h), "a")
sys.stderr = open("output_varpeps_tfi/err_varpeps_tfi3_globalrandom_h={:f}.txt".format(h), "a")

def get_symm_tensor(c):
    A = np.ndarray((2,3,3,3,3))
    A[0,0,0,0,0] = c[0]
    A[0,0,0,0,1] = A[0,0,0,1,0] = A[0,0,1,0,0] = A[0,1,0,0,0] = c[1]
    A[0,0,0,0,2] = A[0,0,0,2,0] = A[0,0,2,0,0] = A[0,2,0,0,0] = c[2]
    A[0,0,0,1,1] = A[0,0,1,1,0] = A[0,1,1,0,0] = A[0,1,0,0,1] = c[3]
    A[0,0,0,1,2] = A[0,0,1,2,0] = A[0,1,2,0,0] = A[0,2,0,0,1] = c[4]
    A[0,0,0,2,1] = A[0,0,2,1,0] = A[0,2,1,0,0] = A[0,1,0,0,2] = c[4]
    A[0,0,0,2,2] = A[0,0,2,2,0] = A[0,2,2,0,0] = A[0,2,0,0,2] = c[5]
    A[0,0,1,0,1] = A[0,1,0,1,0] = c[6]
    A[0,0,1,0,2] = A[0,1,0,2,0] = A[0,0,2,0,1] = A[0,2,0,1,0] = c[7]
    A[0,0,1,1,1] = A[0,1,1,1,0] = A[0,1,1,0,1] = A[0,1,0,1,1] = c[8]
    A[0,0,1,1,2] = A[0,1,1,2,0] = A[0,1,2,0,1] = A[0,2,0,1,1] = c[9]
    A[0,2,1,1,0] = A[0,1,1,0,2] = A[0,1,0,2,1] = A[0,0,2,1,1] = c[9]
    A[0,0,1,2,1] = A[0,1,2,1,0] = A[0,2,1,0,1] = A[0,1,0,1,2] = c[10]
    A[0,0,1,2,2] = A[0,1,2,2,0] = A[0,2,2,0,1] = A[0,2,0,1,2] = c[11]
    A[0,1,0,2,2] = A[0,0,2,2,1] = A[0,2,2,1,0] = A[0,2,1,0,2] = c[11]
    A[0,0,2,0,2] = A[0,2,0,2,0] = c[12]
    A[0,0,2,1,2] = A[0,2,1,2,0] = A[0,1,2,0,2] = A[0,2,0,2,1] = c[13]
    A[0,0,2,2,2] = A[0,2,2,2,0] = A[0,2,2,0,2] = A[0,2,0,2,2] = c[14]
    A[0,1,1,1,1] = c[15]
    A[0,1,1,1,2] = A[0,1,1,2,1] = A[0,1,2,1,1] = A[0,2,1,1,1] = c[16]
    A[0,1,1,2,2] = A[0,1,2,2,1] = A[0,2,2,1,1] = A[0,2,1,1,2] = c[17]
    A[0,1,2,1,2] = A[0,2,1,2,1] = c[18]
    A[0,1,2,2,2] = A[0,2,2,2,1] = A[0,2,2,1,2] = A[0,2,1,2,2] = c[19]
    A[0,2,2,2,2] = c[20]
    A[1,0,0,0,0] = c[21]
    A[1,0,0,0,1] = A[1,0,0,1,0] = A[1,0,1,0,0] = A[1,1,0,0,0] = c[22]
    A[1,0,0,0,2] = A[1,0,0,2,0] = A[1,0,2,0,0] = A[1,2,0,0,0] = c[23]
    A[1,0,0,1,1] = A[1,0,1,1,0] = A[1,1,1,0,0] = A[1,1,0,0,1] = c[24]
    A[1,0,0,1,2] = A[1,0,1,2,0] = A[1,1,2,0,0] = A[1,2,0,0,1] = c[25]
    A[1,0,0,2,1] = A[1,0,2,1,0] = A[1,2,1,0,0] = A[1,1,0,0,2] = c[25]
    A[1,0,0,2,2] = A[1,0,2,2,0] = A[1,2,2,0,0] = A[1,2,0,0,2] = c[26]
    A[1,0,1,0,1] = A[1,1,0,1,0] = c[27]
    A[1,0,1,0,2] = A[1,1,0,2,0] = A[1,0,2,0,1] = A[1,2,0,1,0] = c[28]
    A[1,0,1,1,1] = A[1,1,1,1,0] = A[1,1,1,0,1] = A[1,1,0,1,1] = c[29]
    A[1,0,1,1,2] = A[1,1,1,2,0] = A[1,1,2,0,1] = A[1,2,0,1,1] = c[30]
    A[1,2,1,1,0] = A[1,1,1,0,2] = A[1,1,0,2,1] = A[1,0,2,1,1] = c[30]
    A[1,0,1,2,1] = A[1,1,2,1,0] = A[1,2,1,0,1] = A[1,1,0,1,2] = c[31]
    A[1,0,1,2,2] = A[1,1,2,2,0] = A[1,2,2,0,1] = A[1,2,0,1,2] = c[32]
    A[1,1,0,2,2] = A[1,0,2,2,1] = A[1,2,2,1,0] = A[1,2,1,0,2] = c[32]
    A[1,0,2,0,2] = A[1,2,0,2,0] = c[33]
    A[1,0,2,1,2] = A[1,2,1,2,0] = A[1,1,2,0,2] = A[1,2,0,2,1] = c[34]
    A[1,0,2,2,2] = A[1,2,2,2,0] = A[1,2,2,0,2] = A[1,2,0,2,2] = c[35]
    A[1,1,1,1,1] = c[36]
    A[1,1,1,1,2] = A[1,1,1,2,1] = A[1,1,2,1,1] = A[1,2,1,1,1] = c[37]
    A[1,1,1,2,2] = A[1,1,2,2,1] = A[1,2,2,1,1] = A[1,2,1,1,2] = c[38]
    A[1,1,2,1,2] = A[1,2,1,2,1] = c[39]
    A[1,1,2,2,2] = A[1,2,2,2,1] = A[1,2,2,1,2] = A[1,2,1,2,2] = c[40]
    A[1,2,2,2,2] = c[41]
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

if os.path.isfile("output_varpeps_tfi/params_D=3_h={:f}_globalrandom.dat".format(h)):
    a0_best = np.loadtxt("output_varpeps_tfi/params_D=3_h={:f}_globalrandom.dat".format(h))
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
    peps.save([get_symm_tensor(a0_best)], lut, "output_varpeps_tfi/state_D=3_h={:f}_globalrandom.peps".format(h))
    np.savetxt("output_varpeps_tfi/params_D=3_h={:f}_globalrandom.dat".format(h), a0_best)

print "needed {:f} seconds".format(time() - t0)

