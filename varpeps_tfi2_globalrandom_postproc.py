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

chi = 20
p = 2
D = 2

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
    #env = ctm.ctmrg_1x1_hermitian(A, chi, env, tester)
    env = ctm.ctmrg_1x1_rotsymm(A, chi, env, tester)
    
    E = tester.get_value() if tester.is_converged() else 1e10
    
    if returnmz:
        e = env.get_site_environment()
        Z = peps.make_double_layer(a, o=gates.sigmaz)
        mz = e.contract(Z) / e.contract(A)
        return E, mz
    
    return E

f = open("output_varpeps_tfi/h_mz_E_D=2_globalrandom.dat", "w")
Elist = []
mzlist = []

for filename in sorted(os.listdir("output_varpeps_tfi")):
    if filename.startswith("params_D=2_") and filename.endswith("_globalrandom.dat"):
        h = float(filter(lambda s: s.find("h=") != -1, filename[:-4].split("_"))[0].split("=")[-1])
        x = np.loadtxt("output_varpeps_tfi/" + filename)
        E, mz = get_energy(x, True)
        f.write("{:.15e} {:.15e} {:.15e}\n".format(h, mz, E))
        Elist.append(E)
        mzlist.append(mz)
f.close()

