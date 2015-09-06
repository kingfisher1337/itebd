import numpy as np
import tebd
import gates
import sys
import peps
import util
import os
import globallog

from time import time
t0 = time()

np.random.seed(56158422)


D = int(sys.argv[1])
chi = int(sys.argv[2])
h = float(sys.argv[3])
tau = float(sys.argv[4])
maxiterations = int(sys.argv[5])
statefile = sys.argv[6]
fast_full_update = "-ffu" in sys.argv
trotter_second_order = "-trotter2" in sys.argv
output_to_terminal = "-writehere" in sys.argv
if "-backup" in sys.argv:
    backup_interval = int(sys.argv[sys.argv.index("-backup") + 1])
else:
    backup_interval = 100
name_suffix = ""
if "-namesuffix" in sys.argv:
    name_suffix = sys.argv[sys.argv.index("-namesuffix") + 1]

globallog.write("tfi_gs_2d.py, D={:d}, chi={:d}, h={:f}, tau={:.0e}, iterations={:d}, trotter order {:d}{:s}{:s}\n".format(D, chi, h, tau, maxiterations, 2 if trotter_second_order else 1, ", ffu" if fast_full_update else "", "" if name_suffix == "" else ", name_suffix=" + name_suffix))

basepath = "output_tfi/"
name_suffix = "_" + name_suffix

if not output_to_terminal:
    f = open(basepath + "log_tfi_gs_2d_D={:d}_chi={:d}_h={:f}_tau={:.0e}{:s}.txt".format(D, chi, h, tau, name_suffix), "a")
    sys.stdout = f
    sys.stderr = f

if os.path.isfile(basepath + statefile):
    a, nns = peps.load(basepath + statefile)
    lut = util.build_lattice_lookup_table(nns, [4,4])
    if a[0].shape[1] < D:
        a = peps.increase_bond_dimension(a, D)
    _D = int(filter(lambda s: s.find("D=") != -1, statefile[:-5].split("_"))[0].split("=")[-1])
    _chi = int(filter(lambda s: s.find("chi=") != -1, statefile[:-5].split("_"))[0].split("=")[-1])
    _h = float(filter(lambda s: s.find("h=") != -1, statefile[:-5].split("_"))[0].split("=")[-1])
    _trotter2 = "_trotter2" in statefile
    if _D != D or _chi != chi or _h != h or _trotter2 != trotter_second_order:
        t0 = 0
    else:
        t0 = float(filter(lambda s: s.find("t=") != -1, statefile[:-5].split("_"))[0].split("=")[-1])
    
else:
    print "no file \"{:s}\" found! starting new calculation".format(basepath + statefile)
    a = [None]*2
    for j in xrange(len(a)):
        #a[j] = peps.get_state_ising(1.8)
        a[j] = peps.get_state_random_rotsymm(2, D)
    lut = util.build_lattice_lookup_table([[1,0],[1,0]], [4,4])
    t0 = 0

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

def get_gates(dt):
    if trotter_second_order:
        g1 = gates.exp_sigmax(0.5*tau*h)
        g1pre = g1post = [(0, g1), (1, g1)]
    else:
        g1 = gates.exp_sigmax(tau*h)
        g1pre = [(0, g1), (1, g1)]
        g1post = []
    g2 = gates.exp_sigmaz_sigmaz(tau)
    g2 = [(0, 0, g2), (0, 1, g2), (1, 0, g2), (1, 1, g2)]
    return g1pre, g2, g1post

env_contractor = tebd.CTMRGEnvContractor(lut, chi, test_fct, 1e-12, 1e-15, ctmrg_verbose=True)
simulation_name = "D={:d}_chi={:d}_h={:f}_tau={:.6f}{:s}{:s}".format(D, chi, h, tau, "_trotter2" if trotter_second_order else "", name_suffix)
tebd.itebd_v2(a, lut, t0, tau, maxiterations*tau, get_gates, env_contractor, basepath, simulation_name, backup_interval, mode="fu")

