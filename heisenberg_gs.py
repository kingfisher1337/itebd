import numpy as np
from numpy import tensordot as tdot

import tebd
import gates
import peps
import util

import sys
import os
import globallog

J = int(sys.argv[1])
h = float(sys.argv[2])
D = int(sys.argv[3])
chi = int(sys.argv[4])
tau = float(sys.argv[5])
maxiterations = int(sys.argv[6])
statefile = sys.argv[7]

trotter_second_order = True

if "-backup" in sys.argv:
    backup_interval = int(sys.argv[sys.argv.index("-backup") + 1])
else:
    backup_interval = 10

name_suffix = ""
if "-namesuffix" in sys.argv:
    name_suffix = "_" + sys.argv[sys.argv.index("-namesuffix") + 1]

#np.random.seed(523451109)
#np.random.seed(904265234)

globallog.write("heisenberg_gs.py, D={:d}, chi={:d}, J={:d}, h={:f}, tau={:.0e}, iterations={:d}, trotter order {:d}{:s}{:s}\n".format(D, chi, J, h, tau, maxiterations, 2 if trotter_second_order else 1, ", ffu" if fast_full_update else "", "" if name_suffix == "" else ", name_suffix=" + name_suffix))

def test_fct(a, A):
    n = len(a)
    P = map(lambda b: peps.make_double_layer(b, o=gates.sigmap), a)
    M = map(lambda b: peps.make_double_layer(b, o=gates.sigmam), a)
    Z = map(lambda b: peps.make_double_layer(b, o=gates.sigmaz), a)
    AT = map(lambda B: B.transpose([1,2,3,0]), A)
    PT = map(lambda B: B.transpose([1,2,3,0]), P)
    MT = map(lambda B: B.transpose([1,2,3,0]), M)
    ZT = map(lambda B: B.transpose([1,2,3,0]), Z)
    
    def test_fct_impl(e):
        c_pm = 0
        c_mp = 0
        c_zz = 0
        mz = [0, 0]
        
        for j in xrange(n):
            e1 = e.get_site_environment(j)
            mz[j] = e1.contract(Z[j]) / e1.contract(A[j])
            e2 = e.get_bond_environment_horizontal(j)
            norm = e2.contract(A[j], A[lut[j,1,0]])
            c_pm += e2.contract(P[j], M[lut[j,1,0]]) / norm
            c_mp += e2.contract(M[j], P[lut[j,1,0]]) / norm
            c_zz += e2.contract(Z[j], Z[lut[j,1,0]]) / norm
            norm = e2.contract(AT[j], AT[lut[j,0,1]])
            c_pm += e2.contract(PT[j], MT[lut[j,0,1]]) / norm
            c_mp += e2.contract(MT[j], PT[lut[j,0,1]]) / norm
            c_zz += e2.contract(ZT[j], ZT[lut[j,0,1]]) / norm
        
        c_pm /= (2*n)
        c_mp /= (2*n)
        c_zz /= (2*n)
        E = 2*J*(c_zz + 0.5*c_pm + 0.5*c_mp) + h*0.5*(mz[0] + mz[1])
        
        return mz + [c_pm, c_mp, c_zz, E]
    return test_fct_impl

basepath = "output_heisenberg/"
if os.path.isfile(basepath + statefile):
    a, nns = peps.load(basepath + statefile, dtype=complex)
    lut = util.build_lattice_lookup_table(nns, [4,4])
    _D = int(filter(lambda s: s.find("D=") != -1, statefile[:-5].split("_"))[0].split("=")[-1])
    _chi = int(filter(lambda s: s.find("chi=") != -1, statefile[:-5].split("_"))[0].split("=")[-1])
    if _D != D or _chi != chi:
        t0 = 0
    else:
        t0 = float(filter(lambda s: s.find("t=") != -1, statefile[:-5].split("_"))[0].split("=")[-1])
else:
    a = list(peps.get_state_neel(D))
    #a = peps.get_state_pm(D)
    #a = [a]*2
    a[0] += 1e-3 * (np.random.rand(2,D,D,D,D) - 0.5)
    a[1] += 1e-3 * (np.random.rand(2,D,D,D,D) - 0.5)
    #a = peps.get_state_random_rotsymm(2, D)
    #a = [a]*2
    lut = util.build_lattice_lookup_table([[1,0],[1,0]], [4,4])
    t0 = 0


def get_gates(dt):
    gx = gates.exp_sigmax_sigmax(-0.5*J*dt).reshape(4, 4)
    gy = gates.exp_sigmay_sigmay(-0.5*J*dt).reshape(4, 4)
    gz = gates.exp_sigmaz_sigmaz(-0.5*J*dt).reshape(4, 4)
    gHalf = np.dot(np.dot(gx, gy), gz).reshape(2, 2, 2, 2)
    gx = gates.exp_sigmax_sigmax(-J*dt).reshape(4, 4)
    gy = gates.exp_sigmay_sigmay(-J*dt).reshape(4, 4)
    gz = gates.exp_sigmaz_sigmaz(-J*dt).reshape(4, 4)
    gFull = np.dot(np.dot(gx, gy), gz).reshape(2, 2, 2, 2)
    g2 = [(0,0,gHalf),(0,1,gHalf),(1,0,gHalf),(1,1,gFull),(1,0,gHalf),(0,1,gHalf),(0,0,gHalf)]
    
    if h == 0:
        return [], g2, []
    else:
        g1a = gates.exp_sigmaz(-0.5*dt*h)
        g1b = gates.exp_sigmaz(+0.5*dt*h)
        g1 = [(0,g1a), (1,g1b)]
        return g1, g2, g1
    

env_contractor = tebd.CTMRGEnvContractor(lut, chi, test_fct, 1e-9, 1e-14, max_iterations_per_update=500, ctmrg_verbose=True, plotonfail=False)
simulation_name = "J={:d}_h={:f}_D={:d}_chi={:d}_tau={:.6f}{:s}".format(J, h, D, chi, tau, name_suffix)
tebd.itebd_v2(a, lut, t0, tau, maxiterations*tau, get_gates, env_contractor, basepath, simulation_name, backup_interval)

