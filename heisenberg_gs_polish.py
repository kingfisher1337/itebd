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

chi = int(sys.argv[1])
J = int(sys.argv[2])
h = float(sys.argv[3])
statefile = sys.argv[4]

num_threads = 1
if "-threads" in sys.argv:
    num_threads = int(sys.argv[sys.argv.index("-threads") + 1])

globallog.write("heisenberg_gs_polish.py, chi={:d}, J={:d}, h={:f}, statefile=\"{:s}\"\n".format(chi, J, h, statefile))

basepath = "output_heisenberg_polish/"

if not ("-writehere" in sys.argv):
    logfile = open(basepath + statefile[statefile.rfind("/")+1:-5] + ".log", "a")
    sys.stdout = logfile
    sys.stderr = logfile

def test_fct(a, A):
    n = len(a)
    Z = map(lambda b: peps.make_double_layer(b, o=gates.sigmaz), a)
    AT = map(lambda B: B.transpose([1,2,3,0]), A)
    ZT = map(lambda B: B.transpose([1,2,3,0]), Z)
    
    X = map(lambda b: peps.make_double_layer(b, o=gates.sigmax), a)
    Y = map(lambda b: peps.make_double_layer(b, o=gates.sigmay), a)
    XT = map(lambda B: B.transpose([1,2,3,0]), X)
    YT = map(lambda B: B.transpose([1,2,3,0]), Y)
    
    def test_fct_impl(e):
        c_zz = 0
        c_xx = 0
        c_yy = 0
        
        mz = [0, 0]
        
        for j in xrange(n):
            e1 = e.get_site_environment(j)
            mz[j] = e1.contract(Z[j]) / e1.contract(A[j])
            
            e2 = e.get_bond_environment_horizontal(j)
            norm = e2.contract(A[j], A[lut[j,1,0]])
            c_xx += e2.contract(X[j], X[lut[j,1,0]]) / norm
            c_yy += np.real(e2.contract(Y[j], Y[lut[j,1,0]]) / norm)
            c_zz += e2.contract(Z[j], Z[lut[j,1,0]]) / norm
            
            e2 = e.get_bond_environment_vertical(j)
            norm = e2.contract(AT[j], AT[lut[j,0,1]])
            c_xx += e2.contract(XT[j], XT[lut[j,0,1]]) / norm
            c_yy += np.real(e2.contract(YT[j], YT[lut[j,0,1]]) / norm)
            c_zz += e2.contract(ZT[j], ZT[lut[j,0,1]]) / norm
        
        c_xx /= (2*n)
        c_yy /= (2*n)
        c_zz /= (2*n)
        E = 2*J*(c_xx + c_yy + c_zz) + h*0.5*(mz[0] + mz[1])
        
        return mz + [c_xx, c_yy, c_zz, E]
    return test_fct_impl

a, nns = peps.load(statefile)
lut = util.build_lattice_lookup_table(nns, [4,4])

env_contractor = tebd.CTMRGEnvContractor(lut, chi, test_fct, 1e-12, 1e-15)
a = tebd.polish(a, lut, env_contractor, pepsfilename=(basepath + statefile[statefile.rfind("/")+1:]), num_threads=num_threads)

peps.save(a, lut, basepath + statefile[statefile.rfind("/")+1:])

print "heisenberg_gs_polish.py done; needed {:f} seconds".format(time() - t0)
