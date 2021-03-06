import numpy as np
import tebd
import gates
import sys
import peps
import util
import os
import globallog
from time import time
from polish import polish

t0 = time()

chi = int(sys.argv[1])
h = float(sys.argv[2])
statefile = sys.argv[3]
output_to_terminal = "-writehere" in sys.argv

num_workers = 1
if "-workers" in sys.argv:
    num_workers = int(sys.argv[sys.argv.index("-workers") + 1])

globallog.write("tfi_gs_polish.py, chi={0:d}, h={1:f}, statefile=\"{2:s}\"\n".format(chi, h, statefile))

basepath_in = "output_tfi/"
basepath_out = "output_tfi_polish/"

if not output_to_terminal:
    logfile = open(basepath_out + statefile[:-5] + ".log", "a")
    sys.stdout = logfile
    sys.stderr = logfile

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

a, nns = peps.load(basepath_in + statefile)
lut = util.build_lattice_lookup_table(nns, [4,4])

def save_callback(a):
    peps.save(a, lut, basepath_out + statefile[statefile.rfind("/")+1:])

ecf = tebd.CTMRGEnvContractorFactory(lut, chi, test_fct, 1e-12, 1e-15)
a = polish(a, lut, ecf, num_workers=num_workers, peps_save_callback=save_callback)

peps.save(a, lut, basepath_out + statefile[statefile.rfind("/")+1:])

print "tfi_gs_polish.py done; needed {0:f} seconds".format(time() - t0)



