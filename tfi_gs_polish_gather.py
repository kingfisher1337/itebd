import numpy as np
import tebd
import gates
import sys
import peps
import util
import os

D = int(sys.argv[1])
chi = int(sys.argv[2])
trotter2 = "-trotter2" in sys.argv


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


basepath = "output_tfi_polish/"
f = open(basepath + "h_mz_E_D={:d}_chi={:d}.dat".format((D, chi)))
env_contractor = tebd.CTMRGEnvContractor(lut, chi, test_fct, 1e-12, 1e-15)

for filename in sorted(os.listdir(basepath)):
    if filename.endswith(".peps"):
        _D = int(filter(lambda s: s.find("D=") != -1, filename[:-5].split("_"))[0].split("=")[-1])
        _chi = int(filter(lambda s: s.find("chi=") != -1, filename[:-5].split("_"))[0].split("=")[-1])
        h = float(filter(lambda s: s.find("h=") != -1, filename[:-5].split("_"))[0].split("=")[-1])
        _tau = float(filter(lambda s: s.find("tau=") != -1, filename[:-5].split("_"))[0].split("=")[-1])
        _trotter2 = "trotter2" in filename
        
        if _D == D and _chi == chi:
            a, nns = peps.load(basepath + filename)
            lut = util.build_lattice_lookup_table(nns, [4,4])
            env_contractor.update(a)
            mz, _, E = env_contractor.get_test_values()
            f.write("{:.15e} {:.15e} {:.15e}\n".format(h, mz, E))
f.close()

