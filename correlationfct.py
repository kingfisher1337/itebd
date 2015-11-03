import numpy as np
from numpy import tensordot as tdot
import ctm
import gates
import peps
import util
import sys
import os

statefile = sys.argv[1]
outfile = sys.argv[2]
chi = int(sys.argv[3])
Lmax = int(sys.argv[4])

def test_fct(e):
    mz = [0]*n
    for j in xrange(n):
        e1 = e.get_site_environment(j)
        mz[j] = e1.contract(Z[j]) / e1.contract(A[j])
    return mz

a, nns = peps.load(statefile)
n = len(a)
lut = util.build_lattice_lookup_table(nns, [4,4])
A = map(peps.make_double_layer, a)
Z = map(lambda b: peps.make_double_layer(b, o=gates.sigmaz), a)

tester = ctm.CTMRGTester(test_fct, 1e-12, 1e-15)
env = ctm.ctmrg(A, lut, chi, "random", tester)

m = np.empty((3, n))
sigma = [gates.sigmax, np.imag(gates.sigmay), gates.sigmaz]
# note: handle take sigmay/i instead of sigmay and multiply with (1/i)^2 = -1

for j in xrange(n):
    e1 = env.get_site_environment(j)
    for k in xrange(3):
        op = sigma[k]
        O = peps.make_double_layer(a[j], o=op)
        m[k,j] = e1.contract(O) / e1.contract(A[j])

def correlation_fct(component):
    X = map(lambda b: peps.make_double_layer(b, o=sigma[component]), a)
    mx = m[component]
    
    tmp = tdot(env.c4[lut[0,-1,-1]], env.t4[lut[0,-1,0]], [1,0])
    tmp = tdot(tmp, env.c1[lut[0,-1,1]], [2,0])
    tmp = tdot(tmp, env.t3[lut[0,0,-1]], [0,2])
    tmp2 = tdot(tmp, X[0], [[3,0],[2,3]])
    uX = tdot(tmp2, env.t1[lut[0,0,1]], [[0,2],[0,1]]).flatten()
    tmp2 = tdot(tmp, A[0], [[3,0],[2,3]])
    uA = tdot(tmp2, env.t1[lut[0,0,1]], [[0,2],[0,1]]).flatten()
    
    vX = [None]*n
    vA = [None]*n
    for j in xrange(n):
        tmp = tdot(env.c3[lut[j,1,-1]], env.t2[lut[j,1,0]], [0,2])
        tmp = tdot(tmp, env.c2[lut[j,1,1]], [1,1])
        tmp = tdot(tmp, env.t3[lut[j,0,-1]], [0,0])
        tmp2 = tdot(tmp, X[j], [[0,2],[1,2]])
        vX[j] = tdot(tmp2, env.t1[lut[j,0,1]], [[0,2],[2,1]]).flatten()
        tmp2 = tdot(tmp, A[j], [[0,2],[1,2]])
        vA[j] = tdot(tmp2, env.t1[lut[j,0,1]], [[0,2],[2,1]]).flatten()
    
    M = [None]*n
    for j in xrange(n):
        tmp = tdot(env.t3[lut[j,0,-1]], A[j], [1,2])
        tmp = tdot(tmp, env.t1[lut[j,0,1]], [2,1])
        M[j] = tmp.transpose([1,3,4,0,2,5]).reshape(len(vX[lut[j,-1,0]]), len(vX[lut[j,1,0]]))
    
    res = np.empty(Lmax+1)
    j = lut[0,1,0]
    for L in xrange(Lmax+1):
        print L
        res[L] = np.dot(uX, vX[j]) / np.dot(uA, vA[j]) - mx[0]*mx[j]
        
        if L < Lmax:
            uX = np.dot(uX, M[j])
            uA = np.dot(uA, M[j])
        
        j = nns[0][j]
    
    return res


cxx = correlation_fct(0)
cyy = correlation_fct(1) * (-1.0)
czz = correlation_fct(2)
css = 0.25 * (cxx + cyy + czz)

f = open(outfile, "w")
for L in xrange(1, Lmax+1):
    f.write("{:.15e} {:.15e} {:.15e} {:.15e}\n".format(cxx[L], cyy[L], czz[L], css[L]))
f.close()

