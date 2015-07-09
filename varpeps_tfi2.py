import ctm
import peps
import numpy as np
import gates
import util
import scipy.optimize
import matplotlib.pyplot as plt
import os

chi = 20
p = 2
D = 2
h = 0
lut = util.build_lattice_lookup_table([[0],[0]], [4,4])

num_params = 12

"""
def get_symm_tensor(c):
    A = np.ndarray((2,2,2,2,2))
    A[0,0,0,0,0] = 1
    A[0,0,0,0,1] = A[0,0,0,1,0] = A[0,0,1,0,0] = A[0,1,0,0,0] = c[0]
    A[0,0,0,1,1] = A[0,0,1,1,0] = A[0,1,1,0,0] = A[0,1,0,0,1] = c[1]
    A[0,0,1,0,1] = A[0,1,0,1,0] = c[2]
    A[0,0,1,1,1] = A[0,1,1,1,0] = A[0,1,1,0,1] = A[0,1,0,1,1] = c[3]
    A[0,1,1,1,1] = c[4]
    A[1,0,0,0,0] = c[5]
    A[1,0,0,0,1] = A[1,0,0,1,0] = A[1,0,1,0,0] = A[1,1,0,0,0] = c[6]
    A[1,0,0,1,1] = A[1,0,1,1,0] = A[1,1,1,0,0] = A[1,1,0,0,1] = c[7]
    A[1,0,1,0,1] = A[1,1,0,1,0] = c[8]
    A[1,0,1,1,1] = A[1,1,1,1,0] = A[1,1,1,0,1] = A[1,1,0,1,1] = c[9]
    A[1,1,1,1,1] = c[10]
    return A
"""

def get_symm_tensor(c):
    A = np.zeros([2]*5)
    for (a,b,d) in [(0,1,-1),(1,0,5)]:
        A[a,a,a,a,a] = c[1+d]
        A[a,b,a,a,a] = c[2+d]
        A[a,a,b,a,a] = c[2+d]
        A[a,a,a,b,a] = c[2+d]
        A[a,a,a,a,b] = c[2+d]
        A[a,b,b,a,a] = c[3+d]
        A[a,a,b,b,a] = c[3+d]
        A[a,a,a,b,b] = c[3+d]
        A[a,b,a,a,b] = c[3+d]
        A[a,b,a,b,a] = c[4+d]
        A[a,a,b,a,b] = c[4+d]
        A[a,b,b,b,a] = c[5+d]
        A[a,a,b,b,b] = c[5+d]
        A[a,b,a,b,b] = c[5+d]
        A[a,b,b,a,b] = c[5+d]
        A[a,b,b,b,b] = c[6+d]
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
    #env = ctm.ctmrg([A], lut, chi, env, tester)
    #env = ctm.ctmrg_1x1_rotsymm(A, chi, env, tester)
    env = ctm.ctmrg_1x1_hermitian(A, chi, env, tester)
    
    #env2 = ctm.CTMEnvironment(lut, env.c, env.c, env.c, env.c, env.t1, env.t2, env.t1, env.t2)
    #tester = ctm.CTMRGGenericTester(f, 1e-8)
    #env2 = ctm.ctmrg([A], lut, chi, env2, tester)
    
    E = tester.get_value()
    
    if returnmz:
        e = env.get_site_environment()
        Z = peps.make_double_layer(a, o=gates.sigmaz)
        mz = e.contract(Z) / e.contract(A)
        return E, mz
    
    return E

f = open("output_varpeps_tfi/h_mz_E_D=2.dat", "w")

if num_params == 12:
    #a0 = np.array([1]+[0]*11, dtype=float) # fm peps
    a0 = np.array([1, 0.5, 0.3, 0.5, 0.1, 0.1, 1, 0.2, 0.3, 0.5, 0.1, 1])
elif num_params == 11:
    a0 = np.array([0]*11, dtype=float)

for h_it in np.concatenate([
    np.linspace(0, 2.7, 27, endpoint=False),
    np.linspace(2.7, 2.9, 8, endpoint=False),
    np.linspace(2.9, 3, 8, endpoint=False),
    np.linspace(3, 3.15, 25),
    #np.linspace(2.9, 3.15, 21),
    np.linspace(3.2, 3.4, 3)]):
    h = h_it
    
    res = scipy.optimize.basinhopping(get_energy, a0, niter=100, interval=10, T=2.0, stepsize=0.2, disp=True)
    #res = scipy.optimize.minimize(get_energy, a0)
    a0 = res.x
    
    peps.save([get_symm_tensor(a0)], lut, "output_varpeps_tfi/state_D=2_h={:f}.peps".format(h))
    
    f2 = open("output_varpeps_tfi/params_D=2_h={:f}.dat".format(h), "w")
    for j in xrange(len(a0)):
        f2.write("{:.15e}\n".format(a0[j]))
    f2.close()
    
    #if res.success:
    #    print h, res.success
    #else:
    #    print h, res.success, res.message
    print h, res.message
    
    E, mz = get_energy(a0, True)
    f.write("{:.15e} {:.15e} {:.15e}\n".format(h, mz, E))
    f.flush()

f.close()

hlist1 = []
mzlist1 = []
Elist1 = []
with open("output_varpeps_tfi/h_mz_E_D=2.dat", "r") as f:
    for line in f:
        fields = line.split("\n")[0].split(" ")
        hlist1.append(float(fields[0]))
        mzlist1.append(abs(float(fields[1])))
        Elist1.append(float(fields[2]))
    
plt.plot(hlist1, Elist1, marker="x")
plt.plot(hlist1, np.sign(Elist1[-1])*np.array(hlist1), "--")
plt.grid(True)
plt.ylim(ymax=-2)
plt.xlabel("$h$")
plt.ylabel("$E$")
plt.savefig("output_varpeps_tfi/h_E_D=2.png")
plt.close()

hlist = []
mzlist = []
with open("output_tfi/prb80_2.dat", "r") as f:
    for line in f:
        fields = line.split("\n")[0].split(" ")
        hlist.append(float(fields[0]))
        mzlist.append(float(fields[1]))
#plt.plot(hlist, mzlist, marker="o", label="PRB 80, 094403, $D=2$", mec="blue", mfc="none")
plt.plot(hlist, mzlist, marker="+", label="PRB 80, 094403, $D=2$")

#hlist = []
#mzlist = []
#with open("output_tfi/gent_h_mz_E.dat", "r") as f:
#    for line in f:
#        fields = line.split("\n")[0].split(" ")
#        hlist.append(float(fields[0]))
#        mzlist.append(np.abs(float(fields[1])))
#plt.plot(hlist, mzlist, marker="+", label="TNSS Ghent 2015, $D=2$")
plt.plot(hlist1, mzlist1, marker="x", label="scipy.optimize $D=2$")

plt.grid(True)
plt.xlabel("$h$")
plt.xlim(2.7, 3.2)
plt.ylim(0, 0.7)
plt.ylabel("$m_z$")
plt.legend(loc="best")
plt.savefig("output_varpeps_tfi/h_mz_D=2.png", dpi=300)
plt.close()

x = []
y = None
for filename in sorted(os.listdir("output_varpeps_tfi")):
    if filename.startswith("params_D=2_") and filename.endswith(".dat"):
        h = float(filter(lambda s: s.find("h=") != -1, filename[:-4].split("_"))[0].split("=")[-1])
        x.append(h)
        params = np.loadtxt("output_varpeps_tfi/" + filename)
        if y is None:
            y = [[] for j in xrange(len(params))]
        for j in xrange(len(params)):
            y[j].append(params[j])

for j in xrange(len(y)):
    for k in xrange(len(y[0])):
        y[j][k] /= y[0][k]

for j in xrange(len(y)):
    #plt.plot(x, y[j], label="param {:d}".format(j+1))
    plt.plot(x, y[j])
plt.grid(True)
plt.xlabel("$h$")
plt.ylabel("parameters for rotsymm PEPS")
#plt.legend(loc="best")
#box = plt.gca().get_position()
#plt.gca().set_position([box.x0, box.y0, box.width*0.8, box.height])
#plt.gca().legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig("output_varpeps_tfi/params_D=2.png", dpi=300)
plt.close()


