import ctm
import peps
import numpy as np
import gates
import util
import scipy.optimize
import matplotlib.pyplot as plt

chi = 30
p = 2
D = 3
h = 0
lut = util.build_lattice_lookup_table([[0],[0]], [4,4])

def vec2_to_vec3(c):
    c2 = np.zeros(42)
    c2[0] = c[0]
    c2[1] = c[1]
    c2[3] = c[2]
    c2[6] = c[3]
    c2[8] = c[4]
    c2[15] = c[5]
    c2[36] = c[6]
    c2[8] = c[7]
    c2[24] = c[8]
    c2[27] = c[9]
    c2[22] = c[10]
    c2[21] = c[11]
    return c2

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
    
    tester = ctm.CTMRGGenericTester(test_fct(a, A), 1e-9)
    env = ctm.ctmrg_1x1_hermitian(A, chi, env, tester)
    
    E = tester.get_value()
    
    if returnmz:
        e = env.get_site_environment()
        Z = peps.make_double_layer(a, o=gates.sigmaz)
        mz = e.contract(Z) / e.contract(A)
        return E, mz
    
    return E




f = open("output_varpeps_tfi/h_mz_E_D=3.dat", "w")

#a0 = np.array([1]+[0]*41, dtype=float) # fm peps

for h_it in np.concatenate([
    np.linspace(0, 2.7, 27, endpoint=False),
    np.linspace(2.7, 2.9, 8, endpoint=False),
    np.linspace(2.9, 3.15, 21),
    np.linspace(3.2, 3.4, 3)]):
    h = h_it
    
    a0 = np.loadtxt("output_varpeps_tfi/params_D=2_h={:f}.dat".format(h))
    a0 = vec2_to_vec3(a0)
    res = scipy.optimize.minimize(get_energy, a0)
    a0 = res.x
    
    peps.save([get_symm_tensor(a0)], lut, "output_varpeps_tfi/state_D=3_h={:f}.peps".format(h))
    
    f2 = open("output_varpeps_tfi/params_D=3_h={:f}.dat".format(h), "w")
    for j in xrange(len(a0)):
        f2.write("{:.15e}\n".format(a0[j]))
    f2.close()
    
    if res.success:
        print h, res.success
    else:
        print h, res.success, res.message
    
    E, mz = get_energy(a0, True)
    f.write("{:.15e} {:.15e} {:.15e}\n".format(h, mz, E))
    f.flush()

f.close()

