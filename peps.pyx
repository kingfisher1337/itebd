import numpy as np
from numpy import dot
from numpy import tensordot as tdot

import util

def get_state_pm(D=1, dtype=float):
    return np.ones((2,D,D,D,D), dtype=dtype)

def get_state_fm0(D=1, dtype=float):
    a = np.zeros((2,D), dtype=dtype)
    a[0,0] = 1
    return np.fromfunction(
        lambda j,k,l,m,n: a[j,k]*a[j,l]*a[j,m]*a[j,n], (2,D,D,D,D), dtype=int)
def get_state_fm1(D=1, dtype=float):
    a = np.zeros((2,D), dtype=dtype)
    a[1,0] = 1
    return np.fromfunction(
        lambda j,k,l,m,n: a[j,k]*a[j,l]*a[j,m]*a[j,n], (2,D,D,D,D), dtype=int)
def get_state_fm_symmetric_superposition(D=2, dtype=float):
    if D < 2:
        raise ValueError("the bond dimension for |000...> + |111...> has to be larger than 1!")
    a = np.zeros((2,D), dtype=dtype)
    a[0,0] = a[1,1] = 1
    return np.fromfunction(
        lambda j,k,l,m,n: a[j,k]*a[j,l]*a[j,m]*a[j,n], (2,D,D,D,D), dtype=int)
def get_state_neel(D=1, dtype=float):
    a = np.zeros((2,D),dtype=dtype)
    a[0,0] = 1
    a = np.fromfunction(lambda j,k,l,m,n: a[j,k]*a[j,l]*a[j,m]*a[j,n], (2,D,D,D,D), dtype=int)
    b = np.zeros((2,D),dtype=dtype)
    b[1,0] = 1
    b = np.fromfunction(lambda j,k,l,m,n: b[j,k]*b[j,l]*b[j,m]*b[j,n], (2,D,D,D,D), dtype=int)
    return a, b
def get_state_ising_from_theta(theta, dtype=float):
    a = np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.fromfunction(lambda j,k,l,m,n: a[j,k]*a[j,l]*a[j,m]*a[j,n], (2,2,2,2,2), dtype=int)
def get_state_ising(T, dtype=float):
    theta = 0 if T == 0 else 0.5 * np.arcsin(np.exp(-1.0/T))
    return get_state_ising_from_theta(theta, dtype)

def get_state_random(p, D, dtype=float):
    if dtype == complex:
        return np.random.rand(p, D, D, D, D) + 1j*np.random.rand(p, D, D, D, D)
    else:
        return np.random.rand(p, D, D, D, D)
def get_state_random_hermitian(p, D, dtype=float):
    a = get_state_random(p, D, dtype)
    return a + a.swapaxes(1,3).conj() + a.swapaxes(2,4).conj() + a.transpose([0,3,4,1,2])
def get_state_random_rotsymm(p, D, dtype=float):
    a = get_state_random(p, D, dtype)
    return a + a.transpose([0,2,3,4,1]) + a.transpose([0,3,4,1,2]) + a.transpose([0,4,1,2,3])

def make_double_layer(a, b=None, o=None):
    n = len(a.shape) - 1
    
    if b is None:
        b = a.conj()
    
    if n == 4:
        p, D1, D2, D3, D4 = a.shape
        _, D5, D6, D7, D8 = b.shape
        a = a.reshape(p, D1*D2*D3*D4)
        if o is not None:
            a = dot(o, a)
        x = dot(a.T, b.reshape(p, D5*D6*D7*D8)).reshape(D1, D2, D3, D4, D5, D6, D7, D8)
        return x.transpose([0,4,1,5,2,6,3,7]).reshape(D1*D5, D2*D6, D3*D7, D4*D8)
    
    raise NotImplementedError("Not yet implemented for non-square PEPS!")



def build_transfer_matrix_pbc(a, m, b=None):
    """
    constructs the transfer matrix (in the sketch right to left) with an 
    optional impurity b::
    
           +-------+
           |       |
        ---a---    |
           |       |
        ---a---    |
           |       |
                   |
           .       |
           .       |
           .       |
                   |
           |       |
        ---a---    |
           |       |
        ---b---    |
           |       |
           +-------+
        
    """
    if m > 8:
        raise NotImplemented("not implemented for widths larger 8!")

    D0 = a.shape[0]
    D1 = a.shape[1]
    
    if b is None:
        b = a
        
    if m == 1:
        tm = np.einsum(b, [2,0,2,1])
    else:
        aa = np.einsum(a, [0,1,6,4], a, [6,2,3,5]).reshape(D0, D1**2, D0, D1**2)
        ab = np.einsum(a, [0,1,6,4], b, [6,2,3,5]).reshape(D0, D1**2, D0, D1**2)
        
        if m == 2:
            tm = np.einsum(ab, [2,0,2,1])
        elif m == 3:
            tm = np.einsum(ab, [5,0,4,2], a, [4,1,5,3]).reshape(D1**3, D1**3)
        elif m == 4:
            tm = np.einsum(ab, [5,0,4,2], aa, [4,1,5,3]).reshape(D1**4, D1**4)
        else:
            a4 = np.einsum(aa, [0,1,6,4], aa, [6,2,3,5]).reshape(D0, D1**4, D0, D1**4)
            if m == 5:
                tm = np.einsum(a4, [5,0,4,2], b, [4,1,5,3]).reshape(D1**5, D1**5)
            elif m == 6:
                tm = np.einsum(a4, [5,0,4,2], ab, [4,1,5,3]).reshape(D1**6, D1**6)
            else:
                a6 = np.einsum(a4, [0,1,6,4], aa, [6,2,3,5]).reshape(D0, D1**6, D0, D1**6)
                if m == 7:
                    tm = np.einsum(a6, [5,0,4,2], b, [4,1,5,3]).reshape(D1**7, D1**7)
                elif m == 8:
                    tm = np.einsum(a6, [5,0,4,2], ab, [4,1,5,3]).reshape(D1**8, D1**8)
    return tm

def contract_finite_lattice_pbc(a, m, n, b=None):
    m, n = min(m,n), max(m,n)
    tma = build_transfer_matrix_pbc(a, m)
    tmb = build_transfer_matrix_pbc(a, m, b)
    tmp = np.linalg.matrix_power(tma, n-1)
    return tdot(tmp, tmb, [[0,1],[1,0]]) / tdot(tmp, tma, [[0,1],[1,0]])

def overlap_finite_lattice_pbc(a, b, m, n):
    A = make_double_layer(a)
    B = make_double_layer(b)
    C = make_double_layer(b, a)
    tmA = build_transfer_matrix_pbc(A, m)
    tmB = build_transfer_matrix_pbc(B, m)
    tmC = build_transfer_matrix_pbc(C, m)
    psipsi = np.trace(np.linalg.matrix_power(tmA, n))
    phiphi = np.trace(np.linalg.matrix_power(tmB, n))
    psiphi = np.trace(np.linalg.matrix_power(tmC, n))
    return psiphi / np.sqrt(psipsi * phiphi)

def increase_bond_dimension(a, D):
    D0 = a[0].shape[1]
    U = np.identity(D)[:D0]
    return map(lambda a: tdot(tdot(tdot(tdot(a, U, [1,0]), U, [1,0]), U, [1,0]), U, [1,0]), a)
        

def save(a, lut, filename):
    f = open(filename, "w")
    
    n = len(a)
    f.write("{:d}\n".format(n))
    
    first = True
    for j in xrange(n):
        if not first:
            f.write(" ")
        first = False
        f.write("{0:d}".format(lut[j,1,0]))
    f.write("\n")
        
    first = True
    for j in xrange(n):
        if not first:
            f.write(" ")
        first = False
        f.write("{0:d}".format(lut[j,0,1]))
    f.write("\n")
    
    for j in xrange(n):
        f.write("{0:d}".format(j))
        for D in a[j].shape:
            f.write(" {0:d}".format(D))
        f.write("\n")
        first = True
        for k in np.ndindex(a[j].shape):
            if not first:
                f.write(" ")
            first = False
            f.write("{0:.15e}".format(a[j][k]))
        f.write("\n")
    
    f.close()

def load(filename, lut_minsize=[4,4], dtype=float):
    j = 0
    with open(filename, "r") as f:
        for line in f:
            fields = line.split(" ")
            if j == 0:
                n = int(fields[0])
                a = [None]*n
            elif j == 1:
                nnsx = map(int, fields)
            elif j == 2:
                nnsy = map(int, fields)
            elif j % 2 == 1:
                if len(fields) == 6:
                    i = int(fields[0])
                    a[i] = np.ndarray(map(int, fields[1:]), dtype=dtype)
            else:
                l = 0
                for k in np.ndindex(a[i].shape):
                    a[i][k] = dtype(fields[l])
                    l += 1
            j += 1
    return a, [nnsx, nnsy]

