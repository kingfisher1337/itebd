import numpy as np
from numpy import dot
from scipy.linalg import expm

delta = np.array([[1.,0],[0,1]])
delta.flags.writeable = False

sigmax = np.array([[0.0, 1.0], [1.0, 0.0]])
sigmax.flags.writeable = False

sigmay = np.array([[0.0, 1j], [-1j, 0.0]])
sigmay.flags.writeable = False

sigmaz = np.diag([-1.,1])
sigmaz.flags.writeable = False

sigmap = np.array([[0., 0], [1, 0]])
sigmap.flags.writeable = False

sigmam = np.array([[0., 1], [0, 0]])
sigmam.flags.writeable = False

def odot2(a, b):
    sa, sb = a.shape, b.shape
    return np.outer(a, b).reshape(sa[0], sa[1], sb[0], sb[1]).swapaxes(1, 2)

sigmaxsigmax = odot2(sigmax, sigmax)
sigmaxsigmax.flags.writeable = False
sigmaysigmay = np.real(odot2(sigmay, sigmay))
sigmaysigmay.flags.writeable = False
sigmazsigmaz = odot2(sigmaz, sigmaz)
sigmazsigmaz.flags.writeable = False

sigmapsigmam = odot2(sigmap, sigmam)
sigmapsigmam.flags.writeable = False
sigmamsigmap = odot2(sigmam, sigmap)
sigmamsigmap.flags.writeable = False

def odot3(a, b, c):
    sa, sb, sc = a.shape, b.shape, c.shape
    return np.outer(np.outer(a, b), c).reshape(sa[0], sa[1], sb[0], sb[1], sc[0], sc[1]).transpose([0,2,4,1,3,5])

sigmatripleproduct = odot3(sigmax, sigmay, sigmaz) - odot3(sigmax, sigmaz, sigmay) + odot3(sigmay, sigmaz, sigmax) - odot3(sigmay, sigmax, sigmaz) + odot3(sigmaz, sigmax, sigmay) - odot3(sigmaz, sigmay, sigmax)

def exp_one_body_gate(g):
    return expm(g)

def exp_two_body_gate(g):
    p1, p2 = g.shape[:2]
    g = g.reshape(p1*p2, p1*p2)
    g = expm(g)
    return g.reshape(p1, p2, p1, p2)

def exp_sigmax(alpha):
    """
    gives the tensor-elements g_{j1,j2} of the one-particle gate
    <j1|exp(alpha sigma_x)|j2>
    """
    c = np.cosh(alpha)
    s = np.sinh(alpha)
    return np.array([[c,s],[s,c]])

def exp_sigmay(a):
    """
    returns A_jk = <j|exp(a*sigmay)|k>
    """
    t = np.ndarray((2,2), dtype=complex)
    t[0,0] = t[1,1] = np.cosh(a)
    t[0,1] = 1j*np.sinh(a)
    t[1,0] = -1j*np.sinh(a)
    return t

def exp_sigmaz(alpha):
    """
    gives the tensor-elements g_{j1,j2} of the one-particle gate
    <j1|exp(alpha sigma_z)|j2>
    """
    return np.array([[np.exp(-alpha),0],[0,np.exp(alpha)]])

def exp_sigmax_sigmax(alpha):
    """
    gives the tensor-elements g_{j1,j2,k1,k2} of the two-particle gate
    <j1,k1|exp(alpha sigma_x sigma_x)|j2,k2>
    """
    g = np.zeros((2,2,2,2), dtype=type(alpha))
    g[0,0,0,0] = g[0,1,0,1] = g[1,0,1,0] = g[1,1,1,1] = np.cosh(alpha)
    g[0,0,1,1] = g[1,1,0,0] = g[0,1,1,0] = g[1,0,0,1] = np.sinh(alpha)
    return g

def exp_sigmay_sigmay(alpha):
    g = np.zeros((2,2,2,2))
    g[0,0,0,0] = g[0,1,0,1] = g[1,0,1,0] = g[1,1,1,1] = np.cosh(alpha)
    g[0,0,1,1] = g[1,1,0,0] = -np.sinh(alpha)
    g[0,1,1,0] = g[1,0,0,1] = np.sinh(alpha)
    return g
    #return exp_two_body_gate(alpha * sigmaysigmay)

def exp_sigmaz_sigmaz(alpha):
    """
    gives the tensor-elements g_{j,k,l,m} of the two-particle gate
    <j,k|exp(alpha sigma_z sigma_z)|l,m> 
    = delta_{jl} delta_{km} exp(alpha (-1)^(j+k))

     j   k
     |   |
    +-----+
    |  g  |
    +-----+
     |   |
     l   m
    """
    g = np.zeros((2,2,2,2), dtype=type(alpha))
    g[0,0,0,0] = g[1,1,1,1] = np.exp(alpha)
    g[0,1,0,1] = g[1,0,1,0] = np.exp(-alpha)
    return g

def exp_sigma_sigma(alpha):
    x = exp_sigmax_sigmax(alpha).reshape(4, 4)
    y = exp_sigmay_sigmay(alpha).reshape(4, 4)
    z = exp_sigmaz_sigmaz(alpha).reshape(4, 4)
    return dot(dot(x, y), z)

def build_operator_block(o):
    b = o[0]
    for j in xrange(1, len(o)):
        b = np.outer(b, o[j])
    return b.transpose([2*j for j in xrange(len(o))] + [2*j+1 for j in xrange(len(o))])

def embed_one_body_gate_into_block(g, j, n, p):
    # example: j=1, n=4 returns (id odot g odot id odot id)
    I = np.identity(p)
    return build_operator_block([I]*j + [g] + [I]*(n-j-1)).reshape(g.shape[0]*p**(n-1), g.shape[1]*p**(n-1))

def exp_sigmaz_sigmaz_mpo(alpha):
    """
        i
        |
       +-+
    l--|g|--j
       +-+
        |
        k
    """
    if alpha < 0:
        alpha = alpha*1j
    c = np.sqrt(np.cosh(alpha))
    s = np.sqrt(np.sinh(alpha))
    w = np.array([[c,s],[c,-s]])
    w2 = np.fromfunction(lambda j,k,l: delta[j,k]*w[j,l], (2,2,2), dtype=int)
    return np.einsum(w2, [0,4,1], w2.conj(), [4,2,3])
    
    #f = np.array([np.sqrt(np.sinh(alpha)),np.sqrt(np.cosh(alpha))])
    #if alpha > 0:
    #    return np.fromfunction(lambda i,j,k,l: delta[i,k]*f[j]*f[l]*(-1)**((j+l)*i), (2,2,2,2), dtype=int)
    #else:
    #    raise ValueError("can not handle negative parameter! would give complex output!")

def exp_sigmaz_sigmaz_pepo_square(alpha):
    m = exp_sigmaz_sigmaz_mpo(alpha)
    return np.einsum(m, [0,2,6,3], m, [6,4,1,5])


def imtime_evolution_from_pair_hamiltonian_mpo(h1, h2, tau):
    p = h1.shape[0]
    u1 = expm(-0.5*tau * h1)
    u2 = expm(-tau * h2.reshape(p*p, p*p)).reshape(p,p,p,p).swapaxes(1,2).reshape(p*p, p*p)
    r = np.linalg.matrix_rank(u2)
    u,s,v = np.linalg.svd(u2)
    u = np.dot(u[:,:r], np.diag(np.sqrt(s[:r]))).reshape(p,p,r)
    v = np.dot(np.diag(np.sqrt(s[:r])), v[:r]).reshape(r,p,p)
    g = np.einsum(u, [0,4,1], v, [3,4,2])
    g = np.einsum(u1, [0,4], g, [4,1,5,3], u1, [5,2])
    return g

def imtime_evolution_from_pair_hamiltonian_pepo(h1, h2, tau):
    g = imtime_evolution_from_pair_hamiltonian_mpo(h1, h2, tau)
    return np.einsum(g, [0,2,6,4], g, [6,3,1,5])



