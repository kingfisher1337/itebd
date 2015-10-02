import numpy as np
from scipy.optimize import minimize
import multiprocessing as mp
import sys

import tebd

def _peps_to_vec(a):
    return np.concatenate(map(lambda b: b.flatten(), a))

def _vec_to_peps(x, n, shape):
    return map(lambda y: y.reshape(shape), np.split(x, n))

def _polish_cost_serial(x, n, shape, lut, ecf, energy_idx, dx):
    ec = ecf.create()
    ec.update(_vec_to_peps(x, n, shape))
    E = ec.get_test_values()[energy_idx]
    
    best = ec.get_test_values()
    
    grad = np.empty(len(x))
    for j in xrange(len(x)):
        y = np.copy(x)
        y[j] += dx
        ec.update(_vec_to_peps(y, n, shape))
        grad[j] = ec.get_test_values()[energy_idx]
        
        if grad[j] < best[energy_idx]:
            best = ec.get_test_values()
        
    grad = (grad - E) / dx
    
    for z in best:
        sys.stdout.write("{0:.15e} ".format(z))
    sys.stdout.write("\n")
    sys.stdout.flush()
    
    return E, grad


def _polish_cost_parallel_helper((x, n, shape, lut, ecf, energy_idx, dx, jmin, jmax, evalf)):
    ec = ecf.create()
    f = np.empty(jmax - jmin + 1 + evalf)
    
    best = [1e100] * (energy_idx+1 if energy_idx > 0 else -energy_idx)
    
    for j in xrange(jmin, jmax+1):
        y = np.copy(x)
        y[j] += dx
        ec.update(_vec_to_peps(y, n, shape))
        f[j-jmin] = ec.get_test_values()[energy_idx]
        
        if f[j-jmin] < best[energy_idx]:
            best = ec.get_test_values()
    
    if evalf:
        ec.update(_vec_to_peps(x, n, shape))
        f[-1] = ec.get_test_values()[energy_idx]
        
        if f[-1] < best[energy_idx]:
            best = ec.get_test_values()
    
    return f, best

def _polish_cost_parallel(x, n, shape, lut, ecf, energy_idx, dx, num_workers):
    num_fct_evals = len(x) + 1
    num_fct_evals_per_worker = num_fct_evals / num_workers
    if num_fct_evals_per_worker * num_workers < num_fct_evals:
        num_fct_evals_per_worker += 1
    
    paramlist = [
        (x, n, shape, lut, ecf, energy_idx, dx, j*num_fct_evals_per_worker, min((j+1)*num_fct_evals_per_worker-1, len(x)-1), j+1==num_workers)
        for j in xrange(num_workers)
    ]
    
    pool = mp.Pool(processes=num_workers)
    res = pool.map(_polish_cost_parallel_helper, paramlist)
    pool.close()
    
    tmp = np.concatenate(map(lambda x: x[0], res))
    E = tmp[-1]
    grad = (tmp[:-1] - E) / dx
    
    for z in res[np.argmin(map(lambda x: x[1][energy_idx], res))][1]:
        sys.stdout.write("{0:.15e} ".format(z))
    sys.stdout.write("\n")
    sys.stdout.flush()
    
    return E, grad

def polish(a, lut, env_contractor_factory, energy_idx=-1, num_workers=1):
    dx = 1.4901161193847656e-08
    if num_workers == 1:
        res = minimize(
            _polish_cost_serial, 
            _peps_to_vec(a), 
            jac=True, 
            args=(len(a), a[0].shape, lut, env_contractor_factory, energy_idx, dx), 
            method="BFGS", options={"disp":True})
        return _vec_to_peps(res.x, len(a), a[0].shape)
    else:
        res = minimize(
            _polish_cost_parallel, 
            _peps_to_vec(a), 
            jac=True, 
            args=(len(a), a[0].shape, lut, env_contractor_factory, energy_idx, dx, num_workers), 
            method="BFGS", options={"disp":True})
        return _vec_to_peps(res.x, len(a), a[0].shape)

