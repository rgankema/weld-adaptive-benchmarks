
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ctypes
from weld.weldobject import *
from weld.types import *
from weld.encoders import NumpyArrayEncoder, ScalarDecoder
import weld.bindings as cweld
from collections import namedtuple
from pprint import pprint
import sys
import json
from timeit import default_timer as timer
from pprint import pprint

# Create data
def generate_data(n_R, hit_S, hit_T, hit_U):
    R_a = np.arange(n_R, dtype='int64')
    R_z = np.arange(n_R, dtype='int64')

    n_S = int(hit_S * n_R)
    S_a = np.random.choice(R_a, n_S, replace=False) if n_R > 0 else np.empty(shape=0, dtype='int64')
    S_b = np.arange(n_S, dtype='int64')

    n_T = int(hit_T * n_S)
    T_b = np.random.choice(S_b, n_T, replace=False) if n_T > 0 else np.empty(shape=0, dtype='int64')
    T_c = np.arange(n_T, dtype='int64')

    n_U = int(hit_U * n_T)
    U_c = np.random.choice(T_c, n_U, replace=False) if n_U > 0 else np.empty(shape=0, dtype='int64')
    U_d = np.arange(n_U, dtype='int64')

    return [R_a, R_z, S_a, S_b, T_b, T_c, U_c, U_d]

# Perform the join in Python, check if hit ratios are accurate
def join_python(R_a, R_z, S_a, S_b, T_b, T_c, U_c, U_d):
    start = timer()

    S_ht = {}
    T_ht = {}
    U_ht = {}

    for (sa, sb) in zip(S_a, S_b):
        S_ht[sa] = sb

    for (tb, tc) in zip(T_b, T_c):
        T_ht[tb] = tc
        
    for (uc, ud) in zip(U_c, U_d):
        U_ht[uc] = ud

    aggregate = 0
    s_hit = 0.0
    s_try = 0.0
    t_hit = 0.0
    t_try = 0.0
    u_hit = 0.0
    u_try = 0.0
    hits = 0
    for (ra, rz) in zip(R_a, R_z):
        sb = S_ht.get(ra)
        s_try += 1.0
        if (sb != None):
            s_hit += 1.0
            tc = T_ht.get(sb)
            t_try += 1.0
            if (tc != None):
                t_hit += 1.0
                ud = U_ht.get(tc)
                u_try += 1.0
                if (ud != None):
                    u_hit += 1.0
                    hits += 1
                    aggregate += (ra + sb + tc + ud + rz)
    end = timer()

    print("S hit ratio: " + (str(s_hit / s_try) if s_try > 0 else str(0)))
    print("T hit ratio: " + (str(t_hit / t_try) if t_try > 0 else str(0)))
    print("U hit ratio: " + (str(u_hit / u_try) if u_try > 0 else str(0)))
    print("Hits: " + str(hits))

    return (aggregate, end - start)

# Create the args object for Weld
def args_factory(encoded):
    class Args(ctypes.Structure):
        _fields_ = [e for e in encoded]
    return Args 

# Join the tables using Weld
def join_weld(values, adaptive, lazy, threads):
    weld_code = None
    with open('join.weld', 'r') as content_file:
        weld_code = content_file.read()

    enc = NumpyArrayEncoder()
    names = ['Ra', 'Rz', 'Sa', 'Sb', 'Tb', 'Tc', 'Uc', 'Ud']
    argtypes = [enc.py_to_weld_type(x).ctype_class for x in values]
    encoded = [enc.encode(x) for x in values]

    Args = args_factory(zip(names, argtypes))
    weld_args = Args()
    for name, value in zip(names, encoded):
        setattr(weld_args, name, value)

    void_ptr = ctypes.cast(ctypes.byref(weld_args), ctypes.c_void_p)
    arg = cweld.WeldValue(void_ptr)

    # Compile the module
    err = cweld.WeldError()
    conf = cweld.WeldConf()
    conf.set("weld.optimization.applyExperimentalTransforms", "true" if adaptive else "false")
    conf.set("weld.adaptive.lazyCompilation", "true" if lazy else "false")
    conf.set("weld.threads", str(threads))

    comp_start = timer()
    module = cweld.WeldModule(weld_code, conf, err)
    comp_time = timer() - comp_start

    if err.code() != 0:
        raise ValueError("Could not compile function {}: {}".format(
            weld_code, err.message()))

    # Run the module
    dec = ScalarDecoder()
    restype = WeldLong()
    err = cweld.WeldError()

    exec_start = timer()
    weld_ret = module.run(conf, arg, err)
    exec_time = timer() - exec_start

    if err.code() != 0:
        raise ValueError(("Error while running function,\n{}\n\n"
                        "Error message: {}").format(
            weld_code, err.message()))

    ptrtype = POINTER(restype.ctype_class)
    data = ctypes.cast(weld_ret.data(), ptrtype)
    result = dec.decode(data, restype)
    
    weld_ret.free()
    arg.free()

    return (result, comp_time, exec_time)

# Program
with open('conf.json') as f:
    conf = json.load(f)

pprint(conf)
num_rows = conf['num_rows']
sfs = conf['sf']
num_iters = conf['num_iterations']
s_hits = conf['s_hit']
t_hit = conf['t_hit']
u_hit = conf['u_hit']
types = conf['type']
num_threads = conf['num_threads']

with open('results.csv', 'w') as f:
    f.write('type,n_rows,sf,s_hit,t_hit,u_hit,threads,comp_time,exec_time\n')
    for sf in sfs:
        for s_hit in s_hits:
            data = generate_data(num_rows * sf, s_hit, t_hit, u_hit)
            for t in types:
                adaptive = t == 'Adaptive' or t == 'Lazy'
                lazy = t == 'Lazy'
                for threads in num_threads:
                    last_result = None
                    print('%s, %d, %d, %f, %f, %f, %d' % (t, num_rows, sf, s_hit, t_hit, u_hit, threads))
                    for i in range(num_iters):
                        (result, comp_time, exec_time) = join_weld(data, adaptive, lazy, threads)
                        assert(last_result == None or last_result == result)
                        last_result = result

                        row = '%s,%d,%d,%f,%f,%f,%d,%f,%f\n'  % (t, num_rows, sf, s_hit, t_hit, u_hit, threads, comp_time, exec_time)
                        f.write(row)