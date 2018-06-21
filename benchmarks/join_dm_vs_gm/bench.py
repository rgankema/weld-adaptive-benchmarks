
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
import argparse
import math

# Create data
def generate_data(n_R, hit_S, hit_T, hit_U):
    n_S = int(math.ceil(0.2 * n_R))
    n_T = int(math.ceil(0.05 * n_R))
    n_U = int(math.ceil(0.001 * n_R))

    U_uk = np.arange(n_U, dtype='int64')
    U_val = np.arange(n_U, dtype='int64')

    T_uk = np.random.choice(U_uk, n_T, replace=True)
    T_uk[int(math.ceil(n_T*hit_U)):] += n_U
    T_tk = np.arange(n_T, dtype='int64')

    S_tk = np.random.choice(T_tk, n_S, replace=True)
    S_tk[int(math.ceil(n_S*hit_T)):] += n_T
    S_sk = np.arange(n_S, dtype='int64')

    R_sk = np.random.choice(S_sk, n_R, replace=True)
    R_sk[int(math.ceil(n_R*hit_S)):] += n_S
    R_rk = np.arange(n_R, dtype='int64')

    columns = [R_rk, R_sk, S_sk, S_tk, T_tk, T_uk, U_uk, U_val]
    for col in columns:
        np.random.shuffle(col)
    return columns

# Create a dictionary with the values grouped by the keys
def group_by_key(keys, vals):
    grouped = {}
    for k, v in zip(keys, vals):
        group = grouped.get(k)
        if group is None:
            group = []
        group.append(v)
        grouped[k] = group
    return grouped

# Perform the join in Python, check if hit ratios are accurate
def join_python(R_rk, R_sk, S_sk, S_tk, T_tk, T_uk, U_uk, U_val):
    S_ht = group_by_key(S_sk, S_tk)
    T_ht = group_by_key(T_tk, T_uk)
    U_ht = {}
    for (uk, uval) in zip(U_uk, U_val):
        U_ht[uk] = uval

    aggregate = int(0)
    s_hit = 0.0
    s_try = 0.0
    t_hit = 0.0
    t_try = 0.0
    u_hit = 0.0
    u_try = 0.0
    hits = 0
    for (rk, sk) in zip(R_rk, R_sk):
        tks = S_ht.get(sk)
        s_try += 1.0
        if (tks != None):
            s_hit += 1.0
            for tk in tks:
                uks = T_ht.get(tk)
                t_try += 1.0
                if (uks != None):
                    t_hit += 1.0
                    for uk in uks:
                        uval = U_ht.get(uk)
                        u_try += 1.0
                        if (uval != None):
                            u_hit += 1.0
                            hits += 1
                            aggregate += uval
    end = timer()

    print("S hit ratio: " + (str(s_hit / s_try) if s_try > 0 else str(0)))
    print("T hit ratio: " + (str(t_hit / t_try) if t_try > 0 else str(0)))
    print("U hit ratio: " + (str(u_hit / u_try) if u_try > 0 else str(0)))
    print("Hits: " + str(hits))

    return aggregate

# Create the args object for Weld
def args_factory(encoded):
    class Args(ctypes.Structure):
        _fields_ = [e for e in encoded]
    return Args 

# Join the tables using Weld
def join_weld(values, ty, threads, weld_conf):
    file_path = '%s.weld' % ty
    weld_code = None
    with open(file_path, 'r') as content_file:
        weld_code = content_file.read()

    enc = NumpyArrayEncoder()
    names = ['R_rk', 'R_sk', 'S_sk', 'S_tk', 'T_tk', 'T_uk', 'U_uk', 'U_val']
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
    conf.set("weld.optimization.applyAdaptiveTransforms", "false")
    conf.set("weld.adaptive.lazyCompilation", "false")
    conf.set("weld.threads", str(threads))
    conf.set("weld.memory.limit", "20000000000")
    if weld_conf is not None:
        for key, val in weld_conf.iteritems():
            conf.set(key, val)

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

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Micro benchmark for adaptive joins"
    )
    parser.add_argument('-c', '--conf', type=str, required=True,
                        help="Path to configuration file")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Path to output file")
    cmdline_args = parser.parse_args()
    opt_dict = vars(cmdline_args)
    conf_path = opt_dict['conf']
    out_path = opt_dict['output']

    # Parse configuration file
    with open(conf_path) as f:
        conf = json.load(f)
    num_rows = conf['num_rows']
    sfs = conf['sf']
    num_iters = conf['num_iterations']
    s_hits = conf['s_hit']
    t_hit = conf['t_hit']
    u_hit = conf['u_hit']
    types = conf['type']
    num_threads = conf['num_threads']
    weld_conf = conf.get('weld_conf')

    # Start benchmarking
    total_iters = len(sfs) * len(s_hits) * len(types) * len(num_threads)
    iters = 1
    with open(out_path, 'w') as f:
        f.write('type,n_rows,sf,s_hit,t_hit,u_hit,threads,comp_time,exec_time\n')
        for sf in sfs:
            for s_hit in s_hits:
                data = generate_data(num_rows * sf, s_hit, t_hit, u_hit)
                expect = join_python(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])
                for t in types:
                    for threads in num_threads:
                        print('[%03d/%03d] %s, %d, %d, %.3f, %.3f, %.3f, %d' % (iters, total_iters, t, num_rows, sf, s_hit, t_hit, u_hit, threads))
                        for i in range(num_iters):
                            (result, comp_time, exec_time) = join_weld(data, t, threads, weld_conf)
                            assert(result == expect)

                            row = '%s,%d,%d,%f,%f,%f,%d,%f,%f\n'  % (t, num_rows, sf, s_hit, t_hit, u_hit, threads, comp_time, exec_time)
                            f.write(row)
                        iters += 1