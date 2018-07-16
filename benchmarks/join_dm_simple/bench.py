
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
import glob 
import shutil

# Create data
def generate_data(n_R, s_to_r, hit_S):
    n_S = int(math.ceil(s_to_r * n_R))
    S_b = np.arange(n_S, dtype='int64')
    S_c = np.arange(n_S, dtype='int64')

    R_b = np.random.choice(S_b, n_R, replace=True)
    R_b[int(math.ceil(n_R*hit_S)):] += n_S
    R_a = np.arange(n_R, dtype='int64')

    columns = [R_a, R_b, S_b, S_c]
    for col in columns:
        np.random.shuffle(col)
    return columns

# Perform the join in Python, check if hit ratios are accurate
def join_python(R_a, R_b, S_b, S_c):
    S_ht = {}
    for (b, c) in zip(S_b, S_c):
        S_ht[b] = c

    aggregate = int(0)
    s_hit = 0.0
    s_try = 0.0
    hits = 0
    for (a, b) in zip(R_a, R_b):
        c = S_ht.get(b)
        s_try += 1.0
        if (c != None):
            s_hit += 1.0
            aggregate += (a + b + c)
    end = timer()

    return aggregate

# Create the args object for Weld
def args_factory(encoded):
    class Args(ctypes.Structure):
        _fields_ = [e for e in encoded]
    return Args 

# Join the tables using Weld
def join_weld(values, ty, threads, weld_conf):
    adaptive = ty == 'Adaptive' or ty == 'Lazy'
    lazy = ty == 'Lazy'
    file_path = 'join_bf.weld' if ty == 'Bloom Filter' else 'join.weld'
    
    weld_code = None
    with open(file_path, 'r') as content_file:
        weld_code = content_file.read()

    enc = NumpyArrayEncoder()
    names = ['R_a', 'R_b', 'S_b', 'S_c']
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
    conf.set("weld.optimization.applyAdaptiveTransforms", "true" if adaptive else "false")
    conf.set("weld.adaptive.lazyCompilation", "true" if lazy else "false")
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
    s_to_rs = conf['s_to_r']
    types = conf['type']
    num_threads = conf['num_threads']
    weld_conf = conf.get('weld_conf')

    # Start benchmarking
    total_iters = len(sfs) * len(s_hits) * len(types) * len(num_threads) * len(s_to_rs)
    iters = 1
    with open(out_path, 'w') as f:
        f.write('type,n_rows,sf,s_to_r,s_hit,threads,comp_time,exec_time\n')
        for sf in sfs:
            for s_to_r in s_to_rs:
                for s_hit in s_hits:
                    data = generate_data(num_rows * sf, s_to_r, s_hit)
                    expect = join_python(data[0], data[1], data[2], data[3])
                    for t in types:
                        for threads in num_threads:
                            print('[%03d/%03d] %s, %d, %d, %.3f, %.3f, %d' % (iters, total_iters, t, num_rows, sf, s_to_r, s_hit, threads))
                            for i in range(num_iters):
                                (result, comp_time, exec_time) = join_weld(data, t, threads, weld_conf)
                                assert(result == expect)

                                row = '%s,%d,%d,%f,%f,%d,%f,%f\n'  % (t, num_rows, sf, s_to_r, s_hit, threads, comp_time, exec_time)
                                f.write(row)
                            iters += 1

                            # Move profiling stuff if exists
                            if weld_conf is not None and weld_conf.get('weld.log.profile') == 'true':
                                for file in glob.glob(r'profile-*.csv'):
                                    s_to_r_str = ("%.3g" % s_to_r).replace('.', '')
                                    s_hit_str = ("%.3g" % s_hit).replace('.', '')
                                    shutil.move(file, 'prof-%s_%d_%d_%s_%s_%d_%d.csv' % (t, num_rows, sf, s_to_r, s_hit, threads, i))

                                    