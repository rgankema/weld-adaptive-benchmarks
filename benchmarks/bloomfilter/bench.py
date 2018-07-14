
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
def generate_data(n_total, f_unique):
    data = np.random.choice(np.arange(n_total * f_unique, dtype='int64'), n_total, replace=True)
    np.random.shuffle(data)
    return [data]

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
    names = ['x']
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
    
    num_iters = conf['num_iters']
    n_total = conf['n_total']
    f_uniques = conf['f_unique']
    sfs = conf['sf']
    num_threads = conf['num_threads']
    types = conf['type']
    weld_conf = conf.get('weld_conf')

    # Start benchmarking
    total_iters = len(sfs) * len(f_uniques) * len(types) * len(num_threads)
    iters = 1
    with open(out_path, 'w') as f:
        f.write('type,n_rows,sf,f_unique,threads,comp_time,exec_time\n')
        for sf in sfs:
            for f_unique in f_uniques:
                data = generate_data(n_total * sf, f_unique)
                for t in types:
                    for threads in num_threads:
                        print('[%03d/%03d] %s, %d, %d, %.3f, %d' % (iters, total_iters, t, n_total, sf, f_unique, threads))
                        for i in range(num_iters):
                            (res, comp_time, exec_time) = join_weld(data, t, threads, weld_conf)
                            row = '%s,%d,%d,%f,%d,%f,%f\n'  % (t, n_total, sf, f_unique, threads, comp_time, exec_time)
                            f.write(row)
                        iters += 1