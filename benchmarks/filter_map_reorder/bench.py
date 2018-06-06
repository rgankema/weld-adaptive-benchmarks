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

# Create first column of data (only one that depends on selectivity)
def generate_select_col(num_rows, selectivity):
    hits = np.ones(int(selectivity * num_rows), dtype=np.int64) * 42
    misses = np.ones(num_rows - len(hits), dtype=np.int64)
    col = np.append(hits, misses)
    np.random.shuffle(col)
    return col

# Create remaining columns
def generate_static_cols(num_rows):
    data = {}
    for i in range(2,7):
        data['in%d' % i] = np.random.randint(0, 42, num_rows, dtype=np.int64)

    return data

# Returns an array of selectivity data 
def selectivities(min, max, num_points):
    arr = np.ones(10) - np.geomspace(0.01, 1, 10)
    arr[0] = 1
    arr = np.flip(arr, 0)
    return arr

# Create the args object for Weld
def args_factory(encoded):
    class Args(ctypes.Structure):
        _fields_ = [e for e in encoded]
    return Args 

# Perform the Weld operation
def benchmark(data, type, threads, weld_conf):
    adaptive = type == 'Adaptive'
    lazy = False
    code_path = 'filter_then_map.weld' if type is not 'Map->Filter' else 'map_then_filter.weld'

    weld_code = None
    with open(code_path, 'r') as content_file:
        weld_code = content_file.read()

    enc = NumpyArrayEncoder()
    names = [c for c in sorted(data)]
    argtypes = [enc.py_to_weld_type(data[c]).ctype_class for c in names]
    encoded = [enc.encode(data[c]) for c in names]

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
        description="Micro benchmark for adaptive filter map ordering"
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
    min_select = conf['min_select']
    max_select = conf['max_select']
    num_points = conf['num_points']
    types = conf['type']
    num_threads = conf['num_threads']
    weld_conf = conf.get('weld_conf')

    # Start benchmark
    total_iters = len(sfs) * num_points * len(num_threads) * len(types)
    iters = 1
    with open(out_path, 'w') as f:
        f.write('type,n_rows,sf,selectivity,threads,comp_time,exec_time\n')
        for sf in sfs:
            data = generate_static_cols(num_rows * sf)
            for select in selectivities(min_select, max_select, num_points):
                data['in1'] = generate_select_col(num_rows * sf, select)
                for th in num_threads:
                    for ty in types:
                        print('[%03d/%03d] %s, %d, %d, %.3f, %d' % (iters, total_iters, ty, num_rows, sf, select, th))
                        for i in range(num_iters):
                            (result, comp_time, exec_time) = benchmark(data, ty, th, weld_conf)
                            row = '%s,%d,%d,%f,%d,%f,%f\n'  % (ty, num_rows, sf, select, th, comp_time, exec_time)
                            f.write(row)
                        iters += 1
                            
