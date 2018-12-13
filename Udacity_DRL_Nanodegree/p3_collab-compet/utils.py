#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from beautifultable import BeautifulTable as BT


# Decorator to time function executions
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed

def timer(t0):
    deltatime = time.time() - t0
    if deltatime > 60:
        return deltatime / 60
    return deltatime
        
        
# Count parameters of a model 
def count_parameters(model):
    ''' Count the parameters of a model '''
    return sum(p.numel() for p in model.parameters())


# Create Table
def create_table(state, names):
    table = BT()
    table.column_headers = names 
    if state.shape[0] == 8:
        table.append_row(state.tolist())
    else:
        [table.append_row(state[i].tolist()) for i in range(state.shape[0])]
    return table