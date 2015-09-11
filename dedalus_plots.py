import numpy as np
import matplotlib.pyplot as plt
import h5py

def read_vars(data, vars ):
    '''read in var in vars from data (h5pyfile)  and return dict selected_vars'''  
    selected_vars = {} 
    for key, varname in vars.items():
        # read in data 
        tmp = data['tasks'][varname][:]  
        # make a new dictionary with data
        selected_vars[key] = tmp 
    return selected_vars 

def read_dims(data): 
    '''read in x, z, t dims from data (h5pyfile)  and return dict all_dims'''  
    all_dims = {} 
    all_dims['x'] = data['scales/x/1.0'][:]
    all_dims['z'] = data['scales/z/1.0'][:] 
    all_dims['t'] = data['scales']['sim_time'][:]
    return all_dims

