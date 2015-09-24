import numpy as np
import matplotlib.pyplot as plt
import h5py

plt.rcParams['image.cmap'] = 'RdBu_r'
plt.rcParams.update({'font.size': 16})

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


def make_1D_plot(filename,dim, **kwargs):
    ''' make a line plot''' 

    styles = ['-','--','-.']
    ctr = 0
    for name, var in kwargs.items(): 
        plt.plot(dim, var, label = name, lw = 2, ls = styles[ctr])
        ctr += 1 
    plt.legend() 
    plt.savefig(filename)
    plt.clf() 
    return

def make_2D_plot(filename,dims,var, title = '', xlabel = '', ylabel = ''):
    ''' make a pcolormesh plot '''
    plt.pcolormesh(dims[0] , dims[1], var)
    plt.title(title) 
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel) 
    plt.colorbar() 
    plt.savefig(filename)
    plt.clf() 
    return 


