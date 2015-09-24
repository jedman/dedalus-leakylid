import numpy as np 
import h5py 
import dedalus_plots as dp 
from scipy import stats

sim_name = 'k10m1'
filepath = sim_name + "/" + sim_name + "_s1/" + sim_name + "_s1_p0.h5"
data = h5py.File(filepath, "r")
dict_vars = {'b3d':'buoyancy', 'tropenergy':'tropo energy'}

vars = dp.read_vars(data, dict_vars)
dims = dp.read_dims(data)
data.close()

#parameters 
m = 1 
k = 10. 
N1 = 0.01 
eps = 1
Lx, Lz = (2000000, 10000)


tau_approx = Lx*np.pi*m**2/(2.*Lz*eps*N1*k)
tau_exact = tau_approx + eps * (Lx/Lz) * (2. * (m*np.pi)**2 - 3.)/(12. * N1 * k * np.pi)

tau_off = Lx*(6 + np.pi**2*m**2*(1.+3.*eps**2))/(6.*eps*N1*k*np.pi*Lz)


energ_normed = vars['tropenergy'][:,0,0]/np.max(vars['tropenergy'][:,0,0]) 
energ_theory = np.exp(-(dims['t'] )/tau_exact)
energ_approx  = np.exp(-(dims['t'])/tau_approx)
energ_off  = np.exp(-(dims['t'] )/tau_off)
dp.make_1D_plot('energytest.pdf', dims['t'], simulation = energ_normed, theory = energ_theory, off = energ_off)  
dp.make_1D_plot('logenergytest.pdf', dims['t'], simulation = np.log(energ_normed), theory = np.log(energ_theory))  

