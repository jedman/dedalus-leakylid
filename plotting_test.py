import numpy as np 
import h5py 
import dedalus_plots as dp 

sim_name = 'k10m1'
filepath = sim_name + "/" + sim_name + "_s1/" + sim_name + "_s1_p0.h5"
data = h5py.File(filepath, "r")
dict_vars = {'te':'total e profile','b3d':'buoyancy', 'tropenergy':'tropo energy'}

vars = dp.read_vars(data, dict_vars)
dims = dp.read_dims(data)
data.close()

#parameters 
pulse_len = 300 
m = 1.5 
k = 10. 
N1 = 0.01 
eps = 1
Lx, Lz = (2000000, 10000)


tau_approx = Lx*np.pi*m**2/(2.*Lz*eps*N1*k)
tau_exact = tau_approx + eps * (Lx/Lz) * (2. * (m*np.pi)**2 - 3.)/(12. * N1 * k * np.pi)

tau_off = Lx*(6 + np.pi**2*m**2*(1.+3.*eps**2))/(6.*eps*N1*k*np.pi*Lz)


energ_normed = vars['tropenergy'][:,0,0]/np.max(vars['tropenergy'][:,0,0]) 
energ_theory = np.exp(-(dims['t'] - pulse_len)/tau_exact)
energ_approx  = np.exp(-(dims['t'] - pulse_len)/tau_approx)
energ_off  = np.exp(-(dims['t'] - pulse_len)/tau_off)
energ_normed_2D = vars['te'][:,0,:].T/np.max(vars['te'][:,0,:].T)  
dp.make_1D_plot('energytest.pdf', dims['t'], simulation = energ_normed, theory = energ_theory, off = energ_off)  
dp.make_2D_plot('tetest.pdf', (dims['t'], dims['z']/1000.), energ_normed_2D, title = 'Total Energy', xlabel = 'time (s)', ylabel = 'height (km)')  

