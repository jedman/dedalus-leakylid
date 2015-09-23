import numpy as np
import matplotlib.pyplot as plt
import h5py
from dedalus import public as de
from dedalus.extras import flow_tools
import time
import argparse 

plt.rcParams['image.cmap'] = 'RdBu_r'

parser = argparse.ArgumentParser(description='simulate a Boussinesq pulse')
parser.add_argument('k', metavar = 'k', type = int, help='forcing wavenumber in the horizontal')
parser.add_argument('m', metavar = 'm', type = int, help='forcing wavenumber in the vertical')
parser.add_argument('eps', metavar = 'eps', type = float, help='epsilon, the ratio of buoyancy frequency in troposphere and stratosphere') 
parser.add_argument('-nh','--non-hstat', dest='hstat', action='store_false')
parser.add_argument('-p','--pulse', dest='pulse', action='store_true')
parser.add_argument('-pl', '--pulse-len', dest = 'pulse_len' , type = float) 
parser.set_defaults(pulse_len=100) 
parser.set_defaults(hstat=True)
parser.set_defaults(pulse=False) 
args = parser.parse_args() 

PULSE = args.pulse
HYDROSTATIC = args.hstat
print('pulse_len is ', args.pulse_len) 

if HYDROSTATIC == True:
     print('using hydrostatic boussinesq solver') 
else:
     print('using non-hydrostatic boussinesq solver') 

if PULSE == True:
     print('solving for gaussian forcing') 
else:
     print('solving for cosine forcing (single k)') 


import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
    
logger = logging.getLogger(__name__)

Lx, Lz = (2000000, 10000) # domain size in meters 
nx, nz = (144, 256)  # number of points in each direction
#Lx, Lz = (4000000, 10000) # domain size in meters 
#nx, nz = (4*64, 144)  # number of points in each direction

# parameters (some of these should be set via command line args) 
stop_time = 20000. # simulation stop time  (seconds)
pulse_len = args.pulse_len # seconds of forcing 
N1 = 0.01 # buoyancy frequency in the troposphere (1/s) 
eps = args.eps # ratio of N1/N2
N2 = N1/eps  # buoyancy frequency in the stratosphere
m = args.m # vertical mode number
k = args.k # horizontal mode number
model_top = 8. * Lz # lid height
if eps < 0.4:
    model_top = 4. * Lz # increases resolution near the jump 
    
sim_name = 'k'+ str(k) +'m' + str(m) 
print('simulation name is', sim_name)  

print('effective forcing horizontal wavelength is' , 2.*Lx/k/1000., 'kilometers')
print('effective forcing vertical wavelength is' , 2.*Lz/m/1000., 'kilometers')
print('stratification ratio N1/N2 is' , N1/N2 )


lambda_x = 2.*Lx/k # for defining width of pulse 


# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(-Lx/2., Lx/2.), dealias = 3/2)
# compound z basis -- better to resolve jump condition?
#zb1 = de.Chebyshev('z1',int(nz/4), interval=(0, Lz+1000), dealias=3/2)
#zb2 = de.Chebyshev('z2', nz, interval=(Lz+1000,model_top), dealias = 3/2)
#z_basis = de.Compound('z',(zb1,zb2), dealias = 3/2)
#
z_basis = de.Chebyshev('z', nz, interval= (0, model_top), dealias = 3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
x, z = domain.grids(scales=1)
xd, zd = domain.grids(scales=domain.dealias)


# set up problem 

problem = de.IVP(domain, variables=['p','u','B','w'])

problem.parameters['rho'] = 1. #kg/m^3 
#problem.parameters['Nsq'] = 0.0001 #1/s; constant Nsq

# non-constant coefficient N^2 
ncc = domain.new_field(name='Nsq')
ncc['g'] = N1**2  
strat = np.where( z > Lz)
ncc['g'][:,strat] = N2**2 
ncc.meta['x']['constant'] = True
problem.parameters['Nsq'] = ncc

# mask (for analysis) 
mask = domain.new_field(name = 'mask')
mask['g'] = 1
mask['g'][:,strat] = 0 
mask.meta['x']['constant'] = True
problem.parameters['mask'] = mask


# experimental source term 
# following Daniel's 12/24/14 explanation in the forums 
# https://groups.google.com/forum/#!topic/dedalus-users/BqTjYZzqHHw
if PULSE == True:
    def forcing(solver):
        # if using dealiasing, it's important to apply the forcing on the dealiased doman (xd,zd)
        if solver.sim_time < pulse_len:
            f = 0.0001*np.sin(m * np.pi*zd/Lz)*np.exp(-16.*(xd*xd)/((lambda_x)**2)) #pulse  with "effective wavelength" lambda_x 
            strat = np.where(zd>Lz)
            f[:,strat] = 0.
           # subtract the horizontal mean at each level so there's no k=0
           # fprof = np.mean(f, axis = 0 )
           # ftmp = np.repeat(fprof, xd.shape[0])
           # fmask = ftmp.reshape(zd.shape[1],xd.shape[0])
           # f = f - fmask.T 
        else:
            f = 0. 
        return f
else:
    def forcing(solver):
        # if using dealiasing, it's important to apply the forcing on the dealiased doman (xd,zd)
        if solver.sim_time < pulse_len:
            #f = 0.001*np.sin(np.pi*zd/Lz)*np.exp(-16.*(xd*xd)/((lambda_x)**2)) #pulse  with "effective wavelength" lambda_x
            f = 0.001*np.sin(m * np.pi*zd/Lz)*np.cos(k* np.pi* xd /Lx) # cosine wave
            strat = np.where(zd>Lz)
            f[:,strat] = 0.
        else:
            f = 0. 
        return f
forcing_func = de.operators.GeneralFunction(domain,'g',forcing, args=[])
forcing_func.build_metadata() 
#forcing_func.meta = ncc.meta # just tricking it for now, this metadata is wrong
# let's make a general parameter and use that metadata instead
dummy = domain.new_field(name='dum')
dummy['g'] = 1.
forcing_func.meta = dummy.meta
problem.parameters['forcing_func'] = forcing_func
# need  to add 'meta' attribute for General Function class 
# otherwise system fails consistency check

# system to solve (2D, linearized,  hydrostatic boussinesq) 
problem.add_equation("dt(u) + 1/rho*dx(p) = 0")
problem.add_equation("dt(B) + Nsq*w  = forcing_func")
problem.add_equation("dx(u) + dz(w) = 0")
if HYDROSTATIC == True:
	problem.add_equation("B - 1/rho*dz(p) = 0")
else:
	problem.add_equation("B - 1/rho*dz(p) - dt(w) = 0")

# fourier direction has periodic bc, chebyshev has a lid
problem.add_bc("left(w) = 0") # refers to the first end point in chebyshev direction
problem.add_bc("right(w) = 0", condition="(nx != 0)") # rigid lid, condition note for k = 0 mode
problem.add_bc("integ(p,'z') = 0", condition="(nx == 0)") # pressure gauge condition for k = 0

# build solver

ts = de.timesteppers.RK443 # arbitrary choice of time stepper
solver =  problem.build_solver(ts)

# tell the forcing function what its arg is (clunky) 
forcing_func.args = [solver]
forcing_func.original_args = [solver]

# initial conditions 
x, z = domain.grids(scales=1)
u = solver.state['u']
w = solver.state['w']
p = solver.state['p']
B = solver.state['B'] # zero for everything


solver.stop_sim_time = stop_time
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# CFL conditions
initial_dt = 0.8*Lz/nz
cfl = flow_tools.CFL(solver,initial_dt,safety=0.8, max_change=1.5, min_change=0.5, max_dt=400) 
# too large of a timestep makes things rather diffusive
cfl.add_velocities(('u','w'))

# analysis
# fields to record 
analysis = solver.evaluator.add_file_handler(sim_name, sim_dt=50, max_writes=300)
analysis.add_task('B', name = 'buoyancy' )
analysis.add_task('w', name = 'vertical velocity')
analysis.add_task('u', name = 'horizontal velocity')
analysis.add_task('p', name = 'pressure')
analysis.add_task('Nsq')
analysis.add_task('0.5*(w*w + u*u * B*B/Nsq)', name = 'total e snap')
analysis.add_task('0.5*mask*(w*w + u*u * B*B/Nsq)', layout = 'c', name = 'total e coeffs')
analysis.add_task('-dx(u)', name = 'convergence') 

# profiles
analysis.add_task("integ(0.5 * B*B/Nsq, 'x')", name='pe profile')
analysis.add_task("integ(0.5 * (u*u + w*w) , 'x')", name='ke profile')
analysis.add_task("integ(0.5 * (u*u + w*w +  B*B/Nsq ),'x')", name='total e profile')
analysis.add_task("integ(B,'x')", name = 'b profile')
analysis.add_task("integ(B * mask,'x')", name = 'mask test')
# 1d fields
analysis.add_task('mask')
analysis.add_task("integ(B * mask)", name = 'tropo b')
analysis.add_task("integ(0.5 * mask *(u*u + w*w +  B*B/Nsq ))", name = 'tropo energy') # use mask to integrate over troposphere only
analysis.add_task("integ(0.5 * (u*u + w*w +  B*B/Nsq ))", name='total e')

try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = cfl.compute_dt()
        solver.step(dt)
        if solver.iteration % 20 == 0:
             print('Completed iteration {}'.format(solver.iteration))
             print('simulation time {}'.format(solver.sim_time))
except: 
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    # Print statistics
    logger.info('Run time: %f' %(end_time-start_time))
    logger.info('Iterations: %i' %solver.iteration)

# save a plot or two 
filepath = sim_name + "/" + sim_name + "_s1/" + sim_name + "_s1_p0.h5"
print(filepath) 

# document simulation parameters 
filename = sim_name + '/' + sim_name + 'simulation_params' 
f = open(filename, 'w')
ln1 = 'epsilon: ' + str(eps) + '\n'
ln2 = 'Lx: ' + str(Lx) + ' km\n' 
ln3 = 'Lz: ' + str(Lz) + ' km\n' 
ln4 = 'k = ' + str(k) + '; m = ' + str(m) + '\n'
ln5 = ' The effective forcing horizontal wavelength is ' + str(2.*Lx/k/1000.) + ' kilometers.\n'
ln6 = ' The effective forcing vertical wavelength is ' + str(2.*Lz/m/1000.) + ' kilometers.\n'
ln7 = ' The pulse length is ' + str(pulse_len) + ' seconds. \n' 
ln8 = ' The lid is at' + str(model_top/1000.)  + ' kilometers.\n' 
lines = ln1 + ln2 + ln3 + ln4 + ln5 + ln6 + ln7 + ln8
f.write(lines)  
f.close() 

# import plotting tools
import dedalus_plots as dp 

# read in required data


data = h5py.File(filepath, "r")
#data = h5py.File("analysis_tasks/analysis_tasks_s1/analysis_tasks_s1_p0.h5", "r")
#pe = data['tasks']['pe profile'][:]
#ke = data['tasks']['ke profile'][:]
#te = data['tasks']['total e profile'][:] 
#buoy = data['tasks']['b profile'][:]
#te_1 = data['tasks']['total e'][:]
#z = data['scales/z/1.0'][:]
t = data['scales']['sim_time'][:]
#x = data['scales/x/1.0'][:]
#bb = data['tasks']['buoyancy'][:]
#uu = data['tasks']['horizontal velocity'][:]
#ww = data['tasks']['vertical velocity'][:]
#mt = data['tasks']['mask test'][:]
tropenerg = data['tasks']['tropo energy'][:]
#te_3d = data['tasks']['total e snap'][:]
dict_vars = {'te':'total e profile','b3d':'buoyancy', 'tropenergy':'tropo energy'}
vars = dp.read_vars(data, dict_vars) 
dims = dp.read_dims(data) 
data.close() 


# for pulse, base decay timescale based on domain size
if PULSE == True: 
    k = 4. 
tau_approx = Lx*np.pi*m**2/(2.*Lz*eps*N1*k) 
tau_exact = tau_approx + eps * (Lx/Lz) * (2. * (m*np.pi)**2 - 3.)/(12. * N1 * k * np.pi) 
tau_off = Lx*(6 + np.pi**2*m**2*(1.+3.*eps**2))/(6.*eps*N1*k*np.pi*Lz)

energ_normed = vars['tropenergy'][:,0,0]/np.max(vars['tropenergy'][:,0,0])
energ_theory = np.exp(-(dims['t'] - pulse_len)/tau_exact)
energ_approx  = np.exp(-(dims['t'] - pulse_len)/tau_approx)
energ_off  = np.exp(-(dims['t'] - pulse_len)/tau_off)
energ_normed_2D = vars['te'][:,0,:].T/np.max(vars['te'][:,0,:].T)

dp.make_1D_plot(sim_name+'/energytest.pdf', dims['t'], simulation = energ_normed, 
        theory = energ_theory, offmode = energ_off)

dp.make_2D_plot(sim_name+'/tetest.pdf', (dims['t'], dims['z']/1000.), energ_normed_2D, 
        title = 'Total Energy', xlabel = 'time (s)', ylabel = 'height (km)')



#
#plt.pcolormesh(t,z, te[:,0,:].T)
#plt.colorbar()
#plt.title('total energy' )
#figpath = sim_name + "/tetest.pdf"
#plt.savefig(figpath) 
#plt.clf() 
#
#plt.pcolormesh(t,z, buoy[:,0,:].T)
#plt.colorbar()
#plt.title('b' )
#figpath = sim_name + "/b_prof.pdf"
#plt.savefig(figpath) 
#plt.clf() 

#plt.pcolormesh(t,z, pe[:,0,:].T)
#plt.colorbar()
#plt.title('potential energy' )
#figpath = sim_name + "/petest.pdf"
#plt.savefig(figpath) 
#plt.clf() 

#plt.pcolormesh(t,z, ke[:,0,:].T)
#plt.colorbar()
#plt.title('kinetic energy' )
#figpath = sim_name + "/ketest.pdf"
#plt.savefig(figpath) 
#plt.clf() 
#plt.pcolormesh(x,z, te_3d[10,:,:].T)
#plt.colorbar()
#plt.title('total e, 200 seconds' )
#plt.xlim(-Lx/2, Lx/2) 
#figpath = sim_name + "/tesnap.pdf"
#plt.savefig(figpath) 
#plt.clf() 
#
#plt.pcolormesh(x,z, ww[-1,:,:].T)
#plt.colorbar()
#plt.title('w, 20,000 seconds' )
#plt.xlim(-Lx/2, Lx/2) 
#figpath = sim_name + "/wtest.pdf"
#plt.savefig(figpath) 
#plt.clf() 
#
#plt.pcolormesh(x,z, bb[-1,:,:].T)
#plt.colorbar()
#plt.xlim(-Lx/2, Lx/2) 
#plt.title('b, 50,000 seconds' )
##figpath = sim_name + "/btest.pdf"
#plt.savefig(figpath) 
#plt.clf() 
#
#plt.pcolormesh(x,z, bb[10,:,:].T)
#plt.colorbar()
#plt.xlim(-Lx/2, Lx/2) 
#plt.title('b, 200 seconds' )
#figpath = sim_name + "/b_pulse.pdf"
#plt.savefig(figpath) 
#plt.clf() 
