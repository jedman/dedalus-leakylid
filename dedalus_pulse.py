import numpy as np
import h5py
from dedalus import public as de
from dedalus.extras import flow_tools
import time
import argparse
import dedalus_plots as dp
import matplotlib.pyplot as plt
from subprocess import call

parser = argparse.ArgumentParser(description='simulate a Boussinesq pulse')
parser.add_argument('k', metavar = 'k', type = int, help='forcing wavenumber in the horizontal')
parser.add_argument('m', metavar = 'm', type = int, help='forcing wavenumber in the vertical')
parser.add_argument('eps', metavar = 'eps', type = float, help='epsilon, the ratio of buoyancy frequency in troposphere and stratosphere')
parser.add_argument('sim_name',metavar = 'sim_name', type = str, help = 'simulation name')
parser.add_argument('-nh','--non-hstat', dest='hstat', action='store_false')
parser.add_argument('-p','--pulse', dest='pulse', action='store_true')
parser.add_argument('-pl', '--pulse-len', dest = 'pulse_len' , type = float)
parser.add_argument('-rl', '--rigid-lid', dest='rigid_lid', action = 'store_true')
parser.add_argument('-tau', '--damping-tau', dest = 'tau', type = int, help='rayleigh damping timescale in days')
parser.set_defaults(pulse_len=1000)
parser.set_defaults(hstat=True)
parser.set_defaults(pulse=False)
parser.set_defaults(rigid_lid=False)
args = parser.parse_args()

PULSE = args.pulse
HYDROSTATIC = args.hstat
#print('pulse_len is ', args.pulse_len)

if HYDROSTATIC == True:
     print('using hydrostatic boussinesq solver')
else:
     print('using non-hydrostatic boussinesq solver')

if PULSE == True:
     print('solving for gaussian forcing')
else:
     print('solving initial gaussian buoyancy perturbation')


import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

logger = logging.getLogger(__name__)




def onetwoone(field, niter = 100):
    '''1-2-1 filter'''
    newfield = field
    j=0
    while j < niter:
        lastfield = newfield
        for i in range(1,len(field)-1):
            newfield[i] = 0.5*lastfield[i] + 0.25*(lastfield[i+1] + lastfield[i-1])
        j += 1
    return newfield

stop_time = 86400.*2. # simulation stop time  (seconds)
pulse_len = args.pulse_len # seconds of forcing
N1 = 0.01 # buoyancy frequency in the troposphere (1/s)

Lx, Lz = (stop_time*100, 15000) # domain size in meters
nx, nz = (196*2, 124)  # number of points in each direction

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(-Lx/2., Lx/2.))
# compound z basis -- better to resolve jump condition?
#zb1 = de.Chebyshev('z1',int(nz/4), interval=(0, Lz+1000), dealias=3/2)
#zb2 = de.Chebyshev('z2', nz, interval=(Lz+1000,model_top), dealias = 3/2)
#z_basis = de.Compound('z',(zb1,zb2), dealias = 3/2)
#
m = args.m # vertical mode number
k = args.k # horizontal mode number
eps = args.eps # ratio of N1/N2
N2 = N1/eps  # buoyancy frequency in the stratosphere

if (args.rigid_lid):
    model_top = Lz
    nz = int(nz/4)
else:
    model_top = 8. * Lz # lid height

z_basis = de.Chebyshev('z', nz, interval= (0, model_top))
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
if not (args.rigid_lid):
    strat = np.where( z > Lz)
    ncc['g'][:,strat] = N2**2

ncc['g'][:,0] = N1**2
ncc.meta['x']['constant'] = True
problem.parameters['Nsq'] = ncc
print(ncc['g'][20,:])


# non-constant coefficient alpha (rayleigh drag)
tmp = np.zeros(z.shape[1])

if (args.tau):
    tmp[:] = 1./(args.tau*86400.) # set the rayleigh damping timescale

strat2 = np.where( z[:] > 5.*Lz)
tmp[strat2[1]] = 24./86400.
#tmp[strat2[0]]= 0. ### try inviscid case
tmp = onetwoone(tmp, niter=30)
tmpgrid, _ = np.meshgrid(tmp,x)

aa = domain.new_field(name='alpha')
aa['g'] = tmpgrid
aa.meta['x']['constant'] = True
problem.parameters['alpha'] = aa



# mask (for analysis)
strat = np.where(z>Lz)
mask = domain.new_field(name = 'mask')
mask['g'] = 1
mask['g'][:,strat] = 0
mask.meta['x']['constant'] = True
problem.parameters['mask'] = mask



if PULSE == True:
    sigma_t = args.pulse_len
    sigma_x = args.k
    def forcing(solver):
        # if using dealiasing, it's important to apply the forcing on the dealiased doman (xd,zd)
        if solver.sim_time < stop_time:
            td = solver.sim_time
            f = 0.00001*np.sin(m *np.pi*zd/Lz) * np.exp(-(xd*xd)/(sigma_x**2)) * np.exp(-(td - 2.*sigma_t)**2/sigma_t**2)
            strat = np.where(zd>Lz)
            if not (args.rigid_lid):
                f[:,strat] = 0.
           # subtract the horizontal mean at each level so there's no k=0
           # fprof = np.mean(f, axis = 0 )
           # ftmp = np.repeat(fprof, xd.shape[0])
          #  fmask = ftmp.reshape(zd.shape[1],xd.shape[0])
          #   f = f - fmask.T
        else:
            f = 0.
        return f
else:
    def forcing(solver):
        # if using dealiasing, it's important to apply the forcing on the dealiased doman (xd,zd)
        f = 0.
        return f


#define general forcing function
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
problem.add_equation("dt(u) + 1/rho*dx(p) + alpha*u = 0")
problem.add_equation("dt(B) + Nsq*w + alpha*B = forcing_func")
#problem.add_equation("dt(B) + Nsq*w  = 0")
problem.add_equation("dx(u) + dz(w) = 0")

if HYDROSTATIC == True:
	problem.add_equation("B - 1/rho*dz(p)  = 0")
else:
	problem.add_equation("B - 1/rho*dz(p) - dt(w) = 0")

# fourier direction has periodic bc, chebyshev has a lid
problem.add_bc("left(w) = 0") # refers to the first end point in chebyshev direction
problem.add_bc("right(w) = 0", condition="(nx != 0)") # rigid lid, condition note for k = 0 mode
problem.add_bc("integ(p,'z') = 0", condition="(nx == 0)") # pressure gauge condition for k = 0

# build solver

ts = de.timesteppers.RK443 # arbitrary choice of time stepper
#ts = de.timesteppers.CNAB2
solver =  problem.build_solver(ts)


sim_name = args.sim_name
print('simulation name is', sim_name)
print('effective forcing horizontal wavelength is' , Lx/k/1000., 'kilometers')
print('effective forcing vertical wavelength is' , 2.*Lz/m/1000., 'kilometers')
print('stratification ratio N1/N2 is' , N1/N2 )
# initial conditions

# tell the forcing function what its arg is (clunky)
forcing_func.args = [solver]
forcing_func.original_args = [solver]

# initial conditions
x, z = domain.grids(scales=1)
u = solver.state['u']
w = solver.state['w']
p = solver.state['p']
B = solver.state['B'] # zero for everything
u['g'] = 0.
w['g'] = 0.
p['g'] = 0.
B['g'] = 0.

if not (args.pulse):
    # start with an initial buoyancy perturbation in the tropopshere
    sigma_x = args.k
    B['g'] = 0.1*np.sin(m *np.pi*z/Lz) * np.exp(-(x*x)/(sigma_x**2))
    strat = np.where(zd>Lz)
    if not (args.rigid_lid):
        f[:,strat] = 0.

solver.stop_sim_time = stop_time
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# CFL conditions
#initial_dt = 0.8*Lz/nz
initial_dt = 100
cfl = flow_tools.CFL(solver,initial_dt,safety=0.8, max_change=30., min_change=0.5, max_dt=900)
# too large of a timestep makes things rather diffusive
cfl.add_velocities(('u','w'))

# fields to record
analysis = solver.evaluator.add_file_handler(sim_name, sim_dt=900, max_writes=50000)
analysis.add_task('B', name = 'buoyancy' )
analysis.add_task('u', name = 'horizontal velocity' )
analysis.add_task('w', name = 'vertical velocity' )
analysis.add_task('p', name = 'pressure' )
# 1d fields
analysis.add_task('mask')
analysis.add_task("integ(B, 'z')", name = 'tropo b') # use mask to integrate over troposphere only
analysis.add_task("integ(0.5 * mask *(u*u + w*w +  B*B/Nsq ), 'z')", name = 'tropo energy') # use mask to integrate over troposphere only
#analysis.add_task("integ(0.5 * (u*u + w*w +  B*B/Nsq ))", name='total e')
try:
    logger.info('Starting loops')
    start_time = time.time()
    while solver.ok:
        dt = cfl.compute_dt()
        solver.step(dt)
        if solver.iteration % 4 == 0:
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

# archive decay timescales
# merge parallel files
#call(['./merge.py', sim_name])
#filepath = sim_name + "/" +  sim_name + "_s1.h5"

#filepath = sim_name + "/" + sim_name + "_s1/" + sim_name + "_s1_p0.h5"
#print(filepath)
# open data file
#data = h5py.File(filepath, "r")
# read in variables and dimensions
#dict_vars = {'tropenergy':'tropo energy', 'b3d':'buoyancy', 'u3d':'horizontal velocity', 'w3d':'vertical velocity', 'p':'pressure'}
#vars = dp.read_vars(data, dict_vars)
#dims = dp.read_dims(data)
#data.close()

#energ_normed = vars['tropenergy'][:,0,0]


#dp.make_1D_plot(sim_name+'/energytest.pdf', dims['t'], simulation = energ_normed)

#dp.make_2D_plot(sim_name+'/binit.pdf', (dims['x']/1000., dims['z']/1000.),vars['b3d'][10,:,:].T , title='b initial', xlabel = 'x (km)', ylabel = 'z (km)')
#plt.clf()
#dp.make_2D_plot(sim_name+'/bfinal.pdf', (dims['x']/1000., dims['z']/1000.),vars['b3d'][-1,:,:].T , title='b final', xlabel = 'x (km)', ylabel = 'z (km)')
#plt.clf()
#dp.make_2D_plot(sim_name+'/ufinal.pdf', (dims['x']/1000., dims['z']/1000.),vars['u3d'][-1,:,:].T , title='u final', xlabel = 'x (km)', ylabel = 'z (km)')
#plt.clf()
#dp.make_2D_plot(sim_name+'/umid.pdf', (dims['x']/1000., dims['z']/1000.),vars['u3d'][50,:,:].T , title='u mid', xlabel = 'x (km)', ylabel = 'z (km)')
#plt.clf()
#dp.make_2D_plot(sim_name+'/wfinal.pdf', (dims['x']/1000., dims['z']/1000.),vars['w3d'][-1,:,:].T , title='w final', xlabel = 'x (km)', ylabel = 'z (km)')
#plt.clf()

#taufit.plot_taus(archive_list)

#import pickle
#outfile = open( "eps02_mpi.p", "wb" )
#pickle.dump(archive_list, outfile)
    #dp.make_1D_plot(sim_name+'/energytest.pdf', dims['t'], simulation = energ_normed,
    #    theory = energ_theory, offmode = energ_off)

#dp.make_2D_plot(sim_name+'/bend.pdf', (dims['x']/1000., dims['z']/1000.),vars['b3d'][-1,:,:].T , title='b final', xlabel = 'x (km)', ylabel = 'z (km)')i
