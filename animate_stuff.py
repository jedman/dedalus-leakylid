import numpy as np
from matplotlib import animation
import h5py
import argparse
import matplotlib.pyplot as plt
mywriter = animation.ImageMagickFileWriter()
plt.rcParams['image.cmap'] = 'RdBu_r'


parser = argparse.ArgumentParser(description='animate a field from dedalus output')
parser.add_argument('folder', type = str, help='top level folder to look in')
parser.add_argument('sim_name', type = str, help='simulation name in dedalus file structure')
parser.add_argument('-f','--field', type = str, dest='field_to_animate')
parser.add_argument('-o','--outfile',type = str, dest = 'outfile')
args = parser.parse_args()


# get these from args
folder = args.folder
sim_name = args.sim_name
field_to_animate = args.field_to_animate

filepath =  folder + "/" + sim_name + "_s1/" + sim_name + "_s1_p0.h5"
data = h5py.File(filepath, "r")
#te = data['tasks']['total e profile'][:]
#te_1 = data['tasks']['total e'][:]
z = data['scales/z/1.0'][:]
t = data['scales']['sim_time'][:]
x = data['scales/x/1.0'][:]

field_data = data['tasks'][field_to_animate][:]


fig = plt.figure()
im = plt.pcolormesh(x/1000.,z/1000.,field_data[0,:,:].T, shading='gouraud')  # need quadmesh because of stretched grid
#shading = gouraud is a kludge due bug in using shading=flat with 'set_array' method used below

im.set_clim(np.min(field_data[:,:,:]), np.max(field_data[:,:,:]))
# for now just use max range to scale colorbar
plt.xlim(x[0]/1000.,x[-1]/1000.)
plt.xlabel('km')
plt.ylabel('km')
plt.colorbar()


def init():
    im.set_array([])
    return im

def animate(tstep):
    datagrid = field_data[tstep,:,:].T
    im.set_array(datagrid.ravel())
    return im

def frame(tstep):
    datagrid = field_data[tstep,:,:].T
    return datagrid

anim = animation.FuncAnimation(fig,animate, frames= range(0,len(t),4), interval = 10, blit = True)

filename = args.outfile
anim.save(filename , writer= mywriter)
