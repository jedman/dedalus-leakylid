import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt


def taufit(energ_normed, dims, m, k, eps, Lx, Lz, archive):
    '''fit a tau to simulated energy (energ_normed) and add to archive list''' 
    N1 = 0.01 
    tau_approx = Lx*np.pi*m**2/(2.*Lz*eps*N1*k)
    tau_exact = tau_approx + eps * (Lx/Lz) * (2. * (m*np.pi)**2 - 3.)/(12. * N1 * k * np.pi)
    [slope, intercept, r_value, p_value, std_err] = stats.linregress(dims['t'], np.log(energ_normed))
    plt.plot(dims['t'], np.log(energ_normed)) 
    plt.plot(dims['t'], slope*dims['t'], label = 'regress') 
    plt.legend() 
    tmp = dict(m = m, k = k, eps = eps, tau = -1./slope, intercept = intercept, r = r_value, std_err = std_err, tau_theory = tau_exact, tau_approx = tau_approx)
    archive.append(tmp) 
    return 

def plot_taus(archive): 
    for a in archive:
        plt.scatter(a['k'], a['tau']/a['tau_theory'])
    plt.savefig('tau_scatter')
    return



