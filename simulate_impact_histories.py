import numpy as np
import pickle
import numba as nb

from impacthistory import ImpactMonteCarlo, AnglesVelocities
from impacthistory import utils

def make_sampler():
    
    # Fit to red line Fig. 5 in Morbidelli et al. (2018)
    a = 0.0010721229624386592 
    b = 1.7882966033279404 
    c = -2.764929021041997

    # Main belt SFD from Marchi et al. (2014) scaled so that 1 == an object
    # that makes a 1 km diameter crater on the moon.
    SFD_masses, SFD_frequency = utils.marchi_SFD_scaled_to_1km_lunar()
    
    mass_grid = np.logspace(12,23,200)
    time_grid = np.arange(4.5, 3-1e-8, -0.025)
    
    # Velocity of impacts far from Earth for computing
    # gravitational scaling between Earth and Moon.
    # Following Marchi (2021) ApJ
    v_inf = 13.0
    
    # Perfect extrapolation of SFD
    b_low = 0.41469782881401496
    b_high = 0.41469782881401496

    p = ImpactMonteCarlo(a, b, c, SFD_masses, SFD_frequency,
                         mass_grid, time_grid, v_inf, b_low, b_high)
    
    return p

def main():
    @nb.njit()
    def set_seed(seed):
        np.random.seed(seed)
    set_seed(0)
    np.random.seed(0)

    niters = 5000

    # Mass implied by HSEs
    M_low = 2.0e22
    M_high = 6.0e22

    # Do the simulation
    p = make_sampler()
    N, b, M = p.impact_history_mass_constraint(niters, M_low, M_high)

    # Assign angles and velocties
    mass_min = 0.1e20
    av = p.assign_angles_and_velocities(N, mass_min)

    # Save results
    out = {}
    out['M_low'] = M_low
    out['M_high'] = M_high
    out['p'] = (p.a,p.b,p.c,p.SFD_masses,p.SFD_frequency,p.mass_grid,p.time_grid,p.v_inf,p.b_low,p.b_high)
    out['N'] = N
    out['b'] = b
    out['M'] = M
    out['av'] = (av.mass_min,av.ind_mass_min,av.inds_start,av.inds_len,av.angles,av.velocities)    
    with open('result.pkl','wb') as f:
        pickle.dump(out,f)

if __name__ == '__main__':
    main()
    

    
