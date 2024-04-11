import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import pickle
from impacthistory import ImpactMonteCarlo, AnglesVelocities
from scipy import optimize

def mass_to_diameter(mm): # in g
    m = mm/1e3 # to kg
    M1 = m*0.333 # iron fraction
    rho_1 = 7.87e12 # density of iron kg/km^3 
    M2 = m*(1-0.333)
    rho_2 = 3.27e12 # density of forensite kg/km^3
    return 2*((3/4)*(M2/(rho_2*np.pi)) + (3/4)*(M1/(rho_1*np.pi)))**(1/3)

def diameter_to_mass(D):
    
    def objective(x, diameter):
        return np.array([diameter - mass_to_diameter(x[0])])
    
    sol = optimize.root(objective,[1e23],args=(D,))
    if not sol.success:
        raise Exception()
        
    return sol.x[0]/1e3

def create_vars_from_pickle(filename):
    with open(filename,'rb') as f:
        out = pickle.load(f)
    M_low = out['M_low']
    M_high = out['M_high']
    a,b,c,SFD_masses,SFD_frequency,mass_grid,time_grid,v_inf,b_low,b_high = out['p']
    p = ImpactMonteCarlo(a,b,c,SFD_masses,SFD_frequency,mass_grid,time_grid,v_inf,b_low,b_high)
    N = out['N']
    b = out['b']
    M = out['M']
    mass_min,ind_mass_min,inds_start,inds_len,angles,velocities = out['av']
    av = AnglesVelocities()
    av.init2(mass_min,ind_mass_min,inds_start,inds_len,angles,velocities)

    # ocean vaporizing impacts
    energy_vaporization = 5.0e27
    # assume 10% of impactor energy goes into vaporization
    energy_impact_for_vap = energy_vaporization/0.1
    energy_rng = np.array([energy_impact_for_vap, np.inf])
    theta_rng = np.array([-np.inf, np.inf])
    time_last_vaporization, mass_last_vaporization = p.time_of_last_in_energy_interval(av, energy_rng, theta_rng)
    num_vaporization = p.number_of_impacts_in_energy_interval(av, energy_rng, theta_rng)

    return M_low,M_high,p,N,b,M,av,time_last_vaporization,mass_last_vaporization,num_vaporization

def figure2():
    M_low,M_high,p,N,b,M,av,time_last_vaporization,mass_last_vaporization,num_vaporization = create_vars_from_pickle('result.pkl')

    # plot
    MIN_IMPACTOR_D = 100.0
    diameter_grid_avg = mass_to_diameter(p.mass_grid_avg*1e3)
    ind_d = np.argmin(np.abs(diameter_grid_avg - MIN_IMPACTOR_D))

    plt.rcParams.update({'font.size': 15})
    fig,[ax1,ax2,ax3] = plt.subplots(3,1,figsize = [6,5],sharex=True)
    fig.patch.set_facecolor("w")

    alpha=.6
    ms = 2
    lw = .2
    np.random.seed(0)

    # Benner followed by vaporizing impact: 2108
    # Benner followed by no vaporizing impact: 2105
    # No Benner, lots of mid-size impacts: 2107

    np.random.seed(1)
    ind = 3150
    ax = ax1
    ind_t = np.argmin(np.abs(time_last_vaporization[ind] - p.time_grid_avg))
    ind_m = np.argmin(np.abs(mass_last_vaporization[ind] - p.mass_grid_avg))
    for j in range(ind_d,len(p.mass_grid_avg)):
        for i in range(len(p.time_grid_avg)):
            times = np.random.uniform(low=p.time_grid[i+1],high=p.time_grid[i],size=N[ind,i,j])
            for k in range(N[ind,i,j]):
                ax.plot([times[k],times[k]],[0,p.mass_grid_avg[j]], c='k', alpha=alpha, lw=lw)
                ax.plot(times[k],p.mass_grid_avg[j], ls='', marker='o', c='k', ms=ms)
            if j == ind_m and i == ind_t:
                if N[ind,i,j] != 1:
                    raise Exception()
                
                k = 0
                ax.plot([times[k],times[k]],[0,p.mass_grid_avg[j]], c='r', alpha=1, lw=.4)
                ax.plot(times[k],p.mass_grid_avg[j], ls='', marker='o', c='r', alpha=1, ms=4)
    ax.text(.96, .93, '(a)', \
                size = 25,ha='right', va='top',transform=ax.transAxes) 
            
        
    np.random.seed(0)
    ind = 2128
    ax = ax2
    ind_t = np.argmin(np.abs(time_last_vaporization[ind] - p.time_grid_avg))
    ind_m = np.argmin(np.abs(mass_last_vaporization[ind] - p.mass_grid_avg))
    for j in range(ind_d,len(p.mass_grid_avg)):
        for i in range(len(p.time_grid_avg)):
            times = np.random.uniform(low=p.time_grid[i+1],high=p.time_grid[i],size=N[ind,i,j])
            for k in range(N[ind,i,j]):
                ax.plot([times[k],times[k]],[0,p.mass_grid_avg[j]], c='k', alpha=alpha, lw=lw)
                ax.plot(times[k],p.mass_grid_avg[j], ls='', marker='o', c='k', ms=ms)
            if j == ind_m and i == ind_t:
                if N[ind,i,j] != 1:
                    raise Exception()
                k = 0
                ax.plot([times[k],times[k]],[0,p.mass_grid_avg[j]], c='r', alpha=1, lw=.4)
                ax.plot(times[k],p.mass_grid_avg[j], ls='', marker='o', c='r', alpha=1, ms=4)
    ax.text(.96, .93, '(b)', \
                size = 25,ha='right', va='top',transform=ax.transAxes) 
            
        
    np.random.seed(0)
    ind = 2123
    ax = ax3
    ind_t = np.argmin(np.abs(time_last_vaporization[ind] - p.time_grid_avg))
    ind_m = np.argmin(np.abs(mass_last_vaporization[ind] - p.mass_grid_avg))
    for j in range(ind_d,len(p.mass_grid_avg)):
        for i in range(len(p.time_grid_avg)):
            times = np.random.uniform(low=p.time_grid[i+1],high=p.time_grid[i],size=N[ind,i,j])
            for k in range(N[ind,i,j]):
                ax.plot([times[k],times[k]],[0,p.mass_grid_avg[j]], c='k', alpha=alpha, lw=lw)
                ax.plot(times[k],p.mass_grid_avg[j], ls='', marker='o', c='k', ms=ms)
            if j == ind_m and i == ind_t:
                if N[ind,i,j] != 1:
                    raise Exception()
                k = 0
                ax.plot([times[k],times[k]],[0,p.mass_grid_avg[j]], c='r', alpha=1, lw=.4)
                ax.plot(times[k],p.mass_grid_avg[j], ls='', marker='o', c='r', alpha=1, ms=4)
    ax.set_xlabel('Time (Ga)')
    ax.text(.96, .93, '(c)', \
                size = 25,ha='right', va='top',transform=ax.transAxes)  
        
    for ax in [ax1,ax2,ax3]:
        ax.set_yscale('log')
        ax.invert_xaxis()
        ax.set_xlim(4.5,4.1)
        ax.set_ylim(1.5e19,9e22)
        ax.set_xticks(np.arange(4.5,4.0,-.1))
        ax.grid(alpha=.2, ls='--')

    ax2.set_ylabel('Impactor mass (kg)')
    ax = ax1
    x,y = 4.453, 3.5e22
    xx,yy = 4.42, 2e22
    note = 'Moneta'
    ax.annotate('',(x,y), (xx,yy), fontsize=12, arrowprops= {'width':.7,'headwidth':7,'headlength':6,'edgecolor':'k','fc':'k','alpha':0.6}, color='k')
    ax.text(xx-.002,yy-50,note,ha='left', va='center', c='k', fontsize=12, alpha=.6)

    x,y = 4.343, 2.1e20
    xx,yy = x-.03,2e21
    note = 'Last\nocean vap.'
    ax.annotate('',(x,y), (xx,yy), fontsize=12, arrowprops= {'width':.7,'headwidth':7,'headlength':6,'edgecolor':'r','fc':'r','alpha':0.6}, color='k')
    ax.text(xx-.003,yy+50,note,ha='left', va='center', c='r', fontsize=12, alpha=.6)

    ax = ax2
    x,y = 4.395, 2e22
    xx,yy = 4.355, 1e22
    note = 'Moneta\nand\nlast ocean vap.'
    ax.annotate('',(x,y), (xx,yy), fontsize=12, arrowprops= {'width':.7,'headwidth':7,'headlength':6,'edgecolor':'k','fc':'r','alpha':0.6}, color='k')
    ax.text(xx-.03,yy+0,note,ha='center', va='top', c='k', fontsize=12, alpha=.6)

    ax = ax3
    x,y = 4.245, 5.0e20
    xx,yy = 4.22, 5.0e20
    note = 'Last\nocean vap.'
    ax.annotate('',(x,y), (xx,yy), fontsize=12, arrowprops= {'width':.7,'headwidth':7,'headlength':6,'edgecolor':'r','fc':'r','alpha':0.6}, color='k')
    ax.text(xx-.005,yy+0,note,ha='left', va='center', c='r', fontsize=12, alpha=.6)

    diameters = [500,1000,2000]
    masses = [diameter_to_mass(d) for d in diameters]
    ax = ax1
    for i,mass in enumerate(masses):
        ax.text(4.2,mass+mass*0.2,'%i km'%diameters[i],ha='center', va='bottom', c='k', fontsize=10, alpha=.6)
        ax.plot([4.175,4.225],[mass,mass],lw=1,ls='-',c='k', alpha=0.6)

    plt.subplots_adjust(hspace=.06)        
    plt.savefig('figures/figure2.pdf',bbox_inches='tight')

def life_impactor_stats(p, av, mass, time_last_vaporization, num_vaporization):


    mass_rng = np.array([mass, np.inf])
    theta_rng = np.array([-np.inf, np.inf])
    v_rng = np.array([-np.inf, np.inf])

    time_last_life, mass_last_life = p.time_of_last_in_interval(av, mass_rng, theta_rng, v_rng)
    num_life = p.number_of_impacts_in_interval(av, mass_rng, theta_rng, v_rng)

    inds_happens = np.where(num_life > 0)
    p_happens = (inds_happens[0].shape[0]/num_life.shape[0])

    inds_life = np.where(time_last_life[inds_happens] <= time_last_vaporization[inds_happens])
    if inds_happens[0].shape[0] == 0:
        inds_no_vap = np.where(num_vaporization == 0)
        p_not_vap = inds_no_vap[0].shape[0]/num_life.shape[0]
    else:
        p_not_vap = inds_life[0].shape[0]/inds_happens[0].shape[0]

    p_life = p_happens*p_not_vap
    
    time_last_life_1 = time_last_life[inds_happens][inds_life]
    mass_last_life_1 = mass_last_life[inds_happens][inds_life]
    
    if inds_happens[0].shape[0] == 0:
        CI = np.array([4.5,4.5,4.5,4.5,4.5])
        CIm = np.array([6.0e23,6.0e23,6.0e23,6.0e23,6.0e23])
    else:
#         print(inds_happens[0])
        CI = np.quantile(time_last_life_1, [.025,.16,.5,.84,.975])
        CIm = np.quantile(mass_last_life_1, [.025,.16,.5,.84,.975])
        
    if inds_happens[0].shape[0] == 0:
        CI_h = np.array([4.5,4.5,4.5,4.5,4.5])
    else:
        CI_h = np.quantile(time_last_life[inds_happens], [.025,.16,.5,.84,.975])

    return p_happens, p_not_vap, p_life, CI, CIm, CI_h, time_last_life_1, mass_last_life_1

def life_impactor_probabilities(p, av, masses, time_last_vaporization,  num_vaporization):

    keys = ['p_happens','p_not_vap','p_life']
    sol = {}
    for key in keys:
        sol[key] = np.empty(len(masses))
    sol['CI'] = np.empty((len(masses),5))
    sol['CIm'] = np.empty((len(masses),5))
    sol['CI_h'] = np.empty((len(masses),5))
    sol['mass_of_life'] = []

    for i in range(len(masses)):
        p_happens, p_not_vap, p_life, CI, CIm, CI_h, time_of_life, mass_of_life = life_impactor_stats(p, av, masses[i], time_last_vaporization,  num_vaporization)
        sol['p_happens'][i] = p_happens
        sol['p_not_vap'][i] = p_not_vap
        sol['p_life'][i] = p_life
        sol['CI'][i,:] = CI
        sol['CIm'][i,:] = CIm
        sol['CI_h'][i,:] = CI_h
        sol['mass_of_life'].append(mass_of_life)
        print(i,end='\r')
        
    return sol

def figure3():
    M_low,M_high,p,N,b,M,av,time_last_vaporization,mass_last_vaporization,num_vaporization = create_vars_from_pickle('result.pkl')

    np.random.seed(0)
    masses = np.logspace(np.log10(1.1e19),np.log10(6e22),52)
    sol = life_impactor_probabilities(p, av, masses, time_last_vaporization,  num_vaporization)

    plt.rcParams.update({'font.size': 15})
    fig,ax = plt.subplots(1,1,figsize = [5,4])

    ax.plot(masses, sol['p_happens'], ls='-', lw=2, label='Probability impact\noccurs $\geq 1$ time')
    ax.plot(masses, sol['p_life'], ls='--', lw=2,label='Probability of\na "life-starting impact"')

    note = 'Probability\nimpact\noccurs\n$\geq 1$ time'
    ax.text(0.85e22,.86,note, va='center',c='C0',fontsize=12)

    note = 'Probability\nof impact\nwith no later\nocean vap.'
    ax.text(4.5e20,.26,note, va='bottom',c='C1',fontsize=12)

    ax.set_xscale('log')
    ax.set_ylim(0,1)
    ax.set_xlim(1e20,7e22)
    ax.grid(alpha=.3)
    # ax.legend(ncol=1,bbox_to_anchor=(1, 1.02), loc='lower right',fontsize=12)
    ax.set_xlabel('Minimum mass of impact (kg)\nthat produces prebiotic molecules')
    ax.set_ylabel('Probability')
    ax.axvline(4e20,c='k',ls=':',alpha=.5)
    note = 'Wogan+2023\noptimistic'
    ax.text(2.3e20, 0.02, note, \
            size = 10, rotation=90,ha='left', va='bottom',alpha=.5)
    ax.axvline(5e21,c='k',ls=':',alpha=.5)
    note = 'Wogan+2023\npessimistic'
    ax.text(2.8e21, .02, note, \
            size = 10, rotation=90,ha='left', va='bottom',alpha=.5)

    # Mass implied by HSEs
    ax.fill_between([M_low,M_high],[1,1],alpha=.07,fc='k')
    note = '''Moneta-sized\nimpact'''
    ax.text(np.mean([M_low,M_high])*.9, .3, note, alpha=.7, \
            size = 13,ha='center', va='bottom', c='k',rotation=90)


    plt.subplots_adjust(wspace=.4)
    plt.savefig('figures/figure3.pdf',bbox_inches='tight')

def life_impactor_time_mass(p, av, min_mass, time_last_vaporization, M_low):
    mass_rng = np.array([min_mass, np.inf])
    theta_rng = np.array([-np.inf, np.inf])
    v_rng = np.array([-np.inf, np.inf])

    time_last_life, mass_last_life = p.time_of_last_in_interval(av, mass_rng, theta_rng, v_rng)
    num_life = p.number_of_impacts_in_interval(av, mass_rng, theta_rng, v_rng)

    inds_happens = np.where(num_life > 0)
    p_happens = (inds_happens[0].shape[0]/num_life.shape[0])
    # print('Probability that one happens = %.4f'%p_happens)

    inds_life = np.where(time_last_life[inds_happens] <= time_last_vaporization[inds_happens])
    p_not_vap = inds_life[0].shape[0]/inds_happens[0].shape[0]
    # print('Probability that there is no ocean vaporization after the last one = %.4f'%p_not_vap)

    p_life = p_happens*p_not_vap
    # print('Probability of origin of life when rewinding the tape = %.4f'%p_life)

    time_last_life_1 = time_last_life[inds_happens][inds_life]
    mass_last_life_1 = mass_last_life[inds_happens][inds_life]

    p_HSE_impact = np.where(mass_last_life_1 > M_low)[0].shape[0]/mass_last_life_1.shape[0]
    # print('Probability that a "HSE" impactor is the last one that meets criteria: ',p_HSE_impact)

    res = {}
    res['p_happens'] = p_happens
    res['p_not_vap'] = p_not_vap
    res['p_life'] = p_life
    res['p_HSE'] = p_HSE_impact
    res['time_life'] = time_last_life_1
    res['mass_life'] = mass_last_life_1
    
    res['time_happens'] = time_last_life[inds_happens]
    res['mass_happens'] = mass_last_life[inds_happens]

    return res


def figure4():
    M_low,M_high,p,N,b,M,av,time_last_vaporization,mass_last_vaporization,num_vaporization = create_vars_from_pickle('result.pkl')
    
    min_mass = 4e20
    optimistic = life_impactor_time_mass(p, av, min_mass, time_last_vaporization, M_low)

    min_mass = 5e21
    pessimistic = life_impactor_time_mass(p, av, min_mass, time_last_vaporization, M_low)


    plt.rcParams.update({'font.size': 15})
    fig,[[ax1,ax3],[ax2,ax4]] = plt.subplots(2,2,figsize = [11,7])

    c = 'C0'
    hist, bin_edges = np.histogram(optimistic['time_life'], bins=p.time_grid[::-1])
    ax1.plot(bin_edges[1:], hist,ds='steps-pre', lw=3, c=c)
    ax1.hist(optimistic['time_life'], bins=p.time_grid[::-1], rwidth = 1, alpha=.1, density=False, facecolor=c)

    # ax1.hist(optimistic['time_happens'], bins=p.time_grid[::-1], rwidth = 1, alpha=.1, density=False, facecolor='r')

    ax1.invert_xaxis()
    ax1.set_xlim(4.5,3.5)
    ax1.set_ylim(0,ax1.get_ylim()[1])
    ax1.grid(alpha=.4)
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Time of impact (Ga)')
    ax1.axvline(np.quantile(optimistic['time_life'],[0.025])[0],c='k',ls=':',alpha=.6)
    ax1.axvline(np.quantile(optimistic['time_life'],[1-0.025])[0],c='k',ls=':',alpha=.6)
    ax1.axvline(np.quantile(optimistic['time_life'],[0.5])[0],c='k',ls='--',alpha=.6)

    title = r'$\bf{Wogan}$$\bf{+}$$\bf{(2023)}$ $\bf{optimistic}$ $\bf{case}$'
    title+= '\n'
    title+= r'Last $> 4 \times 10^{20}$ kg impact'
    title+= '\nwith no later ocean vap.'
    ax1.text(.5,1.37, title, \
            size = 14,ha='center', va='top',transform=ax1.transAxes,backgroundcolor='none')
    ax1.text(0.97,.96, '(i)', \
            size = 20,ha='right', va='top',transform=ax1.transAxes,backgroundcolor='.95')


    hist, bin_edges = np.histogram(optimistic['mass_life'], bins=p.mass_grid)
    ax2.plot(bin_edges[1:], hist,ds='steps-pre', lw=3, c=c)
    ax2.hist(optimistic['mass_life'], bins=p.mass_grid, alpha=.1, density=False, facecolor=c)

    ax2.set_xlim(1e20, 1e23)
    ax2.set_ylim(0,ax2.get_ylim()[1])
    ax2.fill_between([M_low,M_high],[1000,1000],zorder=0,alpha=.1,fc='k')
    # ax2.hist(np.ones(1000)*((M_low+M_high)/2),bins=[M_low,M_high], fc='k', zorder=0, alpha=0.1)
    note = '''Moneta\nimpact'''
    ax2.text(np.mean([M_low,M_high])*.9, 230, note, \
            size = 11,ha='center', va='bottom', c='k',rotation=90)
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Mass of impact (kg)')
    ax2.grid(alpha=.4)
    ax2.set_xscale("log")
    # ax2.axvline(np.quantile(optimistic['mass_life'],[0.025])[0],c='k',ls=':',alpha=.6)
    # ax2.axvline(np.quantile(optimistic['mass_life'],[1-0.025])[0],c='k',ls=':',alpha=.6)
    # ax2.axvline(np.quantile(optimistic['mass_life'],[0.5])[0],c='k',ls='--',alpha=.6)
    ax2.text(0.03,.05, '(ii)', \
            size = 20,ha='left', va='bottom',transform=ax2.transAxes,backgroundcolor='.95')


    hist, bin_edges = np.histogram(pessimistic['time_life'], bins=p.time_grid[::-1])
    ax3.plot(bin_edges[1:], hist,ds='steps-pre', lw=3, c=c)
    ax3.hist(pessimistic['time_life'], bins=p.time_grid[::-1], rwidth = 1, alpha=.1, density=False, facecolor=c)

    # ax3.hist(pessimistic['time_happens'], bins=p.time_grid[::-1], rwidth = 1, alpha=.1, density=False, facecolor='r')

    ax3.invert_xaxis()
    ax3.set_xlim(4.5,3.5)
    ax3.set_ylim(0,ax3.get_ylim()[1])
    ax3.grid(alpha=.4)
    ax3.set_ylabel('Count')
    ax3.set_xlabel('Time of impact (Ga)')
    ax3.axvline(np.quantile(pessimistic['time_life'],[0.025])[0],c='k',ls=':',alpha=.6)
    ax3.axvline(np.quantile(pessimistic['time_life'],[1-0.025])[0],c='k',ls=':',alpha=.6)
    ax3.axvline(np.quantile(pessimistic['time_life'],[0.5])[0],c='k',ls='--',alpha=.6)

    title = r'$\bf{Wogan}$$\bf{+}$$\bf{(2023)}$ $\bf{pessimistic}$ $\bf{case}$'
    title+= '\n'
    title+= r'Last $> 5 \times 10^{21}$ kg impact'
    title+= '\nwith no later ocean vap.'
    ax3.text(.5,1.37, title, \
            size = 14,ha='center', va='top',transform=ax3.transAxes,backgroundcolor='none')
    ax3.text(0.97,.96, '(i)', \
            size = 20,ha='right', va='top',transform=ax3.transAxes,backgroundcolor='.95')

    hist, bin_edges = np.histogram(pessimistic['mass_life'], bins=p.mass_grid)
    ax4.plot(bin_edges[1:], hist,ds='steps-pre', lw=3, c=c)
    ax4.hist(pessimistic['mass_life'], bins=p.mass_grid, alpha=.1, density=False, facecolor=c)

    ax4.set_xlim(1e20, 1e23)
    ax4.set_ylim(0,ax4.get_ylim()[1])
    ax4.fill_between([M_low,M_high],[1000,1000],zorder=0,alpha=.1,fc='k')
    # ax4.hist(np.ones(1000)*((M_low+M_high)/2),bins=[M_low,M_high], fc='k', zorder=0, alpha=0.1)

    note = '''Moneta\nimpact'''
    ax4.text(np.mean([M_low,M_high])*.9, 100, note, \
            size = 11,ha='center', va='bottom', c='k',rotation=90)
    ax4.set_ylabel('Count')
    ax4.set_xlabel('Mass of impact (kg)')
    ax4.grid(alpha=.4)
    ax4.set_xscale("log")
    # ax4.axvline(np.quantile(pessimistic['mass_life'],[0.025])[0],c='k',ls=':',alpha=.6)
    # ax4.axvline(np.quantile(pessimistic['mass_life'],[1-0.025])[0],c='k',ls=':',alpha=.6)
    # ax4.axvline(np.quantile(pessimistic['mass_life'],[0.5])[0],c='k',ls='--',alpha=.6)
    ax4.text(0.03,.05, '(ii)', \
            size = 20,ha='left', va='bottom',transform=ax4.transAxes,backgroundcolor='.95')



    rec = matplotlib.patches.Rectangle((-.27,-.35), 1.35, 3.1, fill=False, lw=2, clip_on=False,transform=ax2.transAxes)
    ax2.add_patch(rec)
    ax1.text(-0.23,1.2, '(a)', \
            size = 30,ha='left', va='bottom',transform=ax1.transAxes,backgroundcolor='none')

    rec = matplotlib.patches.Rectangle((-.27,-.35), 1.35, 3.1, fill=False, lw=2, clip_on=False,transform=ax4.transAxes)
    ax4.add_patch(rec)
    ax3.text(-0.23,1.2, '(b)', 
            size = 30,ha='left', va='bottom',transform=ax3.transAxes,backgroundcolor='none')

    plt.subplots_adjust(wspace=.4,hspace=.3)

    plt.savefig('figures/figure4.pdf',bbox_inches='tight')

def main():
    figure2()
    figure3()
    figure4()

if __name__ == '__main__':
    main()



