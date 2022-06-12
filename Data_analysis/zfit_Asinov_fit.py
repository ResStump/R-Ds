# %%
import numpy as np
import uproot
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.ROOT)
plt.rcParams['figure.dpi'] = 40
import hist
import zfit
from scipy.stats import rv_histogram


######################
# Plotting functions #
######################

def plot_hist_model_and_data(model, data_zfit):
    """Plots a histogram of the model with the current parameters and of 
    data_zfit"""
    mplhep.histplot(model.to_hist(), label='model', yerr=False)
    mplhep.histplot(data_zfit.to_hist(), label='data', yerr=False)
    plt.ylabel('Number of events')
    plt.legend()
    return

def plot_stacked_hist(obs):
    """Plots a stacked histogram of the different signals for the observable
    obs. The number of events per signal are the current parameters"""
    bins = hists[obs]['-1'].to_numpy()[1]
    counts = [param.value()/h.sum()*h.counts()
              for param, h in zip(params.values(), hists[obs].values())]
    mplhep.histplot(counts, bins, stack=True, label=labels.values(),
                    histtype='fill')
    plt.minorticks_on()
    plt.xlabel(labels_obs[obs])
    plt.ylabel('Number of events')
    plt.legend()
    return


# Likelihood scans
##################

def plot_profile(param, range, NLL, name=None, num=50):
    """Plots the profile of the parameter param"""
    param_result = param.value()
    NLL0 = NLL.value()
    x = np.linspace(float(range[0]), float(range[1]), num=num)
    y = np.empty(x.shape, dtype=object)
    
    for i, val in enumerate(x):
        param.set_value(val)
        y[i] = NLL.value()
    # set parameter values back to optimized values
    param.set_value(param_result)
    
    plt.plot(x, 2*(y - NLL0))
    if name:
        plt.xlabel(name)
    else:
        plt.xlabel(param.name)
    plt.ylabel('$2*\Delta\,$NLL')
    return

def plot_contour(param1, param2, range1, range2, NLL, xlabel=None, ylabel=None,
                 num=50):
    """Plots the contour of the parameters param1, param2"""
    param1_result = param1.value()
    param2_result = param2.value()
    NLL0 = NLL.value()
    X, Y = np.meshgrid(
        np.linspace(float(range1[0]), float(range1[1]), num=num),
        np.linspace(float(range2[0]), float(range2[1]), num=num)
    )
    Z = np.empty(X.shape)
    
    j = 0 # j needs to be assigned beforehand such that the for loops work
    for i in range(num):
        param2.set_value(Y[i, j]) # this function is expensive
        for j in range(num):      # -> taken out of this loop
            param1.set_value(X[i, j])
            Z[i, j] = NLL.value()
    # set parameter values back to optimized values
    param1.set_value(param1_result)
    param2.set_value(param2_result)
    
    plt.contour(X, Y, 2*(Z - NLL0), levels=[1], colors=['#1f77b4'])
    plt.plot(param1_result, param2_result, '.k', label='fit result')
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(param1.name)
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(param2.name)
    plt.title('$2*\Delta\,$NLL$=1.0$ contour')
    return


###################
# Other functions #
###################

def print_params(rounded=True):
    print('Resulting yields:')
    for k, p in params.items():
        value = p.value().numpy()
        if rounded:
            value = round(p.value().numpy())
        print(f'Signal {k}:', value)



# %%#####################################
# Data import, filtering and processing #
#########################################

# constants
m_Bs = 5.36688 # [GeV]
m_Ds = 1.96834 # [GeV]
m_phi = 1.019461 # [GeV]

# filtering
###########

# cuts to be applied
selection = [
    f'ds_m_mass<={m_Bs}',
]
# convert selection to a string
selection = '(' + ') & ('.join(selection) + ')'


# import of MC Data
###################%

# aliases for importing from the root tree
aliases = {
    'pt_miss': f'ds_m_pt*{m_Bs}/ds_m_mass - kp_pt - km_pt - pi_pt - mu_pt'
    }

# list of branches in the MC tree that are used
keys_tree_MC = ['q2', 'e_star_mu3', 'pt_miss'] # observables
keys_tree_MC += ['ds_mass'] # others
keys_tree_MC += ['sig'] # signal

# open MC root file, filter according to cuts and save required branches
# in the MC tree
root_tree_MC = uproot.open("ROOT_data/reco_plus_gen_ntuple.root:tree")
tree_MC = root_tree_MC.arrays(keys_tree_MC, cut=selection, aliases=aliases,
                                library="np")




# %%##########
# Parameters #
##############

# list of observables that are used from the trees for the optimization
keys_obs = ['q2', 'ds_mass']#, 'e_star_mu', 'pt_miss']

# list of signals in the tree
keys_sig = ['-1', '0', '1', '2', '3']

# labels of the different channels
labels = {'-1': r'$H_b \rightarrow D_s + \mu$',
          '0': r"$B_s \rightarrow D_s \, \mu \nu$",
          '1': r"$B_s \rightarrow D_s^* \mu \nu$",
          '2': r"$B_s \rightarrow D_s \, \tau \nu$",
          '3': r"$B_s \rightarrow D^*_s \tau \nu$"}

# x labels of the different observables
labels_obs = {'q2': '$Q^2$ [GeV$^2$]',
              'ds_mass': '$m_{D_s}$ [GeV]'}


# total number of events in data
N_tot_MC = tree_MC['q2'].size

# number of events per signal in MC data
N_sig_MC = {sig: (tree_MC['sig']==int(sig)).sum() for sig in keys_sig}

# correctons for R, R_star (to include deterctor efficiency, etc.)
R_MC = 0.381
R_star_MC = 0.327
alpha = R_MC*N_sig_MC['0']/N_sig_MC['2']
alpha_star = R_star_MC*N_sig_MC['1']/N_sig_MC['3']

# ratio between number of mu* and mu events (to recudce # free params)
N_1_N_0_ratio_MC = N_sig_MC['1']/N_sig_MC['0']

# initial values for the parameters from MC data
np.random.seed(76820)
sigma = 0.05
R_init      = R_MC + np.random.normal(0, sigma*R_MC)
R_star_init = R_star_MC + np.random.normal(0, sigma*R_star_MC)
N_0_init    = N_sig_MC['0'] + np.random.normal(0, sigma*N_sig_MC['0'])
N_1_init    = N_0_init*N_1_N_0_ratio_MC

# parameters can only be allocated once -> use try and except to only run this
# part of the code, if the parameters don't already exist
try:
    R
except NameError:
    # Parameters that are optimized
    R = zfit.Parameter('R', R_init) # R(Ds)
    R_star = zfit.Parameter('R_star', R_star_init) # R(Ds*)
    N_0 = zfit.Parameter('N_0', N_0_init,) # sig 0
    N_1 = zfit.Parameter('N_1', N_1_init,) # sig 1

    # Functions for composed parameters
    def N_m1_(R, R_star, N_0, N_1):
        return N_tot_MC-N_0*(R/alpha+1)-N_1*(R_star/alpha_star+1)
    N_2_ = lambda R, N_0: R/alpha*N_0
    N_3_ = lambda R_star, N_1: R_star/alpha_star*N_1
    N_1_N_0_ratio_ = lambda N_0, N_1: N_1/N_0

    # Composed parameters
    N_m1 = zfit.ComposedParameter('N_m1', N_m1_,
                                  [R, R_star, N_0, N_1]) # bkg
    N_2 = zfit.ComposedParameter('N_2', N_2_, [R, N_0]) # sig 2
    N_3 = zfit.ComposedParameter('N_3', N_3_, [R_star, N_1]) # sig 3
    N_1_N_0_ratio = zfit.ComposedParameter('N_1_N_0_ratio', N_1_N_0_ratio_,
                                           [N_0, N_1]) # for the constraint

# parameters used to build the model
params = {'-1': N_m1, '0': N_0, '1': N_1, '2': N_2, '3': N_3}

# set parameter ranges
R.lower, R.upper           = 0.2, 0.5
R_star.lower, R_star.upper = 0.2, 0.5
N_0.lower, N_0.upper       = 0, 0.5*N_tot_MC
N_1.lower, N_1.upper       = 0, 0.5*N_tot_MC

# set initial parameter values
R.set_value(R_init)
R_star.set_value(R_star_init)
N_0.set_value(N_0_init)
N_1.set_value(N_1_init)

# set initial stepsizes
R.step_size, R_star.step_size = 1e-3, 1e-3
N_0.step_size = 1
N_1.step_size = 1

# gaussian constraints: 10% uncertainty for ratio N_1/N_0
constraints = [zfit.constraint.GaussianConstraint(
    N_1_N_0_ratio, N_1_N_0_ratio_MC, 0.1*N_1_N_0_ratio_MC
    )]


# %%################################
# creation of zfit model from Data #
####################################

# range of the observables for zfit and for plotting (if needed)
ranges = {'q2': (-0.4, 12.1),
          'pt_miss': (-1.069, 447.),
          'e_star_mu3': (0.16, 2.4),
          'ds_mass': (1.76, 2.17)}
plot_ranges = {'pt_miss': (-1.07, 70),
               'q2': (-1, 15)}

# position of legend
legend_pos = {'q2': 1}

# number of bins
bins = {'q2': 15, 'pt_miss': 200, 'e_star_mu': 20, 'ds_mass': 17}


# create hists from MC data and additional tau signals
hists = dict()
for obs in keys_obs:
    hists[obs] = dict()
    for sig in keys_sig:
        hists[obs][sig] = hist.Hist(hist.axis.Regular(bins[obs], *ranges[obs],
                                                      name=obs))
        hists[obs][sig].fill(tree_MC[obs][tree_MC['sig']==int(sig)])
        

# plot of hists
def obs_plot_sig(obs):
    for sig in keys_sig:
        hists[obs][sig].plot(density=True, yerr=True, label=labels[sig])
    if obs in plot_ranges.keys():
        plt.xlim(plot_ranges[obs])
    if obs in legend_pos:
        plt.legend(loc=legend_pos[obs])
    else:
        plt.legend()
    plt.show()
    return

for obs in keys_obs:
    obs_plot_sig(obs)

# create for every observable a pdf which is a sum of all signals
# using HistogramPDF
histPDFs = dict()
for obs in keys_obs:
    histPDFs[obs] = zfit.pdf.BinnedSumPDF(
        [zfit.pdf.HistogramPDF(hists[obs][sig]) for sig in keys_sig],
        fracs=list(params.values())
        )

# creating observables in zfit (binned)
binning = {obs: zfit.binned.RegularBinning(bins[obs], *ranges[obs], name=obs)
           for obs in keys_obs}
all_obs = {obs: zfit.Space(obs, ranges[obs], binning=binning[obs]) 
           for obs in keys_obs}

# load data into zfit (binned)
data_zfit = {obs_name: zfit.Data.from_numpy(obs, tree_MC[obs_name]
                                            ).to_binned(obs)
            for obs_name, obs in all_obs.items()}

# NLL function
NLL = zfit.loss.BinnedNLL(model=histPDFs.values(), data=data_zfit.values(),
                          constraints=constraints)

# plot of the stacked pdf split into signals with initial values
for obs in keys_obs:
    plot_stacked_hist(obs)
    mplhep.histplot(data_zfit[obs].to_hist(), label='combined MC', yerr=True,
                    color='k', histtype='step')
    if obs in plot_ranges:
        plt.xlim(plot_ranges[obs])
    if obs in legend_pos:
        plt.legend(loc=legend_pos[obs])
    else:
        plt.legend()
    plt.title('Initial values')
    plt.show()


# %%############
# miminization #
################

# selection of the minimizer
minimizer = zfit.minimize.Minuit()#gradient=True)

# run minimizer for first time
result = minimizer.minimize(NLL)
result.hesse()
n = 1
# run minimizer until it converged
while not result.converged:
    result = minimizer.minimize(NLL)
    n += 1

print(f'Number of iterations until the minimizer converged: {n}')


# %%#######
# Results #
###########

# Folder where the results are saved
folder = 'Results_of_Asimov_fit/'

# plot of the result split into signals
for obs in keys_obs:
    plot_stacked_hist(obs)
    mplhep.histplot(data_zfit[obs].to_hist(), label='combined MC', yerr=True,
                    color='k', histtype='step')
    if obs in plot_ranges:
        plt.xlim(plot_ranges[obs])
    if obs in legend_pos:
        plt.legend(loc=legend_pos[obs])
    else:
        plt.legend()
    plt.savefig(folder + f'stacked_hist_plot_{obs}.pdf')
    plt.title('Result')
    plt.show()


result.hesse()

# print results
print(result)
print('')
print_params()

R_res = result.values[R]
DeltaR_res = result.hesse()[R]['error']
R_star_res = result.values[R_star]
DeltaR_star_res = result.hesse()[R_star]['error']


# ranges for plotting
R_plot_range = (R_res - 2.5*DeltaR_res, R_res + 2.5*DeltaR_res)
R_star_plot_range = (R_star_res - 2.5*DeltaR_star_res,
                     R_star_res + 2.5*DeltaR_star_res)

# change ranges of parameters for plotting
R.lower, R.upper = R_plot_range
R_star.lower, R_star.upper = R_star_plot_range

# NLL function for plotting (with different ranges of the parameters)
NLL_plot = zfit.loss.BinnedNLL(model=histPDFs.values(),
                               data=data_zfit.values(),
                               constraints=constraints)

# plot of profile of R
plot_profile(R, R_plot_range, NLL_plot, name='$R(D_s)$', num=70)
plt.minorticks_on()
plt.grid(True)
#plt.savefig(folder + 'R_profile.pdf')
plt.show()

# plot of profile of R_star
plot_profile(R_star, R_star_plot_range, NLL_plot, name='$R(D_s^*)$', num=70)
plt.minorticks_on()
plt.grid(True)
#plt.savefig(folder + 'R_star_profile.pdf')
plt.show()

# plot contour of R, R_star
plot_contour(R, R_star, range1=R_plot_range, range2=R_star_plot_range,
             NLL=NLL_plot, xlabel='$R(D_s)$', ylabel='$R(D_s^*)$', num=50)
#plt.plot(R_init, R_star_init, 'ob', label='initial value')
plt.plot(R_MC, R_star_MC, '.r', label='MC value')
plt.legend()
plt.minorticks_on()
plt.grid(True)
#plt.savefig(folder + 'R_R_star_contour.pdf')
plt.show()


#########
# ToDos #
#########
# - 




# %%
