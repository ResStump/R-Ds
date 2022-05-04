# %%
import numpy as np
import uproot
import matplotlib.pyplot as plt
import mplhep
import hist
import os
os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1" # disables first zfit warning
import zfit


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

def plot_hist_data_sig(obs):
    """Plots a stacked histogram of the different signals for the observable
    obs. The number of events per signal are the current parameters"""
    bins = hists[obs]['-1'].to_numpy()[1]
    counts = [params[sig].value()*hists[obs][sig].density()
              for sig in keys_sig]
    mplhep.histplot(counts, bins, stack=True, label=labels.values(),
                    histtype='fill')
    plt.xlabel(obs)
    plt.ylabel('Number of events')
    plt.legend()
    return

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
    plt.plot(param1_result, param2_result, '.k')
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


# %%#####################################
# Data import, processing and filtering #
#########################################

# import of CMS Data (here, CMS data is simulated by the Reco data)
###################################################################

# list of branches in the CMS tree that are used
keys_tree_CMS = ['q2', 'e_star_mu3'] # observables
keys_tree_CMS += ['ds_m_mass'] # filtering
keys_tree_CMS += ['ds_m_pt', 'mu_pt', 'kp_pt', 'km_pt', 'pi_pt'] # pt_miss

# open CMS root file using lazy and save required branches in the CMS tree
lazy_tree_CMS = uproot.lazy("ROOT_data/reco_plus_gen_ntuple.root:tree")
tree_CMS = {key: lazy_tree_CMS[key].to_numpy() for key in keys_tree_CMS}


# import of Reco Data (for bilding the zfit model)
##################################################

# list of branches in the reco tree that are used
keys_tree_Reco = keys_tree_CMS + ['sig']

# open reco root file using lazy and save required branches in the reco tree
lazy_tree_Reco = uproot.lazy("ROOT_data/reco_plus_gen_ntuple.root:tree")
tree_Reco = {key: lazy_tree_Reco[key].to_numpy() for key in keys_tree_Reco}


# Processing
############

# constants
m_Bs = 5.36688 # [GeV]

# add pt_miss to reco tree and to CMS data tree
tree_Reco['pt_miss'] = tree_Reco['ds_m_pt']*m_Bs/tree_Reco['ds_m_mass'] \
                       - tree_Reco['km_pt'] - tree_Reco['kp_pt'] \
                       - tree_Reco['pi_pt']  - tree_Reco['mu_pt']
tree_CMS['pt_miss'] = tree_CMS['ds_m_pt']*m_Bs/tree_CMS['ds_m_mass'] \
                      - tree_CMS['km_pt'] - tree_CMS['kp_pt'] \
                      - tree_CMS['pi_pt']  - tree_CMS['mu_pt']

# add q2 a second time to look at how uncertainty change, when used the same
# data twice
tree_Reco['q2_copy'] = np.copy(tree_Reco['q2'])
tree_CMS['q2_copy'] = np.copy(tree_CMS['q2'])


# Filtering
###########

# index for which bool_arr != True are set to nan in tree
def tree_filtering(tree, bool_arr):
    bool_arr = np.logical_not(bool_arr) # inverting the condition
    for k in tree.keys():
        if k == 'sig': # don't apply filter to 'sig'
            continue
        tree[k][bool_arr] = np.nan
    return

filters_Reco = dict()
filters_CMS = dict()
# cut for B_s mass at 5.36688 GeV (PDG, Fit)
m_Bs = 5.36688 # [GeV]
filters_Reco['ds_m_mass'] = tree_Reco['ds_m_mass'] <= m_Bs
filters_CMS['ds_m_mass'] = tree_CMS['ds_m_mass'] <= m_Bs

def filterAll():
    for bool_arr in filters_Reco.values():
        tree_filtering(tree_Reco, bool_arr)
    for bool_arr in filters_CMS.values():
        tree_filtering(tree_CMS, bool_arr)
    return

filterAll()


# %%##########
# Parameters #
##############

# list of observables that are used from the trees for the optimization
keys_obs = ['q2', 'pt_miss']#, 'q2_copy']#, 'e_star_mu3']

# list of signals in the tree
keys_sig = ['-1', '0', '1', '2', '3']

# labels of the different channels
labels = {'-1': 'background',
          '0': r"$D_s \, \mu \nu$ channel",
          '1': r"$D_s^* \mu \nu$ channel",
          '2': r"$D_s \, \tau \nu$ channel",
          '3': r"$D^*_s \tau \nu$ channel"}

# total number of events in data
N_tot = tree_CMS['q2'].size - np.isnan(tree_CMS['q2']).sum()

# True parameters
N_sig_true = {sig: (tree_Reco['sig']==int(sig)).sum() for sig in keys_sig}
R_true = N_sig_true['2']/N_sig_true['0']
R_star_true = N_sig_true['3']/N_sig_true['1']
N_true = N_sig_true['0'] + N_sig_true['2']
N_star_true = N_sig_true['1'] + N_sig_true['3']


# parameters can only be allocated once -> use try and except to only run this
# part of the code, if the parameters don't already exist
try:
    R
except NameError:
    # Parameters that are optimized
    R = zfit.Parameter('R', R_true, -1, 10, step_size=0.0001)
    R_star = zfit.Parameter('R_star', R_star_true, -1, 10,
                            step_size=0.0001)
    N = zfit.Parameter('N', N_true, 0, 1e6, step_size=1) # N_0 + N_2
    N_star = zfit.Parameter('N_star', N_star_true, 0, 1e6,
                            step_size=1) # N_1 + N_3

    # Composed parameters
    N_m1 = zfit.ComposedParameter(
        'N_m1', lambda N, N_star: N_tot - N - N_star, params = [N, N_star]
    ) # background
    N_0 = zfit.ComposedParameter(
        'N_0', lambda R, N: N/(R+1), params=[R, N]
    ) # signal 0
    N_1 = zfit.ComposedParameter(
        'N_1', lambda R_star, N_star: N_star/(R_star+1),
        params=[R_star, N_star]
    ) # signal 1
    N_2 = zfit.ComposedParameter(
        'N_2', lambda R, N: R*N/(R+1), params=[R, N]
    ) # signal 2
    N_3 = zfit.ComposedParameter(
        'N_3', lambda R_star, N_star: R_star*N_star/(R_star+1),
        params=[R_star, N_star]
    ) # signal 3

    # parameters used to build the model
    params = {'-1': N_m1, '0': N_0, '1': N_1, '2': N_2, '3': N_3}

# gaussian constraints
R_exp = R_true # expected value
DeltaR = R_true*0.2 # 20% uncertainty for R_exp

R_star_exp = R_star_true # expected value
DeltaR_star = R_star_true*0.2 # 20% uncertainty for R_star_exp

constraints = []
""" # [zfit.constraint.GaussianConstraint(R, R_exp, DeltaR),
               zfit.constraint.GaussianConstraint(R_star, R_star_exp,
                                                  DeltaR_star)]"""


# %%################################
# creation of zfit model from Data #
####################################

# range of the observables for zfit and for plotting (if needed)
ranges = {obs: (np.nanmin(tree_Reco[obs]), np.nanmax(tree_Reco[obs]))
          for obs in keys_obs}
plot_ranges = {'pt_miss': (-1.0685997, 70)}

# number of bins
bins = {'q2': 25, 'pt_miss': 200, 'q2_copy': 25, 'e_star_mu3': 30}

# create hists from reco data
hists = dict()
for obs in keys_obs:
    hists[obs] = dict()
    for sig in keys_sig:
        hists[obs][sig] = hist.Hist(hist.axis.Regular(bins[obs], *ranges[obs],
                                                      name=obs))
        hists[obs][sig].fill(tree_Reco[obs][tree_Reco['sig']==int(sig)])


# plot of hists
def obs_plot_sig(obs):
    if obs in plot_ranges.keys():
            plt.xlim(plot_ranges[obs])
    for sig in keys_sig:
        hists[obs][sig].plot(density=True, yerr=False, label=labels[sig])
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
data_zfit = {obs_name: zfit.Data.from_numpy(obs, tree_CMS[obs_name])
                       .to_binned(obs)
             for obs_name, obs in all_obs.items()}


# NLL function
NLL = zfit.loss.BinnedNLL(model=histPDFs.values(), data=data_zfit.values(),
                          constraints=constraints)

# set initial parameter values
R.set_value(R_true)
R_star.set_value(R_star_true)
N.set_value(N_true)
N_star.set_value(N_star_true)


# %%############
# miminization #
################

# plots before minimization
"""for obs in keys_obs:
    if obs in plot_ranges.keys():
        plt.xlim(plot_ranges[obs]) 
    plot_hist_model_and_data(histPDFs[obs], data_zfit[obs])
    plt.title('Befor minimizing')
    plt.show()"""

# selection of the minimizer
minimizer = zfit.minimize.Minuit()#gradient=True)

# run minimizer for first time
result = minimizer.minimize(NLL)
# calculate uncertainty of result, then minimize again and calculate the
# uncertainty again. For some reason this reduces the uncertainty of the result
# such that it is close to the 2*DeltaNLL = 1 value
result.hesse()
n = 1
# run minimizer until it converged but at least 1 additional time for the hesse
# (see comment above)
while not result.converged or n==1:
    result = minimizer.minimize(NLL)
    n += 1

"""# run minimizer until it converged and until the parameters don't change
# much anymoretemp_results = np.array(result.values)
params_stable = False
while not result.converged and not params_stable:
    result = minimizer.minimize(NLL)
    n += 1
    params_stable = np.allclose(np.array(result.values), temp_results,
                                rtol=1e-5, atol=1e-8)
    temp_results = np.array(result.values)"""

print(f'Number of iterations until the minimizer converged: {n}')

# plots after minimization
"""for obs in keys_obs:
    if obs in plot_ranges.keys():
        plt.xlim(plot_ranges[obs])
    plot_hist_model_and_data(histPDFs[obs], data_zfit[obs])
    plt.title('After minimizing')
    plt.show()"""


# %%#######
# Results #
###########

# plot of the result split into signals
for obs in keys_obs:
    if obs in plot_ranges.keys():
        plt.xlim(plot_ranges[obs])
    plot_hist_data_sig(obs)
    plt.show()

result.hesse()

# results
print(result)
R_res = result.values[R]
DeltaR_res = result.hesse()[R]['error']
R_star_res = result.values[R_star]
DeltaR_star_res = result.hesse()[R_star]['error']

# ranges for plotting
R_range = (R_res - 1.5*DeltaR_res, R_res + 1.5*DeltaR_res)
R_star_range = (R_star_res - 1.5*DeltaR_star_res,
                R_star_res + 1.5*DeltaR_star_res)


# plot of profile of R
plot_profile(R, R_range, NLL)
plt.minorticks_on()
plt.grid(True)
plt.show()

# plot of profile of R_star
plot_profile(R_star, R_star_range, NLL, name='$R^*$')
plt.minorticks_on()
plt.grid(True)
plt.show()

# plot contour of R, R_star
plot_contour(R, R_star, range1=R_range, range2=R_star_range, NLL=NLL,
             ylabel='$R^*$')
plt.minorticks_on()
plt.grid(True)
plt.show()


#########
# ToDos #
#########
# - Create selection to handle (artificial) K pi ambiguity
# - Also plot 2 sigma contour? i.e. 2*DeltaNLL = 4 = 2**2
# - Remove q2_copy from code if no longer needed (tree and bins)



# %%
