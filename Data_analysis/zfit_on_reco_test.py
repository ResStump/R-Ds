# %%
import numpy as np
import uproot
import matplotlib.pyplot as plt
import mplhep
import hist
import os
os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1" # disables first zfit warning
import zfit

def plot_hist_model(model, data_bin):
    mplhep.histplot(model.to_hist(), label='model')
    mplhep.histplot(data_bin.to_hist(), label='data')
    plt.legend()
    return

# plot profile
def plot_profile(param, range, NLL, name=None, num=50):
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

# plot contour
def plot_contour(param1, param2, range1, range2, NLL, xlabel=None, ylabel=None,
                 num=25):
    param1_result = param1.value()
    param2_result = param2.value()
    NLL0 = NLL.value()
    X, Y = np.meshgrid(
        np.linspace(float(range1[0]), float(range1[1]), num=num),
        np.linspace(float(range2[0]), float(range2[1]), num=num)
    )
    Z = np.empty(X.shape)
    
    for i in range(num):
        for j in range(num):
            param1.set_value(X[i, j])
            param2.set_value(Y[i, j])
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


# %%#########################
# Data import and filtering #
#############################

# list of branches in the tree that are used
keys_tree = ['q2', 'ds_m_mass', 'm2_miss', 'sig']

# list of observables that are used from the tree for the optimization
keys_obs = ['q2', 'm2_miss']

# list of signals in the tree
keys_sig = ['-1', '0', '1', '2', '3']

# open root files using lazy and save required branches in tree
lazy_tree = uproot.lazy("ROOT_data/reco_plus_gen_ntuple.root:tree")
tree = {key: lazy_tree[key].to_numpy() for key in keys_tree}

# total number of events
N_tot = tree['q2'].size


# Filtering
###########

# index for which bool_arr != True are set to nan in tree
def tree_filtering(bool_arr):
    bool_arr = np.logical_not(bool_arr) # inverting the condition
    for k in tree.keys():
        if k == 'sig': # don't apply filter to 'sig'
            continue
        tree[k][bool_arr] = np.nan
    return

filters = dict()
# cut for B_s mass at 5.36688 GeV (PDG, Fit)
m_Bs = 5.36688 # [GeV]
filters['ds_m_mass'] = tree['ds_m_mass'] <= m_Bs

def filterAll():
    for bool_arr in filters.values():
        tree_filtering(bool_arr)
    return

filterAll()


# %%##########
# Parameters #
##############

# True parameters
N_true = {sig: (tree['sig']==int(sig)).sum() for sig in keys_sig}
R_true = N_true['2']/N_true['0']
R_star_true = N_true['3']/N_true['1']


# Parameters that are optimized
R = zfit.Parameter('R', R_true + 0.01, 0, 100)
R_star = zfit.Parameter('R_star', R_star_true + 0.02, 0, 100)
N = zfit.Parameter('N', (N_true['0']+N_true['2'])+50000, 0, 1e6,
                   step_size=1) # N_0 + N_2
N_star = zfit.Parameter('N_star', (N_true['1']+N_true['3'])-80000, 0, 1e6,
                        step_size=1) # N_1 + N_3

# Composed parameters
N_m1 = zfit.ComposedParameter('N_m1', lambda N, N_star: N_tot - N - N_star,
                              params = [N, N_star])
N_0 = zfit.ComposedParameter('N_0', lambda R, N: N/(R+1),
                             params=[R, N]) # signal 0
N_1 = zfit.ComposedParameter('N_1', lambda R_star, N_star: N_star/(R_star+1),
                             params=[R_star, N_star]) # signal 1
N_2 = zfit.ComposedParameter('N_2', lambda R, N: R*N/(R+1),
                             params=[R, N]) # signal 2
N_3 = zfit.ComposedParameter('N_3',
                             lambda R_star, N_star: R_star*N_star/(R_star+1),
                             params=[R_star, N_star]) # signal 3

# parameters used to build the model
params = {'-1': N_m1, '0': N_0, '1': N_1, '2': N_2, '3': N_3}

# gaussian constraints
R_exp = R_true # expected value
DeltaR = R_true*0.1 # 10% uncertainty for R_exp

R_star_exp = R_star_true # expected value
DeltaR_star = R_star_true*0.1 # 10% uncertainty for R_star_exp

constraints = [zfit.constraint.GaussianConstraint(R, R_exp, DeltaR),
               zfit.constraint.GaussianConstraint(R_star, R_star_exp,
                                                  DeltaR_star)]


# %%#################################################
# creation of zfit model from Data and minimization #
#####################################################

# range of the observables
ranges = {obs: (np.nanmin(tree[obs]), np.nanmax(tree[obs]))
          for obs in keys_obs}

# todo from here!!!!!!

# number of Bins
bins = {'obs1': 50, 'obs2': 50}

# create hists from 'gen' data
hists = dict()
for obs in keys_obs:
    hists[obs] = dict()
    for sig in keys_sig:
        hists[obs][sig] = hist.Hist(hist.axis.Regular(bins[obs], *ranges[obs],
                                                      name=obs))
        hists[obs][sig].fill(tree[obs][tree['sig']==int(sig)])


# plot of hists
def obs_plot_sig(obs):
    for sig in keys_sig:
        hists[obs][sig].plot(density=True)
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
data_zfit = {obs_name: zfit.Data.from_numpy(obs, tree_Data[obs_name])
             .to_binned(obs)
             for obs_name, obs in all_obs.items()}


# NLL function
NLL = zfit.loss.BinnedNLL(model=histPDFs.values(), data=data_zfit.values(),
                          constraints=constraints)

# set parameter values back
R.set_value(R_true+1)
R_star.set_value(R_star_true+0.7)
N.set_value((N_0_true+N_2_true)+20000)
N_star.set_value((N_1_true+N_2_true)-10000)

# plots before minimization
for obs in keys_obs:
    plot_hist_model(histPDFs[obs], data_zfit[obs])
    plt.title('Befor minimizing')
    plt.show()

# miminization
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(NLL)
result.hesse()

# plots after minimization
for obs in keys_obs:
    plot_hist_model(histPDFs[obs], data_zfit[obs])
    plt.title('After minimizing')
    plt.show()

print(result)

# plot of profile of R
plot_profile(R, (R.value()-0.05, R.value()+0.05), NLL)
plt.minorticks_on()
plt.grid(True)
plt.show()

# plot of profile of R_star
plot_profile(R_star, (R_star.value()-0.05, R_star.value()+0.05), NLL,
             name='$R^*$')
plt.minorticks_on()
plt.grid(True)
plt.show()

# plot contour of R, R_star
plot_contour(R, R_star, range1=(R.value()-0.05, R.value()+0.05),
             range2=(R_star.value()-0.05, R_star.value()+0.05),
             NLL=NLL, ylabel='$R^*$')
plt.minorticks_on()
plt.grid(True)
plt.show()


#########
# ToDos #
#########
# - Create selection to handle (artificial) K pi ambiguity



# %%
