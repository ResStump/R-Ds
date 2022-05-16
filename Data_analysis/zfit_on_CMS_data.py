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

def plot_stacked_hist(obs):
    """Plots a stacked histogram of the different signals for the observable
    obs. The number of events per signal are the current parameters"""
    bins = hists[obs]['-1'].to_numpy()[1]
    counts = [param.value()/h.sum()*h.counts()
              for param, h in zip(params.values(), hists[obs].values())]
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

###################
# Other functions #
###################

def print_params():
    print('Resulting yields:')
    for k, p in params.items():
        print(f'Signal {k}:', p.value().numpy())

def det_ranges(obs):
    """Function that determines the maximum and minimum values of entry obs
    in tree_Reco and tree_Reco_tau """
    minimum = min(np.nanmin(tree_Reco[obs]), np.nanmin(tree_Reco_tau[obs]))
    maximum = max(np.nanmax(tree_Reco[obs]), np.nanmax(tree_Reco_tau[obs]))
    return (minimum, maximum)


# %%#####################################
# Data import, processing and filtering #
#########################################

# constants
m_Bs = 5.36688 # [GeV]

# filtering
###########

# cuts to be applied
"""cuts = f"(ds_m_mass<={m_Bs}) & (abs(mu_bs_dxy_sig)>5) & (mu_pt>8) & (HLT_Mu7_IP4==1) & (mu_rel_iso<.2) & (mu_charge*pi_charge<0) & (ds_vtx_prob>0.1) & (cos3D_ds_m>0.995) & (phi_vtx_prob>0.1) & (abs(phi_mass-1.020)<0.01)  & (lxyz_ds_m_sig>10)"
selection = ') & ('.join([
    '(abs(mu_bs_dxy_sig)>5',
    'mu_pt>8',
    'k1_pt>1.',
    'k2_pt>1.',
    'pi_pt>1.',
    'HLT_Mu7_IP4==1',
    'mu_rel_iso<.1',
    'ds_vtx_prob>0.1',
    'cos3D_ds_m>0.995',
    'phi_vtx_prob>0.1',
    'lxyz_ds_m_sig>10',
    'mu_id_medium==1',
    f'ds_m_mass<={m_Bs}',
    'mu_charge*pi_charge<0)'])"""

selection = [
    f'ds_m_mass<={m_Bs}',
    'ds_m_mass>2.16',
    'mu_pt>8',
    'mu_id_medium==1', 
    'mu_rel_iso<0.1',
    'abs(mu_bs_dxy_sig)>5',
    'k1_pt>1.',
    'k2_pt>1.',
    'pi_pt>1.',
    'phi_vtx_prob>0.1',
    'ds_vtx_prob>0.1',
    'cos3D_ds_m>0.995',
    'lxyz_ds_m_sig>10',
    'HLT_Mu7_IP4==1',
    'pt_miss>0',
    'mu_charge*pi_charge<0']
# convert selection to a string
selection = '(' + ') & ('.join(selection) + ')'

# possible changes in the selection:
# - mu_rel_iso: <0.1 -> <1
# - ds_vtx_prob: >0.1 -> >0.05
# - lxyz_ds_m_sig: >10 -> >5,
# - pt_miss: >0 -> >1.6501322
#   (because that's the smalles pt_miss for tau signals in Reco)
# - ds_m_mass: >2.16 -> =>2.3491971
# - someting like abs(mu_bs_dxy) > 0.01


# import of CMS Data
####################

# aliases for importing from the root tree
aliases = {
    'pt_miss': f'ds_m_pt*{m_Bs}/ds_m_mass - k1_pt - k2_pt - pi_pt - mu_pt'
    }

# list of branches in the CMS tree that are used
keys_tree_CMS = ['q2', 'e_star_mu', 'pt_miss'] # observables
keys_tree_CMS += ['ds_mass'] # others

# open CMS root file, filter according to cuts and save required branches
# in the CMS tree
root_tree_CMS = uproot.open("ROOT_data/data_skimmed.root:tree")
tree_CMS = root_tree_CMS.arrays(keys_tree_CMS, cut=selection, aliases=aliases,
                                library="np")

# create tree with combinatorial background (pi, mu same charge) from 
# CMS root file
selection_comb = selection.replace('mu_charge*pi_charge<0',
                                   'mu_charge*pi_charge>0')
tree_comb = root_tree_CMS.arrays(keys_tree_CMS, cut=selection_comb,
                                 aliases=aliases, library="np")

# add sig branch containing -2 to tree_comb
tree_comb['sig'] = -2*np.ones_like(tree_comb['q2'])


# import of Reco Data (for bilding the zfit model)
##################################################

# list of branches in the reco tree that are used
keys_tree_Reco = keys_tree_CMS + ['sig'] # signal

# open Reco root file, filter according to cuts and save required branches
# in the Reco tree
root_tree_Reco = uproot.open("ROOT_data/inclusive_mc_3may.root:tree")
tree_Reco = root_tree_Reco.arrays(keys_tree_Reco, cut=selection,
                                  aliases=aliases, library="np")

# import the additional tau signals
root_tree_Reco_tau = uproot.open("ROOT_data/tau_mc_3may.root:tree")
tree_Reco_tau = root_tree_Reco_tau.arrays(keys_tree_Reco, cut=selection,
                                          aliases=aliases, library="np")


# %%##########
# Parameters #
##############

# list of observables that are used from the trees for the optimization
keys_obs = ['q2', 'e_star_mu']#, 'ds_mass', 'pt_miss']

# list of signals in the tree
keys_sig = ['-2', '-1', '0', '1', '2', '3']

# labels of the different channels
labels = {'-2': 'comb. bkg.',
          '-1': r'$H_b \rightarrow D_s + \mu$',
          '0': r"$B_s \rightarrow D_s \, \mu \nu$",
          '1': r"$B_s \rightarrow D_s^* \mu \nu$",
          '2': r"$B_s \rightarrow D_s \, \tau \nu$",
          '3': r"$B_s \rightarrow D^*_s \tau \nu$"}

# total number of events in data and in reco data
N_tot_CMS = tree_CMS['q2'].size
N_tot_Reco = tree_Reco['q2'].size

# initial values for the parameters (scaled to # events in data)
N_sig_init = {sig: (tree_Reco['sig']==int(sig)).sum()*N_tot_CMS/N_tot_Reco
                   for sig in keys_sig}
R_init      = N_sig_init['2']/N_sig_init['0']
R_star_init = N_sig_init['3']/N_sig_init['1']
N_0_init    = 0.55*N_sig_init['0']
N_1_init    = 0.55*N_sig_init['1']
N_comb_init = 1.45*tree_comb['q2'].size

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
    N_comb = zfit.Parameter('N_comb', N_comb_init) # combinatorial background

    # Functions for composed parameters
    def N_m1_(R, R_star, N_0, N_1, N_comb):
        return N_tot_CMS - N_0*(R+1) - N_1*(R_star+1) - N_comb
        #return zfit.z.numpy.max([N_tot_CMS-N_0*(R+1)-N_1*(R_star+1)-N_comb,
        #                         1e-6])
    N_2_ = lambda R, N_0: R*N_0
    N_3_ = lambda R_star, N_1: R_star*N_1

    # Composed parameters
    N_m1 = zfit.ComposedParameter('N_m1', N_m1_,
                                  [R, R_star, N_0, N_1, N_comb]) # bkg
    N_2 = zfit.ComposedParameter('N_2', N_2_, [R, N_0]) # sig 2
    N_3 = zfit.ComposedParameter('N_3', N_3_, [R_star, N_1]) # sig 3

# parameters used to build the model
params = {'-2': N_comb, '-1': N_m1, '0': N_0, '1': N_1, '2': N_2, '3': N_3}

# set parameter ranges
R.lower, R.upper           = 0, 0.7
R_star.lower, R_star.upper = 0, 0.4
N_0.lower, N_0.upper       = 0, 0.5*N_tot_CMS
N_1.lower, N_1.upper       = 0, 0.5*N_tot_CMS
N_comb.lower, N_comb.upper = 0, 0.8*N_tot_CMS

# set initial parameter values
R.set_value(R_init)
R_star.set_value(R_star_init)
N_0.set_value(N_0_init)
N_1.set_value(N_1_init)
N_comb.set_value(N_comb_init)

# set stepsizes
R.step_size, R_star.step_size = 1e-4, 1e-4
N_0.step_size, N_1.step_size, N_comb.step_size = 1, 1, 1

"""# initial values for the parameters (scaled to # events in data)
N_sig_init = {sig: (tree_Reco['sig']==int(sig)).sum()*N_tot_CMS/N_tot_Reco
                   for sig in keys_sig}
R_init      = N_sig_init['2']/N_sig_init['0']
R_star_init = N_sig_init['3']/N_sig_init['1']
N_init      = 0.70*N_sig_init['0'] + N_sig_init['2']
N_star_init = 0.70*N_sig_init['1'] + N_sig_init['3']
N_comb_init = 1.30*tree_comb['q2'].size

# parameters can only be allocated once -> use try and except to only run this
# part of the code, if the parameters don't already exist
try:
    R
except NameError:
    # Parameters that are optimized
    R = zfit.Parameter('R', R_init) # R(Ds)
    R_star = zfit.Parameter('R_star', R_star_init) # R(Ds*)
    N = zfit.Parameter('N', N_init,) # N_0 + N_2
    N_star = zfit.Parameter('N_star', N_star_init,) # N_1 + N_3
    N_comb = zfit.Parameter('N_comb', N_comb_init) # combinatorial background

    # Functions for composed parameters
    N_m1_ = lambda N, N_star, N_comb: N_tot_CMS - N - N_star - N_comb
    N_0_ = lambda R, N: N/(R+1)
    N_1_ = lambda R_star, N_star: N_star/(R_star+1)
    N_2_ = lambda R, N: R*N/(R+1)
    N_3_ = lambda R_star, N_star: R_star*N_star/(R_star+1)

    # Composed parameters
    N_m1 = zfit.ComposedParameter('N_m1', N_m1_, [N, N_star, N_comb]) # bkg
    N_0 = zfit.ComposedParameter('N_0', N_0_, [R, N]) # sig 0
    N_1 = zfit.ComposedParameter('N_1', N_1_, [R_star, N_star]) # sig 1
    N_2 = zfit.ComposedParameter('N_2', N_2_, [R, N]) # sig 2
    N_3 = zfit.ComposedParameter('N_3', N_3_, [R_star, N_star]) # sig 3

# parameters used to build the model
params = {'-2': N_comb, '-1': N_m1, '0': N_0, '1': N_1, '2': N_2, '3': N_3}

# set parameter ranges
R.lower, R.upper           = 0, 0.1
R_star.lower, R_star.upper = 0, 0.1
N.lower, N.upper           = 0, N_tot_CMS
N_star.lower, N_star.upper = 0, N_tot_CMS
N_comb.lower, N_comb.upper = 0, 100_000

# set initial parameter values
R.set_value(R_init)
R_star.set_value(R_star_init)
N.set_value(N_init)
N_star.set_value(N_star_init)
N_comb.set_value(N_comb_init)

# set stepsizes
R.step_size, R_star.step_size = 0.0001, 0.0001
N.step_size, N_star.step_size, N_comb.step_size = 1, 1, 1"""

# gaussian constraints
"""R_exp = R_init # expected value
DeltaR = R_init*0.2 # 20% uncertainty for R_exp

R_star_exp = R_star_init # expected value
DeltaR_star = R_star_init*0.2 # 20% uncertainty for R_star_exp"""

constraints = []
""" # [zfit.constraint.GaussianConstraint(R, R_exp, DeltaR),
               zfit.constraint.GaussianConstraint(R_star, R_star_exp,
                                                  DeltaR_star)]"""


# %%################################
# creation of zfit model from Data #
####################################

# range of the observables for zfit and for plotting (if needed)
ranges = {'q2': (-0.39773342, 12.068998),
          'pt_miss': (-1.0685997, 447.23743),
          'e_star_mu': (0.16563936, 2.3912044),
          'ds_mass': (1.9185385, 2.0183234)}
plot_ranges = {'pt_miss': (-1.0685997, 70)}

# position of legend
legend_pos = {'q2': 2}

# number of bins
bins = {'q2': 25, 'pt_miss': 200, 'e_star_mu': 20, 'ds_mass': 25}

# create hists from reco data and additional tau signals
hists = dict()
for obs in keys_obs:
    hists[obs] = dict()
    for sig in keys_sig:
        hists[obs][sig] = hist.Hist(hist.axis.Regular(bins[obs], *ranges[obs],
                                                      name=obs))
        # for the tau signals only use the tau_mc_3may.root tree
        if sig not in ['2', '3']:
            hists[obs][sig].fill(tree_Reco[obs][tree_Reco['sig']==int(sig)])
            hists[obs][sig].fill(tree_comb[obs][tree_comb['sig']==int(sig)])
        else:
            hists[obs][sig].fill(tree_Reco_tau[obs][tree_Reco_tau['sig']==int(sig)])
        


# plot of hists
def obs_plot_sig(obs):
    for sig in keys_sig:
        hists[obs][sig].plot(density=True, yerr=False, label=labels[sig])
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
data_zfit = {obs_name: zfit.Data.from_numpy(obs, tree_CMS[obs_name])
                       .to_binned(obs)
             for obs_name, obs in all_obs.items()}

# NLL function
NLL = zfit.loss.BinnedNLL(model=histPDFs.values(), data=data_zfit.values(),
                          constraints=constraints)

# plot of the stacked pdf split into signals with initial values
for obs in keys_obs:
    plot_stacked_hist(obs)
    mplhep.histplot(data_zfit[obs].to_hist(), label='Observed', yerr=False,
                    color='k')
    if obs in plot_ranges:
        plt.xlim(plot_ranges[obs])
    plt.legend()
    plt.title('Initial values')
    plt.show()


# %%############
# miminization #
################

"""# plots before minimization
for obs in keys_obs:
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
    plot_stacked_hist(obs)
    mplhep.histplot(data_zfit[obs].to_hist(), label='Observed', yerr=False,
                    color='k')
    if obs in plot_ranges:
        plt.xlim(plot_ranges[obs])
    plt.legend()
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
R_plot_range = (R_res - 1.5*DeltaR_res, R_res + 1.5*DeltaR_res)
R_star_plot_range = (R_star_res - 1.5*DeltaR_star_res,
                     R_star_res + 1.5*DeltaR_star_res)

# change ranges of parameters for plotting
R.lower, R.upper = R_plot_range
R_star.lower, R_star.upper = R_star_plot_range

# NLL function for plotting (with different ranges of the parameters)
NLL_plot = zfit.loss.BinnedNLL(model=histPDFs.values(),
                               data=data_zfit.values(),
                               constraints=constraints)

# plot of profile of R
plot_profile(R, R_plot_range, NLL_plot)
plt.minorticks_on()
plt.grid(True)
plt.show()

# plot of profile of R_star
plot_profile(R_star, R_star_plot_range, NLL_plot, name='$R^*$')
plt.minorticks_on()
plt.grid(True)
plt.show()

# plot contour of R, R_star
plot_contour(R, R_star, range1=R_plot_range, range2=R_star_plot_range,
             NLL=NLL_plot, ylabel='$R^*$')
plt.minorticks_on()
plt.grid(True)
plt.show()


#########
# ToDos #
#########
# - Also plot 2 sigma contour? i.e. 2*DeltaNLL = 4 = 2**2
# - Add plot of ds_mass, even if its no an observable used for minimization




# %%
