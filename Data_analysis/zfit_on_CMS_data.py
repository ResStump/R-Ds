# %%
import numpy as np
import uproot
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.ROOT)
plt.rcParams['figure.dpi'] = 50
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

# plots visualization of ds_mass shift with final results of fit
def plot_ds_mass_shift(nbins=None, save_plot=False):
    if nbins == None:
        nbins = bins['ds_mass']
    bin_edges = np.histogram_bin_edges(1, nbins, ranges['ds_mass'])
    data = {'-2': tree_comb['ds_mass'],
            '-1': tree_Reco['ds_mass'][tree_Reco['sig']==int('-1')],
             '0': tree_Reco['ds_mass'][tree_Reco['sig']==int('0')],
             '1': tree_Reco['ds_mass'][tree_Reco['sig']==int('1')],
             '2': tree_Reco_tau['ds_mass'][tree_Reco_tau['sig']==int('2')],
             '3': tree_Reco_tau['ds_mass'][tree_Reco_tau['sig']==int('3')]}
    counts = np.zeros(nbins)
    counts_no_shift = np.zeros(nbins)
    for sig in keys_sig:
        factor = params[sig].numpy()/data[sig].size
        counts += factor*np.histogram(data[sig], bin_edges)[0]
        if sig == '-2':
            counts_no_shift += factor*np.histogram(data[sig], bin_edges)[0]
        else:
            counts_no_shift += factor*np.histogram(data[sig]/ds_mass_shift,
                                                  bin_edges)[0]
    hist_CMS = np.histogram(tree_CMS['ds_mass'], bin_edges)

    mplhep.histplot((counts, bin_edges), label='MC', yerr=False)
    mplhep.histplot(hist_CMS, label='observed', yerr=True)
    plt.xlabel(labels_obs['ds_mass'])
    plt.ylabel('Number of events')
    plt.legend()
    if save_plot:
        plt.savefig(folder + 'ds_mass_with_mass_shift.pdf')
    plt.title('With mass shift')
    plt.show()

    mplhep.histplot((counts_no_shift, bin_edges), label='MC',
                    yerr=False)
    mplhep.histplot(hist_CMS, label='observed', yerr=True)
    plt.xlabel(labels_obs['ds_mass'])
    plt.ylabel('Number of events')
    plt.legend()
    if save_plot:
        plt.savefig(folder + 'ds_mass_without_mass_shift.pdf')
    plt.title('Without mass shift')
    plt.show()
    return


# Likelihood scans
###################

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

def print_params(rounded=True):
    print('Resulting yields:')
    for k, p in params.items():
        value = p.value().numpy()
        if rounded:
            value = round(p.value().numpy())
        print(f'Signal {k}:', value)

def det_ranges(obs):
    """Function that determines the maximum and minimum values of entry obs
    in tree_Reco and tree_Reco_tau """
    minimum = min(np.nanmin(tree_Reco[obs]), np.nanmin(tree_Reco_tau[obs]))
    maximum = max(np.nanmax(tree_Reco[obs]), np.nanmax(tree_Reco_tau[obs]))
    return (minimum, maximum)

def weights_for_comb_bkg(obs, nbins=None, make_plots=False, save_plot=False):
    """Returns the weights for the combinatorial background of the observable
    obs to correct the difference in shape for same and opposite mu and pi
    charge."""
    # selection to isolate the combinatorial background from the other signals
    selection = [
        f'ds_m_mass<={m_Bs}',
        'mu_pt>8',
        'mu_id_medium==1',
        'abs(mu_bs_dxy_sig)>5',
        'k1_pt>1.',
        'k2_pt>1.',
        'pi_pt>1.',
        'phi_vtx_prob>0.1',
        'ds_vtx_prob>0.1',
        'cos3D_ds_m>0.995',
        'HLT_Mu7_IP4==1',
        'pt_miss>0',
        'mu_charge*pi_charge<0',
        'mu_rel_iso>0.3',
        'lxyz_ds_m_sig<10',
        'lxyz_ds_sig<10',
        #f'abs(ds_mass - {m_Ds})>0.01',
        ]
    # convert selection to a string
    selection = '(' + ') & ('.join(selection) + ')'

    # import of the comb. bkg. with opposite mu and pi charge
    comb_opposite = root_tree_CMS.arrays(obs, cut=selection, aliases=aliases,
                                            library="np")[obs]

    # import of the comb. bkg. with opposite mu and pi charge
    selection = selection.replace('mu_charge*pi_charge<0',
                                        'mu_charge*pi_charge>0')
    comb_same = root_tree_CMS.arrays(obs, cut=selection, aliases=aliases,
                                        library="np")[obs]

    # choose bins
    if not nbins:
        nbins = bins[obs]

    # calculation the histogram
    counts_opposite, binning = np.histogram(comb_opposite, nbins, ranges[obs],
                                            density=True)
    counts_same, binning = np.histogram(comb_same, nbins, ranges[obs],
                                        density=True)

    # determine the weights of the bins as the ratio between the counts
    weights_binned = counts_opposite/counts_same
    # set possible nans to one
    weights_binned = np.nan_to_num(weights_binned, nan=1.0)

    if make_plots or save_plot:
        plt.hist(comb_same, bins[obs], density=True, range=ranges[obs],
                 label='same charge', histtype='step')
        plt.hist(comb_opposite, bins[obs], density=True, range=ranges[obs],
                 label='opposite charge', histtype='step')
        plt.xlabel(labels_obs[obs])
        plt.ylabel('a.u.')
        if obs in legend_pos:
            plt.legend(loc=legend_pos[obs])
        else:
            plt.legend()
        if save_plot:
            plt.savefig(folder + f'comb_bkg_{obs}.pdf')
        plt.title('comb_bkg')
        plt.show()
        mplhep.histplot(weights_binned, binning)
        plt.xlabel(labels_obs[obs])
        plt.ylabel('Weights')
        plt.show()

    # calculate weights for the individual events
    weights = rv_histogram((weights_binned, binning)).pdf(tree_comb[obs])
    
    # rescale weights such that np.mean(weights) == 1
    return weights/np.mean(weights)


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
    'ds_m_mass>2.16',
    'mu_pt>8',
    'mu_id_medium==1', 
    'mu_rel_iso<0.1',
    'abs(mu_bs_dxy_sig)>5',
    'k1_pt>1.',
    'k2_pt>1.',
    'pi_pt>1.',
    f'abs(phi_mass-{m_phi})<0.01',
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


# Processing
############

# shift ds_mass in Reco data
ds_mass_shift = 0.999
tree_Reco['ds_mass'] = ds_mass_shift*tree_Reco['ds_mass']
tree_Reco_tau['ds_mass'] = ds_mass_shift*tree_Reco_tau['ds_mass']


# %%##########
# Parameters #
##############

# list of observables that are used from the trees for the optimization
keys_obs = ['q2', 'ds_mass']#, 'e_star_mu', 'pt_miss']

# list of signals in the tree
keys_sig = ['-2', '-1', '0', '1', '2', '3']

# labels of the different channels
labels = {'-2': 'comb. bkg.',
          '-1': r'$H_b \rightarrow D_s + \mu$',
          '0': r"$B_s \rightarrow D_s \, \mu \nu$",
          '1': r"$B_s \rightarrow D_s^* \mu \nu$",
          '2': r"$B_s \rightarrow D_s \, \tau \nu$",
          '3': r"$B_s \rightarrow D^*_s \tau \nu$"}

# x labels of the different observables
labels_obs = {'q2': '$Q^2$ [GeV$^2$]',
          'ds_mass': '$m_{D_s}$ [GeV]'}


# total number of events in data
N_tot_CMS = tree_CMS['q2'].size

# number of events per signal in Reco data
N_sig_Reco = {sig: (tree_Reco['sig']==int(sig)).sum() for sig in keys_sig}

# correctons for R, R_star (to include deterctor efficiency, etc.)
R_MC = 0.381
R_star_MC = 0.327
epsilon = R_MC*N_sig_Reco['0']/N_sig_Reco['2']
epsilon_star = R_star_MC*N_sig_Reco['1']/N_sig_Reco['3']

# initial values for the parameters from Reco data and comb bkg
# (scaled to rasonable values)
R_init      = R_MC
R_star_init = R_star_MC
N_0_init    = 5*N_sig_Reco['0']
N_1_init    = 5*N_sig_Reco['1']
N_comb_init = 1.58*tree_comb['q2'].size

# ratio between number of mu* and mu events (to recudce # free params)
N_1_N_0_ratio_Reco = N_sig_Reco['1']/N_sig_Reco['0']

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
        return N_tot_CMS-N_0*(R/epsilon+1)-N_1*(R_star/epsilon_star+1)-N_comb
    # N_1_ = lambda N_0: N_1_N_0_ratio_Reco*N_0
    N_2_ = lambda R, N_0: R/epsilon*N_0
    N_3_ = lambda R_star, N_1: R_star/epsilon_star*N_1
    N_1_N_0_ratio_ = lambda N_0, N_1: N_1/N_0

    # Composed parameters
    # N_1 = zfit.ComposedParameter('N_1', N_1_, [N_0]) # sig 1
    N_m1 = zfit.ComposedParameter('N_m1', N_m1_,
                                  [R, R_star, N_0, N_1, N_comb]) # bkg
    N_2 = zfit.ComposedParameter('N_2', N_2_, [R, N_0]) # sig 2
    N_3 = zfit.ComposedParameter('N_3', N_3_, [R_star, N_1]) # sig 3
    N_1_N_0_ratio = zfit.ComposedParameter('N_1_N_0_ratio', N_1_N_0_ratio_,
                                           [N_0, N_1]) # for the constraint

# parameters used to build the model
params = {'-2': N_comb, '-1': N_m1, '0': N_0, '1': N_1, '2': N_2, '3': N_3}

# set parameter ranges
R.lower, R.upper           = 0, 15
R_star.lower, R_star.upper = 0, 15
N_0.lower, N_0.upper       = 0, 0.5*N_tot_CMS
N_1.lower, N_1.upper       = 0, 0.5*N_tot_CMS
N_comb.lower, N_comb.upper = 0, 0.8*N_tot_CMS

# set initial parameter values
R.set_value(R_init)
R_star.set_value(R_star_init)
N_0.set_value(N_0_init)
N_1.set_value(N_1_init)
N_comb.set_value(N_comb_init)

# set initial stepsizes
R.step_size, R_star.step_size = 1e-3, 1e-3
N_0.step_size = 1
N_1.step_size = 1
N_comb.step_size = 1

# gaussian constraints: 10% uncertainty for ratio N_1/N_0
constraints = [zfit.constraint.GaussianConstraint(
    N_1_N_0_ratio, N_1_N_0_ratio_Reco, 0.1*N_1_N_0_ratio_Reco
    )]


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
bins = {'q2': 15, 'pt_miss': 200, 'e_star_mu': 20, 'ds_mass': 17}

# weights of the combinatorial background
weights_comb = {'q2': None, #weights_for_comb_bkg('q2'),
               'ds_mass': None} #weights_for_comb_bkg('ds_mass')}


# create hists from reco data and additional tau signals
hists = dict()
for obs in keys_obs:
    hists[obs] = dict()
    for sig in keys_sig:
        hists[obs][sig] = hist.Hist(hist.axis.Regular(bins[obs], *ranges[obs],
                                                      name=obs))
        if sig == '-2':
            # weight the combinatorial background
            hists[obs][sig].fill(tree_comb[obs][tree_comb['sig']==int(sig)],
                                 weight=weights_comb[obs])
        if sig in ['-1', '0', '1']:
            hists[obs][sig].fill(tree_Reco[obs][tree_Reco['sig']==int(sig)])
        if sig in ['2', '3']:
            # for the tau signals only use the tau_mc_3may.root tree
            hists[obs][sig].fill(
                tree_Reco_tau[obs][tree_Reco_tau['sig']==int(sig)]
            )        

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
data_zfit = {obs_name: zfit.Data.from_numpy(obs, tree_CMS[obs_name]
                                            ).to_binned(obs)
            for obs_name, obs in all_obs.items()}

# NLL function
NLL = zfit.loss.BinnedNLL(model=histPDFs.values(), data=data_zfit.values(),
                          constraints=constraints)

# plot of the stacked pdf split into signals with initial values
for obs in keys_obs:
    plot_stacked_hist(obs)
    mplhep.histplot(data_zfit[obs].to_hist(), label='Observed', yerr=True,
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


# %%#######
# Results #
###########

# Folder where the results are saved
folder = 'Results_of_NLL_fit/'

# plot of the result split into signals
for obs in keys_obs:
    plot_stacked_hist(obs)
    mplhep.histplot(data_zfit[obs].to_hist(), label='Observed', yerr=True,
                    color='k', histtype='step')
    if obs in plot_ranges:
        plt.xlim(plot_ranges[obs])
    if obs in legend_pos:
        plt.legend(loc=legend_pos[obs])
    else:
        plt.legend()
    #plt.savefig(folder + f'stacked_hist_plot_{obs}.pdf')
    plt.title('Result')
    plt.show()

# plots visualization of ds_mass shift with final results of fit
#plot_ds_mass_shift(nbins=30, save_plot=False)

result.hesse()

# print results
print(result)
print('')
print_params()

R_res = result.values[R]
DeltaR_res = result.hesse()[R]['error']
R_star_res = result.values[R_star]
DeltaR_star_res = result.hesse()[R_star]['error']

# theoretical predictions
R_theo = 0.2971
DeltaR_theo = 0.0034
R_star_theo = 0.2472
DeltaR_star_theo = 0.0077


# ranges for plotting
R_plot_range = (R_res - 1.0*DeltaR_res, R_res + 1.0*DeltaR_res)
R_star_plot_range = (R_star_res - 1.0*DeltaR_star_res,
                     R_star_res + 1.0*DeltaR_star_res)

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
#plt.errorbar(R_theo, DeltaR_theo, R_star_theo, DeltaR_star_theo, 'r',
             #label='Theroetical prediction')
plt.minorticks_on()
plt.grid(True)
#plt.savefig(folder + 'R_R_star_contour.pdf')
plt.show()


#########
# ToDos #
#########
# - Also plot 2 sigma contour? i.e. 2*DeltaNLL = 4 = 2**2




# %%
