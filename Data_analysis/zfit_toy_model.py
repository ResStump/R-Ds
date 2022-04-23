# %%
import numpy as np
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
def plot_profile(param, range, NLL):
    param_result = param.value()
    NLL0 = NLL.value()
    x = np.linspace(float(range[0]), float(range[1]), num=100)
    y = []
    
    for val in x:
        param.set_value(val)
        y.append(2*(NLL.value()-NLL0)) # todo
    
    param.set_value(param_result)
    
    plt.plot(x, y)
    plt.xlabel(param.name)
    plt.ylabel('$2\Delta$NLL')
    return

# %%##########
# Parameters #
##############

# True parameters
N_m1_true, N_0_true, N_1_true = 50000, 10000, 20000
R_true = N_1_true/N_0_true

# total number of events
N_tot = N_m1_true + N_0_true + N_1_true

# Parameters that are optimized
R = zfit.Parameter('R', R_true + 1, 0, 100)
N_m1 = zfit.Parameter('N_m1', N_m1_true+20000, 0, 1e6, step_size=1) #background

# Composed parameters
N_0 = zfit.ComposedParameter(
        'N_0', lambda R, N_m1: (N_tot-N_m1)/(R+1), params=[R, N_m1]
        ) # signal 0
N_1 = zfit.ComposedParameter(
        'N_1', lambda R, N_m1: R*(N_tot-N_m1)/(R+1), params=[R, N_m1]
        ) # signal 1

# parameters used to build the model
params = {'-1': N_m1, '0': N_0, '1': N_1}

# gaussian constraints
R_exp = R_true # expected value
DeltaR = R_true*0.1 # 10% uncertainty for R_exp

N_m1_exp = N_m1_true
DeltaN_m1 = N_m1_true*0.2 # 20% uncertainty for N_m1_exp

constraints = [zfit.constraint.GaussianConstraint(R, R_exp, DeltaR),
               zfit.constraint.GaussianConstraint(N_m1, N_m1_exp, DeltaN_m1)]

# other paremeters
mu_m1, mu_0, mu_1 = 0, -0.5, 1
sigma_m1, sigma_0, sigma_1 = 5, 0.3, 0.7

beta_m1, beta_0, beta_1 = 3, 1, 0.5
shift_m1, shift_0, shift_1 = 0, 0.8, 1.2

range_obs1 = (-10, 10) # range of obs 1
range_obs2 = (0, 10) # range of obs 2
ranges = {'obs1': range_obs1, 'obs2': range_obs2}

# %%############
# create trees #
################

def create_tree():
    # observable 1
    # create more events to cut what is outside of range_obs1
    Data_obs1_m1 = np.random.normal(mu_m1, sigma_m1, 2*N_m1_true)
    Data_obs1_m1 = Data_obs1_m1[
        (Data_obs1_m1 >= range_obs1[0]) & (Data_obs1_m1 <= range_obs1[1]) # cut
        ][:N_m1_true] # reduction of size

    Data_obs1_0 = np.random.normal(mu_0, sigma_0, 2*N_0_true)
    Data_obs1_0 = Data_obs1_0[
        (Data_obs1_0 >= range_obs1[0]) &
        (Data_obs1_0 <= range_obs1[1])
        ][:N_0_true]
    
    Data_obs1_1 = np.random.normal(mu_1, sigma_1, 2*N_1_true)
    Data_obs1_1 = Data_obs1_1[
        (Data_obs1_1 >= range_obs1[0]) &
        (Data_obs1_1 <= range_obs1[1])
        ][:N_1_true]

    # observable 2
    Data_obs2_m1 = np.random.exponential(beta_m1, 2*N_m1_true) + shift_m1
    Data_obs2_m1 = Data_obs2_m1[
        (Data_obs2_m1 >= range_obs2[0]) &
        (Data_obs2_m1 <= range_obs2[1])
        ][:N_m1_true]

    Data_obs2_0 = np.random.exponential(beta_0, 2*N_0_true) + shift_0
    Data_obs2_0 = Data_obs2_0[
        (Data_obs2_0 >= range_obs2[0]) &
        (Data_obs2_0 <= range_obs2[1])
        ][:N_0_true]

    Data_obs2_1 = np.random.exponential(beta_1, 2*N_1_true) + shift_1
    Data_obs2_1 = Data_obs2_1[
        (Data_obs2_1 >= range_obs2[0]) &
        (Data_obs2_1 <= range_obs2[1])
        ][:N_1_true]

    # save data in dict
    tree = dict()
    tree['obs1'] = np.concatenate( (Data_obs1_m1, Data_obs1_0, Data_obs1_1) )
    tree['obs2'] = np.concatenate( (Data_obs2_m1, Data_obs2_0, Data_obs2_1) )
    tree['sig'] = np.concatenate(
        (-np.ones(N_m1_true), np.zeros(N_0_true), np.ones(N_1_true))
        )

    return tree

tree_Reco = create_tree()
tree_Data = create_tree()

# number of Bins
bins = {'obs1': 50, 'obs2': 50}

keys_obs = ['obs1', 'obs2']
keys_sig = ['-1', '0', '1']



#####################################################
# creation of zfit model from Data and minimization #
#####################################################

# create hists from 'gen' data
hists = dict()
for obs in keys_obs:
    hists[obs] = dict()
    for sig in keys_sig:
        hists[obs][sig] = hist.Hist(hist.axis.Regular(bins[obs], *ranges[obs],
                                                      name=obs))
        hists[obs][sig].fill( tree_Reco[obs][ tree_Reco['sig']==int(sig) ] )


# plot of hists
def obs_plot_sig(obs):
    for sig in keys_sig:
        hists[obs][sig].plot()
    plt.show()
    return

"""for obs in keys_obs:
    obs_plot_sig(obs)"""


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

all_obs = {obs:zfit.Space(obs, ranges[obs], binning=binning[obs]) 
           for obs in keys_obs}


# load Reco data into zfit (binned)
data_zfit = {obs_name: zfit.Data.from_numpy(obs, tree_Data[obs_name])
             .to_binned(obs)
             for obs_name, obs in all_obs.items()}



# NLL function
NLL = zfit.loss.BinnedNLL(model=histPDFs.values(), data=data_zfit.values(),
                          constraints=constraints)

# set parameter values back
R.set_value(R_true+1)
N_m1.set_value(N_m1_true+20000)

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

# plot of profile
plot_profile(R, (R.value()-0.05, R.value()+0.05), NLL)
plt.minorticks_on()
plt.grid(True)
plt.show()


#########
# ToDos #
#########



# %%
