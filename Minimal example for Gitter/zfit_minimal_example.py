import numpy as np
import hist
import zfit

##############
# Parameters #
##############

# True parameters
N_0_true, N_1_true = 10000, 20000
R_true = N_1_true/N_0_true

# total number of events
N_tot = N_0_true + N_1_true

# Parameter that is optimized with some initial value
R = zfit.Parameter('R', R_true + 1, 0, 100)

# Composed parameters
N_0 = zfit.ComposedParameter('N_0', lambda R: N_tot/(R+1), params=[R])
N_1 = zfit.ComposedParameter('N_1', lambda R: R*N_tot/(R+1), params=[R])


###############
# create data #
###############

# create normally distributed data in the range [-10, 10] for observable 1
Data_obs1_0 = np.random.normal(-2, 0.5, 2*N_0_true)
Data_obs1_0 = Data_obs1_0[(Data_obs1_0 >= -10) 
                          & (Data_obs1_0 <= 10)][:N_0_true]

Data_obs1_1 = np.random.normal(1, 1, 2*N_1_true)
Data_obs1_1 = Data_obs1_1[(Data_obs1_1 >= -10)
                          & (Data_obs1_1 <= 10)][:N_1_true]

# create exponentially distributed data in the range [0, 10] for observable 2
Data_obs2_0 = np.random.exponential(1, 2*N_0_true)
Data_obs2_0 = Data_obs2_0[(Data_obs2_0 >= 0) & (Data_obs2_0 <= 10)][:N_0_true]

Data_obs2_1 = np.random.exponential(0.5, 2*N_1_true) + 1
Data_obs2_1 = Data_obs2_1[(Data_obs2_1 >= 0) & (Data_obs2_1 <= 10)][:N_1_true]


#####################################################
# creation of zfit model from data and minimization #
#####################################################

# number of bins (the code only works if they are the same)
bins_obs1, bins_obs2 = 50, 40

# create histograms from data
hist_obs1_0 = hist.Hist(hist.axis.Regular(bins_obs1, -10, 10, name='obs1'))
hist_obs1_0.fill(Data_obs1_0)
hist_obs1_1 = hist.Hist(hist.axis.Regular(bins_obs1, -10, 10, name='obs1'))
hist_obs1_1.fill(Data_obs1_1)

hist_obs2_0 = hist.Hist(hist.axis.Regular(bins_obs2, 0, 10, name='obs2'))
hist_obs2_0.fill(Data_obs2_0)
hist_obs2_1 = hist.Hist(hist.axis.Regular(bins_obs2, 0, 10, name='obs2'))
hist_obs2_1.fill(Data_obs2_1)


# create for every observable a pdf using HistogramPDF and BinnedSumPDF
histPDF_obs1 = zfit.pdf.BinnedSumPDF(
    [zfit.pdf.HistogramPDF(hist_obs1_0), zfit.pdf.HistogramPDF(hist_obs1_1)],
    fracs=[N_0, N_1]
)
histPDF_obs2 = zfit.pdf.BinnedSumPDF(
    [zfit.pdf.HistogramPDF(hist_obs2_0), zfit.pdf.HistogramPDF(hist_obs2_1)],
    fracs=[N_0, N_1]
)

# create observables in zfit
binning_obs1 = zfit.binned.RegularBinning(bins_obs1, -10, 10, name='obs1')
binning_obs2 = zfit.binned.RegularBinning(bins_obs2, 0, 10, name='obs2')
obs1 = zfit.Space('obs1', (-10, 10), binning=binning_obs1)
obs2 = zfit.Space('obs2', (0, 10), binning=binning_obs2)

# load data into zfit
# (in this case the same data that was used to create the pdfs)
data_zfit_obs1 = zfit.Data.from_numpy(
    obs1,
    np.concatenate((Data_obs1_0, Data_obs1_1))
).to_binned(obs1)
data_zfit_obs2 = zfit.Data.from_numpy(
    obs2,
    np.concatenate((Data_obs2_0, Data_obs2_1))
).to_binned(obs2)

# NLL function
NLL = zfit.loss.BinnedNLL(model=[histPDF_obs1, histPDF_obs2],
                          data=[data_zfit_obs1, data_zfit_obs2])

# miminization
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(NLL)
result.hesse()

print(result)