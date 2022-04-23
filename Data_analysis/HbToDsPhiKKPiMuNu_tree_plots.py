#%%
import numpy as np
import matplotlib.pyplot as plt
import uproot
import vector

# path where plots are saved
path = "Plots_of_kinematic_variables/HbToDsPhiKKPiMuNu_tree_plots/"

# folders to save plots in
folders = {'without_filtering': path + "without_filtering/",
           'with_filtering': path + "with_filtering/",
           'K_pi_ambiguity': path + "K_pi_ambiguity/"}

# open root file
root_file = uproot.open("ROOT_data/HbToDsPhiKKPiMuNu_tree.root")
tree = root_file['tree'].arrays(root_file['tree'].keys(), library='np')

# labelling of the different signals in sig
# -1: background
str_sig_m1 = 'background'
# 0: Ds  mu nu
str_sig_0 = r"$D_s \, \mu \nu$ channel"
# 1: Ds* mu nu
str_sig_1 = r"$D_s^* \mu \nu$ channel"
# 2: Ds  tau nu
str_sig_2 = r"$D_s \, \tau \nu$ channel"
# 3: Ds* tau nu
str_sig_3 = r"$D^*_s \tau \nu$ channel"

labels = {'-1':str_sig_m1,
          '0':str_sig_0, '1':str_sig_1, '2':str_sig_2, '3':str_sig_3}


# number of events (per signal)
N_tot = tree['q2'].size
N_m1 = np.sum(tree['sig'] == -1)    # background
N_0 = np.sum(tree['sig'] == 0)      # Ds  mu nu
N_1 = np.sum(tree['sig'] == 1)      # Ds* mu nu
N_2 = np.sum(tree['sig'] == 2)      # Ds  tau nu
N_3 = np.sum(tree['sig'] == 3)      # Ds* tau nu


# list of elements containing nans
list_of_nans = []
for k, leaf in tree.items():
    if np.isnan(leaf).any():
        list_of_nans += [k]




#############
# Constants #
#############

m_Bs = 5.36688 # [GeV]
m_k = 0.493677 # [GeV]
m_pi = 0.13957039 # [GeV]




#################################
# sorting/processing of momenta #
#################################

# making new entries in tree in which kp and km are sorted into k_big and k_small
#################################################################################
bool_kp_big = tree['kp_e'] >= tree['km_e']

keys_kp = ['kp_pt', 'kp_eta', 'kp_phi', 'kp_e', 'kp_charge']
keys_km = ['km_pt', 'km_eta', 'km_phi', 'km_e', 'km_charge']
keys_big = ['k_big_pt', 'k_big_eta', 'k_big_phi', 'k_big_e', 'k_big_charge']
keys_small = ['k_small_pt', 'k_small_eta',
              'k_small_phi', 'k_small_e', 'k_small_charge']
         
# insertint the k_big   
for k_kp, k_km, k_big, k_small in zip(keys_kp, keys_km, keys_big, keys_small):
    tree[k_big] = bool_kp_big*tree[k_kp] \
                  + np.logical_not(bool_kp_big)*tree[k_km]

# insertint the k_small
for k_kp, k_km, k_big, k_small in zip(keys_kp, keys_km, keys_big, keys_small):
    tree[k_small] = np.logical_not(bool_kp_big)*tree[k_kp] \
                    + bool_kp_big*tree[k_km]


# converting momentum in tree into 4-momenta
############################################
keys_p4 = ['mu', 'kp', 'km', 'k_big', 'k_small', 'pi']

p4 = dict()
for k_p4 in keys_p4:
    p4[k_p4] = vector.array({'pt': tree[k_p4+'_pt'],
                             "phi": tree[k_p4+'_phi'],
                             "eta": tree[k_p4+'_eta'],
                             "E": tree[k_p4+'_e']})

# 4-momentum of the K which has the same charge of pi (opposite charge as mu)
p4['k_same'] = (tree['mu_charge']==-1.).astype(int)*p4['kp'] \
                + (tree['mu_charge']==1.).astype(int)*p4['km']

# 4-momentum of the K which has the opposite charge of pi (same charge as mu)
p4['k_opposite'] = (tree['mu_charge']==1.).astype(int)*p4['kp'] \
                   + (tree['mu_charge']==-1.).astype(int)*p4['km']

# 4-momentum of pi with K_same mass (i.e. changing its energy)
p4['pi_m_exch'] = p4['pi']*1
p4['pi_m_exch']['E'] = np.sqrt(p4['pi'].p**2 + m_k**2)


keys_p4 = list(p4.keys())
keys_tree = list(tree.keys())




#############
# Filtering #
#############

# index for which bool_arr != True are set to nan in tree
def tree_filtering(bool_arr):
    bool_arr = np.logical_not(bool_arr) # inverting the condition
    for k in tree.keys():
        if k == 'sig': # don't apply filter to 'sig'
            continue
        tree[k][bool_arr] = np.nan
        #tree[k][np.logical_and( tree['sig'] == -1, bool_arr)] = np.nan
    return

# index for which bool_arr != True are set to nan in p4
def p4_filtering(bool_arr):
    bool_arr = np.logical_not(bool_arr) # inverting the condition
    for k in p4.keys():
        if k == 'sig': # don't apply filter to 'sig'
            continue
        for index in ['pt', "phi", "eta", "E"]:
            p4[k][index][bool_arr] = np.nan
            #p4[k][index][np.logical_and( tree['sig'] == -1, bool_arr)] = np.nan
    return


filters = dict()

# cut for B_s mass at 5.36688 GeV (PDG, Fit)
filters['ds_m_mass'] = tree['ds_m_mass'] <= m_Bs

"""# cut for pt of D_s mu system (not applied)
filters['ds_m_pt'] = tree['ds_m_pt'] >= 7.5


# cut for Q^2  (not applied)
filters['q2'] = tree['q2'] >= -0.264


# cut for e_ster_mu3  (not applied)
filters['e_star_mu3'] = tree['e_star_mu3'] <= 2.31


# cut for scalar pt_miss  (not applied)
scalar_pt_miss = tree['ds_m_pt']*m_Bs/tree['ds_m_mass'] \
          - tree['km_pt']- tree['kp_pt'] - tree['pi_pt']- tree['mu_pt']
filters['scalar_pt_miss'] = scalar_pt_miss >= -1.12


# cut for deltaR(mu, k_big)
filters['dR_mu_k_big'] = p4['mu'].deltaR(p4['k_big']) <= 1.5

# cut for deltaR(mu, k_small)
filters['dR_mu_k_small'] = p4['mu'].deltaR(p4['k_small']) <= 1.5

# cut for deltaR(mu, pi)
filters['dR_mu_pi']= p4['mu'].deltaR(p4['pi']) <= 2."""




def filterAll():
    for bool_arr in filters.values():
        tree_filtering(bool_arr)
        p4_filtering(bool_arr)
    return

filterAll()




########################
# Plots of histogramms #
########################

def plotHist(data, label, xlabel, filename, bins=None, xlim=None, density=True,
             show=True, folder=None, alpha=None):
    plt.hist(data, bins=bins, density=density, label=label,
             alpha=alpha, histtype='stepfilled')
    if xlim:
        plt.xlim(xlim)
    if label:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("a.u.")
    if folder:
        plt.savefig(folder + filename)
    if show:
        plt.show()
    elif show==None:
        pass
    else:
        plt.close()
    return


alpha_std = {'-1':0.4, '0':0.5, '1':0.6, '2':0.8, '3':1}
order_std = ['3', '2', '1', '0', '-1']
bins_std = {'-1':None, '0':None, '1':None, '2':None, '3':None}

def plotHist_sig(data, xlabel, filename, bins=bins_std, xlim=None, show=True,
                 folder=None, order=order_std, alpha=alpha_std):
    for k in order:
        plt.hist(data[ tree['sig']==int(k) ], bins=bins[k], density=True,
                 alpha=alpha[k], label=labels[k], histtype='stepfilled')
    if xlim:
        plt.xlim(xlim)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("a.u.")
    if folder:
        plt.savefig(folder + filename)
    if show:
        plt.show()
    elif show==None:
        pass
    else:
        plt.close()
    return




# K, pi ambiguity
################

# plot of deltaR between K_same and pi
"""def dR_pi_k_same_plots(show=True, folder=None):
    xlabel = r"$\Delta R(\pi, K_{\mathrm{same}})$"
    filename = "dR_pi_k_same.pdf"
    bins = {'-1':50, '0':50, '1':50, '2':20, '3':20}
    plotHist_sig(p4['k_same'].deltaR(p4['pi']), xlabel, filename, bins=bins,
                 show=show, folder=folder)
    return"""


# plot of phi mass
def phi_mass_plot(show=True, folder=None):
    phi_p4 = p4['k_same'] + p4['k_opposite']
    xlabel = r"$m_\phi$ [GeV]"
    filename = "phi_mass.pdf"
    plotHist(phi_p4.m, None, xlabel, filename, bins=200, xlim=[0.98, 1.085],
             show=show, folder=folder)
    return    


# plot of phi mass split into different signals
def phi_mass_split_sig_plots(show=True, folder=None):
    phi_p4 = p4['k_same'] + p4['k_opposite']
    xlabel = r"$m_\phi$ [GeV]"
    filename = "phi_mass_split_sig.pdf"
    bins = {'-1':100, '0':100, '1':100, '2':30, '3':30}
    plotHist_sig(phi_p4.m, xlabel, filename, bins=bins, xlim=[0.98, 1.085],
                 show=show, folder=folder)
    return


# plot of phi mass when exchanging pi and k_same mass
def phi_mass_k_same_pi_mass_exch_plot(show=True, folder=None):
    phi_p4_m_exch = p4['pi_m_exch'] + p4['k_opposite']
    xlabel = r"$m_\phi$ for $K_{\mathrm{opposite}}$ and " \
             + r"$\pi$ with $K_{\mathrm{same}}$ mass [GeV]"
    filename = "phi_mass_k_same_pi_mass_exch.pdf"
    plotHist(phi_p4_m_exch.m, None, xlabel, filename, bins=400, xlim=[1, 2.2],
             show=show, folder=folder)
    return


# plot of phi mass when exchanging pi and k_same mass, split into different signals
def phi_mass_k_same_pi_mass_exch_split_sig_plots(show=True, folder=None):
    phi_p4_m_exch = p4['pi_m_exch'] + p4['k_opposite']
    xlabel = r"$m_\phi$ for $K_{\mathrm{opposite}}$ and $\pi$ "\
             + r"with $K_{\mathrm{same}}$ mass [GeV]"
    filename = "phi_mass_k_same_pi_mass_exch_split_sig.pdf"
    bins = {'-1':200, '0':200, '1':200, '2':30, '3':30}
    plotHist_sig(phi_p4_m_exch.m, xlabel, filename, bins=bins, xlim=[1, 2.2],
                 show=show, folder=folder)
    return


# plot of phi mass when exchanging pi and k_same mass, split into K charge
def phi_mass_k_same_pi_mass_exch_split_charge_plots(show=True, folder=None):
    phi_p4_m_exch = p4['pi_m_exch'] + p4['k_opposite']
    xlabel = r"$m_\phi$ for $K_{\mathrm{opposite}}$ and $\pi$ " \
             + r"with $K_{\mathrm{same}}$ mass [GeV]"
    filename = "phi_mass_k_same_pi_mass_exch_split_charge.pdf"
    
    label = '$\pi^-$ with $m_{K}}}$ and $K^+$'
    plotHist(phi_p4_m_exch.m[tree['mu_charge'] == -1], label, xlabel, filename,
             bins=200, xlim=None, show=None, folder=None, alpha=1)
    
    label = '$\pi^+$ with $m_{K}}}$ and $K^-$'
    plotHist(phi_p4_m_exch.m[tree['mu_charge'] == +1], label, xlabel, filename,
             bins=200, xlim=[1, 2.2], show=show, folder=folder, alpha=0.7)
    return


# plot of phi mass for both variants
def phi_mass_both_variant_plot(show=True, folder=None):
    xlabel = r"$m_\phi$ [GeV]"
    filename = "phi_mass_both_variant.pdf"
    plt.ylim([0, 6])
    
    phi_p4 = p4['k_same'] + p4['k_opposite']
    label = '$K_{\mathrm{same}}$ and $K_{\mathrm{opposite}}$'
    plotHist(phi_p4.m, label, xlabel, filename, bins=30, xlim=None,
             density=True, show=None, folder=None)
    
    phi_p4_m_exch = p4['pi_m_exch'] + p4['k_opposite']
    label = '$\pi$ with $m_{K}}$ and $K_{\mathrm{opposite}}$'
    plotHist(phi_p4_m_exch.m, label, xlabel, filename, bins=400,
             xlim=[0.98, 2], density=True, show=show, folder=folder, alpha=None)    
    return


# plot of phi mass for both variants in log scale
def phi_mass_both_variant_logPlot(show=True, folder=None):
    xlabel = r"$m_\phi$ [GeV]"
    filename = "phi_mass_both_variant_log.pdf"
    plt.yscale('log')
    
    phi_p4 = p4['k_same'] + p4['k_opposite']
    label = '$K_{\mathrm{same}}$ and $K_{\mathrm{opposite}}$'
    plotHist(phi_p4.m, label, xlabel, filename, bins=30, xlim=None,
             density=True, show=None, folder=None)
    
    phi_p4_m_exch = p4['pi_m_exch'] + p4['k_opposite']
    label = '$\pi$ with $m_{K_{\mathrm{same}}}$ and $K_{\mathrm{opposite}}$'
    plotHist(phi_p4_m_exch.m, label, xlabel, filename, bins=400,
             xlim=[0.98, 2], density=True, show=show, folder=folder, alpha=0.8)    
    return


def plotAllHist1(show=False, folder=folders['K_pi_ambiguity']):
    phi_mass_plot(show=show, folder=folder)
    phi_mass_split_sig_plots(show=show, folder=folder)
    phi_mass_k_same_pi_mass_exch_plot(show=show, folder=folder)
    phi_mass_k_same_pi_mass_exch_split_sig_plots(show=show, folder=folder)
    phi_mass_both_variant_plot(show=show, folder=folder)
    phi_mass_both_variant_logPlot(show=show, folder=folder)
    return




# plot of other kinematic variables
###################################


# plot of mass, pt, eta, phi of ds_m
def ds_m_plots(show=True, folder=None):
    xlabel = r"Mass of $D_s\,\mu$ system [GeV]"
    filename = "ds_m_mass.pdf"
    bins = {'-1':30, '0':30, '1':30, '2':15, '3':15}
    plotHist_sig(tree['ds_m_mass'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    
    # cutoff for higher p_T for better readability
    xlabel = r"$p_T$ of $D_s\,\mu$ system [GeV]"
    filename = "ds_m_pt.pdf"
    bins = {'-1':300, '0':300, '1':300, '2':50, '3':50}
    plotHist_sig(tree['ds_m_pt'], xlabel, filename, bins=bins, xlim=[0, 70],
                 show=show, folder=folder)
     
    xlabel = r"$\eta$ of $D_s\,\mu$ system"
    filename = "ds_m_eta.pdf"
    bins = {'-1':30, '0':30, '1':30, '2':15, '3':15}
    plotHist_sig(tree['ds_m_eta'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    
    xlabel = r"$\phi$ of $D_s\,\mu$ system"
    filename = "ds_m_phi.pdf"
    bins = {'-1':30, '0':30, '1':30, '2':15, '3':15}
    plotHist_sig(tree['ds_m_phi'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    return
# plot of mass, pt, eta, phi of ds_m, with log scale
def ds_m_pt_logPlots(show=True, folder=None):
    # cutoff for higher p_T for better readability
    xlabel = r"$p_T$ of $D_s\,\mu$ system [GeV]"
    filename = "ds_m_pt_log.pdf"
    bins = {'-1':300, '0':300, '1':300, '2':50, '3':50}
    plt.yscale('log')
    plt.minorticks_on()
    plotHist_sig(tree['ds_m_pt'], xlabel, filename, bins=bins, xlim=[0, 70],
                 show=show, folder=folder)
    return




# plot of q2
def q2_plot(show=True, folder=None):
    bins = {'-1':30, '0':20, '1':20, '2':10, '3':10}
    xlabel = r"$Q^2$ [GeV$^2$]"
    filename = "q2.pdf"
    plotHist_sig(tree['q2'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    return
# plot of q2, with log scale
def q2_logPlot(show=True, folder=None):
    bins = {'-1':30, '0':20, '1':20, '2':10, '3':10}
    xlabel = r"$Q^2$ [GeV$^2$]"
    filename = "q2_log.pdf"
    plt.yscale('log')
    plt.minorticks_on()
    plotHist_sig(tree['q2'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    return


# plot of m2_miss
def m2_miss_plot(show=True, folder=None):
    bins = {'-1':50, '0':50, '1':50, '2':15, '3':15}
    xlabel = r"$m^2_{miss}$ [GeV$^2$]"
    filename = "m2_miss.pdf"
    plotHist_sig(tree['m2_miss'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    return


# plot of energy of unpaired mu
def e_star_mu3_plot(show=True, folder=None):
    bins = {'-1':50, '0':50, '1':50, '2':15, '3':15}
    xlabel = r"$E^*_\mu$ [GeV]"
    filename = "e_star_mu3.pdf"
    plotHist_sig(tree['e_star_mu3'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    return
# plot of energy of unpaired mu, with log scale
def e_star_mu3_logPlot(show=True, folder=None):
    bins = {'-1':50, '0':50, '1':50, '2':15, '3':15}
    xlabel = r"$E^*_\mu$ [GeV]"
    filename = "e_star_mu3_log.pdf"
    plt.yscale('log')
    plt.minorticks_on()
    plotHist_sig(tree['e_star_mu3'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    return




# plot of missing transversal momentum

# cutoff for higher p_T for better readability
# neglecting direction, photon in ds_st signal not taken into account
def pt_miss_scalar_plot(show=True, folder=None):
    pt_miss = tree['ds_m_pt']*m_Bs/tree['ds_m_mass'] \
              - tree['km_pt']- tree['kp_pt'] - tree['pi_pt']- tree['mu_pt']
    bins = {'-1':150, '0':150, '1':200, '2':50, '3':50}
    xlabel = r"scalar $p_T^{miss}$ [GeV]"
    filename = "pt_miss_scalar.pdf"
    plotHist_sig(pt_miss, xlabel, filename, bins=bins, xlim=[-10, 60],
                 show=show, folder=folder)
    return
# plot of missing transversal momentum, with log scale
def pt_miss_scalar_logPlot(show=True, folder=None):
    pt_miss = tree['ds_m_pt']*m_Bs/tree['ds_m_mass'] \
              - tree['km_pt']- tree['kp_pt'] - tree['pi_pt']- tree['mu_pt']
    bins = {'-1':150, '0':150, '1':200, '2':50, '3':50}
    xlabel = r"scalar $p_T^{miss}$ [GeV]"
    filename = "pt_miss_scalar_log.pdf"
    plt.yscale('log')
    plt.minorticks_on()
    plotHist_sig(pt_miss, xlabel, filename, bins=bins, xlim=[-28, 60],
                 show=show, folder=folder)
    return




# plot of mass, pt, eta, phi of mu
def mu_plots(show=True, folder=None):    
    # cutoff for higher p_T for better readability
    xlabel = r"$p_T^\mu$ [GeV]"
    filename = "mu_pt.pdf"
    bins = {'-1':200, '0':200, '1':200, '2':30, '3':30}
    plotHist_sig(tree['mu_pt'], xlabel, filename, bins=bins, xlim=[6, 35],
                 show=show, folder=folder)
     
    xlabel = r"$\eta_\mu$"
    filename = "mu_eta.pdf"
    bins = {'-1':30, '0':30, '1':30, '2':15, '3':15}
    plotHist_sig(tree['mu_eta'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    
    xlabel = r"$\phi_\mu$"
    filename = "mu_phi.pdf"
    bins = {'-1':30, '0':30, '1':30, '2':15, '3':15}
    plotHist_sig(tree['mu_phi'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    
    # cutoff for higher mu_e for better readability
    xlabel = r"$E_\mu$ [GeV]"
    filename = "mu_e.pdf"
    bins = {'-1':200, '0':200, '1':200, '2':40, '3':40}
    plotHist_sig(tree['mu_e'], xlabel, filename, bins=bins, xlim=[6, 50],
                 show=show, folder=folder)
    return


# plot of mass, pt, eta, phi of kp
def kp_plots(show=True, folder=None):    
    # cutoff for higher p_T for better readability
    xlabel = r"$p_T^{K^+}$ [GeV]"
    filename = "kp_pt.pdf"
    bins = {'-1':200, '0':200, '1':200, '2':40, '3':40}
    plotHist_sig(tree['kp_pt'], xlabel, filename, bins=bins, xlim=[-0.1, 20],
                 show=show, folder=folder)
     
    xlabel = r"$\eta_{K^+}$"
    filename = "kp_eta.pdf"
    bins = {'-1':30, '0':30, '1':30, '2':10, '3':10}
    plotHist_sig(tree['kp_eta'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    
    xlabel = r"$\phi_{K^+}$"
    filename = "kp_phi.pdf"
    bins = {'-1':30, '0':30, '1':30, '2':15, '3':15}
    plotHist_sig(tree['kp_phi'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    
    # cutoff for higher km_e for better readability
    xlabel = r"$E_{K^+}$ [GeV]"
    filename = "kp_e.pdf"
    bins = {'-1':200, '0':200, '1':200, '2':40, '3':40}
    plotHist_sig(tree['kp_e'], xlabel, filename, bins=bins, xlim=[0, 25],
                 show=show, folder=folder)
    return


# plot of mass, pt, eta, phi of km
def km_plots(show=True, folder=None):    
    # cutoff for higher p_T for better readability
    xlabel = r"$p_T^{K^-}$ [GeV]"
    filename = "km_pt.pdf"
    bins = {'-1':300, '0':300, '1':300, '2':40, '3':40}
    plotHist_sig(tree['km_pt'], xlabel, filename, bins=bins, xlim=[-0.1, 20],
                 show=show, folder=folder)
     
    xlabel = r"$\eta_{K^-}$"
    filename = "km_eta.pdf"
    bins = {'-1':30, '0':30, '1':30, '2':10, '3':10}
    plotHist_sig(tree['km_eta'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    
    xlabel = r"$\phi_{K^-}$"
    filename = "km_phi.pdf"
    bins = {'-1':30, '0':30, '1':30, '2':15, '3':15}
    plotHist_sig(tree['km_phi'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    
    # cutoff for higher km_e for better readability
    xlabel = r"$E_{K^-}$ [GeV]"
    filename = "km_e.pdf"
    bins = {'-1':200, '0':200, '1':200, '2':40, '3':40}
    plotHist_sig(tree['km_e'], xlabel, filename, bins=bins, xlim=[0, 25],
                 show=show, folder=folder)
    return

# plot of mass, pt, eta, phi of k_big
def k_big_plots(show=True, folder=None):    
    # cutoff for higher p_T for better readability
    xlabel = r"$p_T^{K_\mathrm{big}}$ [GeV]"
    filename = "k_big_pt.pdf"
    bins = {'-1':300, '0':300, '1':300, '2':40, '3':40}
    plotHist_sig(tree['k_big_pt'], xlabel, filename, bins=bins,
                 xlim=[-0.1, 20], show=show, folder=folder)
     
    xlabel = r"$\eta_{K_\mathrm{big}}$"
    filename = "k_big_eta.pdf"
    bins = {'-1':30, '0':30, '1':30, '2':10, '3':10}
    plotHist_sig(tree['k_big_eta'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    
    xlabel = r"$\phi_{K_\mathrm{big}}$"
    filename = "k_big_phi.pdf"
    bins = {'-1':30, '0':30, '1':30, '2':15, '3':15}
    plotHist_sig(tree['k_big_phi'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    
    # cutoff for higher km_e for better readability
    xlabel = r"$E_{K_\mathrm{big}}$ [GeV]"
    filename = "k_big_e.pdf"
    bins = {'-1':300, '0':300, '1':300, '2':40, '3':40}
    plotHist_sig(tree['k_big_e'], xlabel, filename, bins=bins, xlim=[0, 25],
                 show=show, folder=folder)
    return


# plot of mass, pt, eta, phi of k_small
def k_small_plots(show=True, folder=None):    
    # cutoff for higher p_T for better readability
    xlabel = r"$p_T^{K_\mathrm{small}}$ [GeV]"
    filename = "k_small_pt.pdf"
    bins = {'-1':200, '0':200, '1':200, '2':40, '3':40}
    plotHist_sig(tree['k_small_pt'], xlabel, filename, bins=bins,
                 xlim=[-0.1, 20], show=show, folder=folder)
     
    xlabel = r"$\eta_{K_\mathrm{small}}$"
    filename = "k_small_eta.pdf"
    bins = {'-1':30, '0':30, '1':30, '2':10, '3':10}
    plotHist_sig(tree['k_small_eta'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    
    xlabel = r"$\phi_{K_\mathrm{small}}$"
    filename = "k_small_phi.pdf"
    bins = {'-1':30, '0':30, '1':30, '2':15, '3':15}
    plotHist_sig(tree['k_small_phi'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    
    # cutoff for higher k_small_e for better readability
    xlabel = r"$E_{K_\mathrm{small}}$ [GeV]"
    filename = "k_small_e.pdf"
    bins = {'-1':200, '0':200, '1':200, '2':40, '3':40}
    plotHist_sig(tree['k_small_e'], xlabel, filename, bins=bins, xlim=[0, 25],
                 show=show, folder=folder)
    return


# plot of mass, pt, eta, phi of pi
def pi_plots(show=True, folder=None):
    # cutoff for higher p_T for better readability
    xlabel = r"$p_T^\pi$ [GeV]"
    filename = "pi_pt.pdf"
    bins = {'-1':200, '0':200, '1':200, '2':40, '3':40}
    plotHist_sig(tree['pi_pt'], xlabel, filename, bins=bins, xlim=[0, 20],
                 show=show, folder=folder)
     
    xlabel = r"$\eta_\pi$"
    filename = "pi_eta.pdf"
    bins = {'-1':30, '0':30, '1':30, '2':10, '3':10}
    plotHist_sig(tree['pi_eta'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    
    xlabel = r"$\phi_\pi$"
    filename = "pi_phi.pdf"
    bins = {'-1':30, '0':30, '1':30, '2':15, '3':15}
    plotHist_sig(tree['pi_phi'], xlabel, filename, bins=bins, show=show,
                 folder=folder)
    
    # cutoff for higher pi_e for better readability
    xlabel = r"$E_\pi$ [GeV]"
    filename = "pi_e.pdf"
    bins = {'-1':200, '0':200, '1':200, '2':40, '3':40}
    plotHist_sig(tree['pi_e'], xlabel, filename, bins=bins, xlim=[0, 30],
                 show=show, folder=folder)
    return




# plot of dR of kp mu
def dR_mu_kp_plot(show=True, folder=None):
    dR = p4['mu'].deltaR(p4['kp'])
    xlabel = r"$\Delta R(\mu, K^+)$"
    filename = "dR_mu_kp.pdf"
    bins = {'-1':100, '0':100, '1':100, '2':15, '3':15}
    plotHist_sig(dR, xlabel, filename, bins=bins, xlim=[0, 2.5], show=show,
                 folder=folder)
    return


# plot of dR of km mu
def dR_mu_km_plot(show=True, folder=None):
    dR = p4['mu'].deltaR(p4['km'])
    xlabel = r"$\Delta R(\mu, K^-)$"
    filename = "dR_mu_km.pdf"
    bins = {'-1':100, '0':100, '1':100, '2':15, '3':15}
    plotHist_sig(dR, xlabel, filename, bins=bins, xlim=[0, 2.5], show=show,
                 folder=folder)
    return


# plot of dR of k_big mu
def dR_mu_k_big_plot(show=True, folder=None):
    dR = p4['mu'].deltaR(p4['k_big'])
    xlabel = r"$\Delta R(\mu, K_\mathrm{big})$"
    filename = "dR_mu_k_big.pdf"
    bins = {'-1':100, '0':100, '1':100, '2':15, '3':15}
    plotHist_sig(dR, xlabel, filename, bins=bins, xlim=[0, 2.5], show=show,
                 folder=folder)
    return

# plot of dR of k_big mu, with log scale
def dR_mu_k_big_logPlot(show=True, folder=None):
    dR = p4['mu'].deltaR(p4['k_big'])
    xlabel = r"$\Delta R(\mu, K_\mathrm{big})$"
    filename = "dR_mu_k_big_log.pdf"
    bins = {'-1':100, '0':100, '1':100, '2':15, '3':15}
    plt.yscale('log')
    plt.minorticks_on()
    plotHist_sig(dR, xlabel, filename, bins=bins, xlim=[0, 2.5], show=show,
                 folder=folder)
    return


# plot of dR of k_small mu
def dR_mu_k_small_plot(show=True, folder=None):
    dR = p4['mu'].deltaR(p4['k_small'])
    xlabel = r"$\Delta R(\mu, K_\mathrm{small})$"
    filename = "dR_mu_k_small.pdf"
    bins = {'-1':100, '0':100, '1':100, '2':15, '3':15}
    plotHist_sig(dR, xlabel, filename, bins=bins, xlim=[0, 2.5], show=show,
                 folder=folder)
    return
# plot of dR of k_small mu, with log scale
def dR_mu_k_small_logPlot(show=True, folder=None):
    dR = p4['mu'].deltaR(p4['k_small'])
    xlabel = r"$\Delta R(\mu, K_\mathrm{small})$"
    filename = "dR_mu_k_small_log.pdf"
    bins = {'-1':100, '0':100, '1':100, '2':15, '3':15}
    plt.yscale('log')
    plt.minorticks_on()
    plotHist_sig(dR, xlabel, filename, bins=bins, xlim=[0, 2.5], show=show,
                 folder=folder)
    return


# plot of dR of pi mu
def dR_mu_pi_plot(show=True, folder=None):
    dR = p4['mu'].deltaR(p4['pi'])
    xlabel = r"$\Delta R(\mu, \pi)$"
    filename = "dR_mu_pi.pdf"
    bins = {'-1':80, '0':80, '1':80, '2':15, '3':15}
    plotHist_sig(dR, xlabel, filename, bins=bins, xlim=[0, 3.3], show=show,
                 folder=folder)
    return
# plot of dR of pi mu, with log scale
def dR_mu_pi_logPlot(show=True, folder=None):
    dR = p4['mu'].deltaR(p4['pi'])
    xlabel = r"$\Delta R(\mu, \pi)$"
    filename = "dR_mu_pi_log.pdf"
    bins = {'-1':80, '0':80, '1':80, '2':15, '3':15}
    plt.yscale('log')
    plt.minorticks_on()
    plotHist_sig(dR, xlabel, filename, bins=bins, xlim=[0, 3.3], show=show,
                 folder=folder)
    return




def plotAllHist2(show=False, folder=folders['with_filtering']):
    ds_m_plots(show=show, folder=folder)
    #ds_m_pt_logPlots(show=show, folder=folder)
    q2_plot(show=show, folder=folder)
    #q2_logPlot(show=show, folder=folder)
    m2_miss_plot(show=show, folder=folder)
    e_star_mu3_plot(show=show, folder=folder)
    #e_star_mu3_logPlot(show=show, folder=folder)
    pt_miss_scalar_plot(show=show, folder=folder)
    #pt_miss_scalar_logPlot(show=show, folder=folder)
    mu_plots(show=show, folder=folder)
    kp_plots(show=show, folder=folder)
    km_plots(show=show, folder=folder)
    k_big_plots(show=show, folder=folder)
    k_small_plots(show=show, folder=folder)
    pi_plots(show=show, folder=folder)
    dR_mu_kp_plot(show=show, folder=folder)
    dR_mu_km_plot(show=show, folder=folder)
    dR_mu_k_big_plot(show=show, folder=folder)
    #dR_mu_k_big_logPlot(show=show, folder=folder)
    dR_mu_k_small_plot(show=show, folder=folder)
    #dR_mu_k_small_logPlot(show=show, folder=folder)
    dR_mu_pi_plot(show=show, folder=folder)
    #dR_mu_pi_logPlot(show=show, folder=folder)
    return






















# %%
