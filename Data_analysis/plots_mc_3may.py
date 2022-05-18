# %%
import numpy as np
import matplotlib.pyplot as plt
import uproot
import vector

# path where plots are saved
path = "Plots_of_kinematic_variables/mc_3may_plots/"

# folders in that path
folders = {'Reco_plus': path + 'tree_Reco_plus/'}

#########################################
# Data import, processing and filtering #
#########################################

# constants
m_Bs = 5.36688 # [GeV]
m_k = 0.493677 # [GeV]
m_pi = 0.13957039 # [GeV]

# filtering
###########

# cuts to be applied
"""cuts = f"(ds_m_mass<={m_Bs}) & (abs(mu_bs_dxy_sig)>5) & (mu_pt>8) & (HLT_Mu7_IP4==1) & (mu_rel_iso<.2) & (mu_charge*pi_charge<0) & (ds_vtx_prob>0.1) & (cos3D_ds_m>0.995) & (phi_vtx_prob>0.1) & (abs(phi_mass-1.020)<0.01)  & (lxyz_ds_m_sig>10)" """
#"(abs(ds_mass-1.98)<10.015 & abs(phi_mass-1.020)<0.006 & mu_id_medium & ds_m_mass<5.37 & HLT_Mu7_IP4) & mu_iso<0.2 & cos3D_ds_m>0.995 & pi_charge != mu_charge & ds_vtx_prob>0.5")
"""selection = ') & ('.join([
    f'(ds_m_mass<={m_Bs}',
    'abs(mu_bs_dxy_sig)>5',
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
    'mu_charge*pi_charge<0)'])"""

# changes in the selecion:
# - mu_rel_iso: <0.1 -> <1


# possible changes in the selection:
# - ds_vtx_prob: >0.1 -> >0.05
# - lxyz_ds_m_sig: >10 -> >5,
# - pt_miss: >0 -> >1.6501322
#   (because that's the smalles pt_miss for tau signals in Reco)
# - ds_m_mass: >2.16 -> =>2.3491971
# - someting like abs(mu_bs_dxy) > 0.01


selection = [
    f'ds_m_mass<={m_Bs}',
    'mu_pt>8',
    'mu_id_medium==1', 
    'mu_rel_iso<1',
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
    'ds_m_mass>2.16',
    'mu_charge*pi_charge<0']
# convert selection to a string
selection = '(' + ') & ('.join(selection) + ')'


# import of CMS Data
####################

# aliases for importing from the root tree
aliases = {
    'pt_miss': f'ds_m_pt*{m_Bs}/ds_m_mass - k1_pt - k2_pt - pi_pt - mu_pt'
    }

# open CMS root file
root_tree_CMS = uproot.open("ROOT_data/data_skimmed.root:tree")

# list of branches in the CMS tree that are used
keys_tree_CMS = root_tree_CMS.keys() + ['pt_miss']

# filter according to cuts and save required branches in the CMS tree
tree_CMS = root_tree_CMS.arrays(keys_tree_CMS, cut=selection, aliases=aliases,
                                library="np")

# create tree with combinatorial background (pi, mu same charge) from 
# CMS root file
if selection:
    selection_comb = selection.replace('mu_charge*pi_charge<0',
                                   'mu_charge*pi_charge>0')
else:
    selection_comb=None
tree_comb = root_tree_CMS.arrays(keys_tree_CMS, cut=selection_comb,
                                 aliases=aliases, library="np")

# add sig branch containing -2 to tree_comb
tree_comb['sig'] = -2*np.ones_like(tree_comb['q2'])


# import of Reco Data
#####################

# open Reco root file
root_tree_Reco = uproot.open("ROOT_data/inclusive_mc_3may.root:tree")

# list of branches in the reco tree that are used
keys_tree_Reco = root_tree_Reco.keys() + ['pt_miss']

# filter according to cuts and save required branches in the Reco tree
tree_Reco = root_tree_Reco.arrays(keys_tree_Reco, cut=selection,
                                  aliases=aliases, library="np")

# import the additional tau signals
root_tree_Reco_tau = uproot.open("ROOT_data/tau_mc_3may.root:tree")
tree_Reco_tau = root_tree_Reco_tau.arrays(keys_tree_Reco, cut=selection,
                                          aliases=aliases, library="np")

# create a tree containing tree_Reco and tree_Reco_tau
tree_Reco_plus = {
    k: np.append(v, v_tau)
    for (k, v), v_tau in zip(tree_Reco.items(), tree_Reco_tau.values())
    }

# labelling of the different signals in sig
labels = {'-1': r'$H_b \rightarrow D_s + \mu$',
          '0': r"$B_s \rightarrow D_s \, \mu \nu$",
          '1': r"$B_s \rightarrow D_s^* \mu \nu$",
          '2': r"$B_s \rightarrow D_s \, \tau \nu$",
          '3': r"$B_s \rightarrow D^*_s \tau \nu$"}

# number of events (per signal)
N_tot = tree_Reco['q2'].size
N_m1 = np.sum(tree_Reco['sig'] == -1)    # background
N_0 = np.sum(tree_Reco['sig'] == 0)      # Ds  mu nu
N_1 = np.sum(tree_Reco['sig'] == 1)      # Ds* mu nu
N_2 = np.sum(tree_Reco['sig'] == 2)      # Ds  tau nu
N_3 = np.sum(tree_Reco['sig'] == 3)      # Ds* tau nu

# list of elements containing nans
"""list_of_nans = []
for k, leaf in tree_Reco.items():
    if np.isnan(leaf).any():
        list_of_nans += [k]"""


#################################
# sorting/processing of momenta #
#################################

# converting momentum in tree into 4-momenta
############################################
keys_p4 = ['mu', 'k1', 'k2', 'pi']

p4_Reco = dict()
for k_p4 in keys_p4:
    p4_Reco[k_p4] = vector.array({'pt': tree_Reco[k_p4+'_pt'],
                             "phi": tree_Reco[k_p4+'_phi'],
                             "eta": tree_Reco[k_p4+'_eta'],
                             "E": tree_Reco[k_p4+'_e']})

p4_CMS = dict()
for k_p4 in keys_p4:
    p4_CMS[k_p4] = vector.array({'pt': tree_CMS[k_p4+'_pt'],
                             "phi": tree_CMS[k_p4+'_phi'],
                             "eta": tree_CMS[k_p4+'_eta'],
                             "E": tree_CMS[k_p4+'_e']})


#############
# Filtering #
#############



# %%####################
# Plots of histogramms #
########################

# special properties for the plots
##################################
bins_dict = dict()
legend_locs = {
    'q2': 2
    }
xlims = {
    'mu_pt': (5, 50),
    'mu_rel_iso': (-0.2, 3),
    'mu_bs_dxy_sig': (-80, 80),
    'k1_pt': (0, 20),
    'k2_pt': (0, 17),
    'pi_pt': (0, 30),
    'cos3D_ds_m': (0.8, 1.05),
    'cos3D_ds_m': (0.97, 1.001),
    'lxyz_ds_m_sig': (0, 200),
    'pt_miss': (-5, 70)
}

keys = [
    'q2',
    'ds_m_mass',
    'ds_mass',
    'phi_mass',
    'mu_bs_dxy_sig',
    'mu_pt',
    'k1_pt',
    'k2_pt',
    'pi_pt',
    'pt_miss',
    'mu_rel_iso',
    'ds_vtx_prob',
    'cos3D_ds_m',
    'phi_vtx_prob',
    'lxyz_ds_m_sig',
    'mu_id_medium',
    'HLT_Mu7_IP4'
]

not_normalized = [
    'mu_id_medium',
    'HLT_Mu7_IP4'
    ]


def rawHistPlots(tree):
    """Plots and saves all entries in tree as histogram"""
    for i, (k, v) in enumerate(tree.items()):
        # if only nans, continue
        if np.isfinite(v).sum() == 0:
            continue
        plt.hist(v, bins='sqrt', histtype='step')
        plt.title(k), plt.minorticks_on()
        plt.savefig(f'Plots_of_kinematic_variables/all_hists2/{i}_{k}.pdf')
        plt.close()

def rawHistPlots_sig_plus_CMS(tree, keys, folder, bins='auto'):
    """Plots and saves all entries in tree as histogram where the signals
    are split. 'tree' has to be a Reco tree"""
    for i, (k, v) in enumerate(tree.items()):
        if k not in keys:
            continue
        # if there is no such a branche in the CMS data (not only containing
        #  NaNs) continue
        if k not in tree_CMS.keys() or np.isfinite(tree_CMS[k]).sum() == 0:
            continue
        # if one signal only nans, continue
        if min([np.isfinite(v[tree['sig']==int(sig)]).sum()
                for sig in keys_sig]) == 0:
            continue

        range = (min(np.nanmin(v), np.nanmin(tree_CMS[k])),
                 max(np.nanmax(v), np.nanmax(tree_CMS[k])))
        if k in bins_dict:
            bins_current = bins_dict[k]
        if type(bins) == str:
            bins_list = [np.histogram_bin_edges(v[(tree['sig']==int(sig))
                                                  & (np.isfinite(v))],
                                                bins, range=range)
                        for sig in keys_sig]
            bins_current = bins_list[np.argmin(list(map(len, bins_list)))]
        density = not k in not_normalized
        # plot hists of reco tree
        plotHist_sig(tree[k], tree, xlabel=k, filename=f'{k}.pdf',
                     bins=bins_current, range=range, density=density,
                     show=None)
        # plot hist of CMS tree
        plt.hist(tree_CMS[k], bins_current, range=range, label='data',
                 density=density, color='k', histtype='step')

        if k in xlims:
            plt.xlim(xlims[k])
        if k in legend_locs:
            plt.legend(loc=legend_locs[k])
        else:
            plt.legend()
        plt.savefig(folder + f'{i}_{k}.pdf')
        plt.close()
    

def plotHist(data, xlabel=None, filename=None, label=None, bins='auto',
             xlim=None, range=None, density=True, show=True, folder=None,
             alpha=None, minorticks=True, legend_loc=0):
    plt.hist(data, bins=bins, range=range, density=density, label=label,
             alpha=alpha, histtype='step')
    if xlim:
        plt.xlim(xlim)
    if label:
        plt.legend(loc=legend_loc)
    if minorticks:
        plt.minorticks_on()
    plt.xlabel(xlabel) ,plt.ylabel("a.u.")
    if folder:
        plt.savefig(folder + filename)
    if show:
        plt.show()
    elif show==None:
        pass
    else:
        plt.close()
    return


#alpha_std = {'-1':0.4, '0':0.5, '1':0.6, '2':0.8, '3':1}
alpha_std = {'-1': 1, '0': 1, '1': 1, '2': 1, '3': 1}
keys_sig = ['-1', '0', '1', '2', '3'] #['3', '2', '1', '0', '-1']

def plotHist_sig(data, tree=tree_Reco, xlabel=None, filename=None, bins='auto',
                 xlim=None, range=None, density=True, show=True, folder=None,
                 keys_sig=keys_sig, alpha=alpha_std, minorticks=True,
                 legend_loc=0):
    if range==None:
        range = (np.nanmin(data), np.nanmax(data))
    if type(bins) == str:
        bins_list = [np.histogram_bin_edges(data[(tree['sig']==int(sig))
                                                 & (np.isfinite(data))],
                                            bins, range=range)
                     for sig in keys_sig]
        bins = bins_list[np.argmin(list(map(len, bins_list)))]
    for sig in keys_sig:
        plt.hist(data[tree['sig']==int(sig)], bins=bins, range=range,
                 density=density, alpha=alpha[sig], label=labels[sig],
                 histtype='step')
    if xlim:
        plt.xlim(xlim)
    plt.legend(loc=legend_loc)
    if minorticks:
        plt.minorticks_on()
    plt.xlabel(xlabel), plt.ylabel("a.u.")
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
#################

# plot of phi mass
def phi_mass_plot(show=True, folder=None):
    plotHist((p4_Reco['k_same'] + p4_Reco['k_opposite']).m,
             xlabel=r"$m_\phi$ [GeV]", filename="phi_mass.pdf", bins=100,
             xlim=(0.25, 0.8), show=show, folder=folder)
    
# plot of phi mass split into different signals
def phi_mass_split_sig_plots(show=True, folder=None):
    plotHist_sig((p4_Reco['k_same'] + p4_Reco['k_opposite']).m,
                 xlabel=r"$m_\phi$ [GeV]", filename="phi_mass_split_sig.pdf",
                 bins=50, xlim=(0.25,0.8), show=show, folder=folder)

# plot of phi mass when exchanging pi and k_same mass
def phi_mass_k_same_pi_mass_exch_plot(show=True, folder=None):
    xlabel = r"$m_\phi$ for $K_{\mathrm{opposite}}$ and " \
             + r"$\pi$ with $K_{\mathrm{same}}$ mass [GeV]"
    plotHist((p4_Reco['pi_m_exch'] + p4_Reco['k_opposite']).m, xlabel,
             filename="phi_mass_k_same_pi_mass_exch.pdf",
             bins=100, xlim=(0.5, 2.5), show=show, folder=folder)

# plot of phi mass when exchanging pi and k_same mass, split into different
# signals
def phi_mass_k_same_pi_mass_exch_split_sig_plots(show=True, folder=None):
    xlabel = r"$m_\phi$ for $K_{\mathrm{opposite}}$ and $\pi$ "\
             + r"with $K_{\mathrm{same}}$ mass [GeV]"
    plotHist_sig((p4_Reco['pi_m_exch'] + p4_Reco['k_opposite']).m, xlabel,
                 filename="phi_mass_k_same_pi_mass_exch_split_sig.pdf",
                 bins=50, xlim=(0.5, 2.7), show=show, folder=folder)

# plot of phi mass when exchanging pi and k_same mass, split into K charge
# (sanity check)
def phi_mass_k_same_pi_mass_exch_split_charge_plots(show=True, folder=None):
    phi_p4_m_exch = p4_Reco['pi_m_exch'] + p4_Reco['k_opposite']
    xlabel = r"$m_\phi$ for $K_{\mathrm{opposite}}$ and $\pi$ " \
             + r"with $K_{\mathrm{same}}$ mass [GeV]"
    plotHist(phi_p4_m_exch.m[tree_Reco['mu_charge'] == -1],
             label='$\pi^-$ with $m_{K}}}$ and $K^+$', xlabel=None,
             filename=None, bins=100, show=None, alpha=1)
    plotHist(phi_p4_m_exch.m[tree_Reco['mu_charge'] == +1],
             label='$\pi^+$ with $m_{K}}}$ and $K^-$', xlabel=xlabel,
             filename="phi_mass_k_same_pi_mass_exch_split_charge.pdf",
             bins=100, xlim=(0.6, 2.5), show=show, folder=folder, alpha=1)

# plot of phi mass for both variants
def phi_mass_both_variant_plot(show=True, folder=None):
    plt.ylim([0, 3])
    plotHist((p4_Reco['k_same'] + p4_Reco['k_opposite']).m,
             label='$K_{\mathrm{same}}$ and $K_{\mathrm{opposite}}$',
             bins=30, show=None)
    plotHist((p4_Reco['pi_m_exch'] + p4_Reco['k_opposite']).m,
             label='$\pi$ with $m_{K}}$ and $K_{\mathrm{opposite}}$',
             xlabel=r"$m_\phi$ [GeV]", filename="phi_mass_both_variant.pdf",
             bins=100, xlim=(0.2, 2.1), show=show, folder=folder)

# plot of phi mass for both variants in log scale
def phi_mass_both_variant_logPlot(show=True, folder=None):
    plt.yscale('log')
    plotHist((p4_Reco['k_same'] + p4_Reco['k_opposite']).m,
             label='$K_{\mathrm{same}}$ and $K_{\mathrm{opposite}}$',
             bins=30, show=None)
    plotHist((p4_Reco['pi_m_exch'] + p4_Reco['k_opposite']).m,
             label='$\pi$ with $m_{K}}$ and $K_{\mathrm{opposite}}$',
             xlabel=r"$m_\phi$ [GeV]",
             filename="phi_mass_both_variant_log.pdf",
             bins=100, xlim=(0, 3), show=show, folder=folder)


def plotAllHist1(show=False, folder=None):
    phi_mass_plot(show=show, folder=folder)
    phi_mass_split_sig_plots(show=show, folder=folder)
    phi_mass_k_same_pi_mass_exch_plot(show=show, folder=folder)
    phi_mass_k_same_pi_mass_exch_split_sig_plots(show=show, folder=folder)
    phi_mass_k_same_pi_mass_exch_split_charge_plots(show=show, folder=folder)
    phi_mass_both_variant_plot(show=show, folder=folder)
    phi_mass_both_variant_logPlot(show=show, folder=folder)





# plot of other kinematic variables
###################################

# plot of mass, pt, eta, phi of ds_m
def ds_m_plots(show=True, folder=None):
    plotHist_sig(tree_Reco['ds_m_mass'], xlabel=r"Mass of $D_s\,\mu$ system [GeV]",
                 filename="ds_m_mass.pdf", bins=30, range=(2, 8),
                 show=show, folder=folder)
    plotHist_sig(tree_Reco['ds_m_pt'], xlabel=r"$p_T$ of $D_s\,\mu$ system [GeV]",
                 filename="ds_m_pt.pdf", bins=150, xlim=(7, 70),
                 show=show, folder=folder)
    plotHist_sig(tree_Reco['ds_m_eta'], xlabel=r"$\eta$ of $D_s\,\mu$ system",
                 filename="ds_m_eta.pdf", bins=20, show=show, folder=folder)
    plotHist_sig(tree_Reco['ds_m_phi'], xlabel=r"$\phi$ of $D_s\,\mu$ system",
                 filename="ds_m_phi.pdf", bins=20, show=show, folder=folder)

# plot of mass, pt, eta, phi of ds_m, with log scale
def ds_m_pt_logPlots(show=True, folder=None):
    plt.yscale('log')
    plotHist_sig(tree_Reco['ds_m_pt'], xlabel=r"$p_T$ of $D_s\,\mu$ system [GeV]",
                 filename="ds_m_pt_log.pdf", bins=50, xlim=(5, 230),
                 show=show, folder=folder)

# plot of q2
def q2_plot(show=True, folder=None):
    plotHist_sig(tree_Reco['q2'], xlabel=r"$Q^2$ [GeV$^2$]", filename="q2.pdf",
                 bins=40, range=(-13.21, 12.1), show=show, folder=folder,
                 legend_loc=2)

# plot of q2, with log scale
def q2_logPlot(show=True, folder=None):
    plt.yscale('log')
    plotHist_sig(tree_Reco['q2'], xlabel=r"$Q^2$ [GeV$^2$]", filename="q2_log.pdf",
                 bins=20, show=show, folder=folder, legend_loc=2)

# plot of m2_miss
def m2_miss_plot(show=True, folder=None):
    plotHist_sig(tree_Reco['m2_miss'], xlabel=r"$m^2_{miss}$ [GeV$^2$]",
                 filename="m2_miss.pdf", bins=30, show=show, folder=folder)

# plot of energy of unpaired mu
def e_star_mu3_plot(show=True, folder=None):
    plotHist_sig(tree_Reco['e_star_mu3'], xlabel=r"$E^*_\mu$ [GeV]",
                 filename="e_star_mu3.pdf", bins=40, range=(0.16, 3.8),
                 xlim=(0.2, 3.3), show=show, folder=folder)

# plot of energy of unpaired mu, with log scale
def e_star_mu3_logPlot(show=True, folder=None):
    plt.yscale('log')
    plotHist_sig(tree_Reco['e_star_mu3'], xlabel=r"$E^*_\mu$ [GeV]",
                 filename="e_star_mu3_log.pdf", bins=40,
                 show=show, folder=folder)


# plot of missing transversal momentum

# neglecting direction, photon in ds_st signal not taken into account
def pt_miss_scalar_plot(show=True, folder=None):
    pt_miss = tree_Reco['ds_m_pt']*m_Bs/tree_Reco['ds_m_mass'] \
              - tree_Reco['km_pt'] - tree_Reco['kp_pt'] - tree_Reco['pi_pt']- tree_Reco['mu_pt']
    plotHist_sig(pt_miss, xlabel=r"scalar $p_T^{miss}$ [GeV]",
                 filename="pt_miss_scalar.pdf", bins=250, xlim=(-10, 60),
                 show=show, folder=folder)

# plot of missing transversal momentum, with log scale
def pt_miss_scalar_logPlot(show=True, folder=None):
    pt_miss = tree_Reco['ds_m_pt']*m_Bs/tree_Reco['ds_m_mass'] \
              - tree_Reco['km_pt'] - tree_Reco['kp_pt'] - tree_Reco['pi_pt']- tree_Reco['mu_pt']
    plt.yscale('log')
    plotHist_sig(pt_miss, xlabel=r"scalar $p_T^{miss}$ [GeV]",
                 filename="pt_miss_scalar_log.pdf", bins=120, xlim=(-30, 150),
                 show=show, folder=folder)


# plot of mass, pt, eta, phi of mu
def mu_plots(show=True, folder=None):
    plotHist_sig(tree_Reco['mu_pt'], xlabel=r"$p_T^\mu$ [GeV]",
                 filename="mu_pt.pdf", bins=200, xlim=(6.5, 27),
                 show=show, folder=folder)
    plotHist_sig(tree_Reco['mu_eta'], xlabel=r"$\eta_\mu$", filename="mu_eta.pdf",
                 bins=20, show=show, folder=folder)
    plotHist_sig(tree_Reco['mu_phi'], xlabel=r"$\phi_\mu$", filename="mu_phi.pdf",
                 bins=20, show=show, folder=folder)
    plotHist_sig(tree_Reco['mu_e'], xlabel=r"$E_\mu$ [GeV]", filename="mu_e.pdf",
                 bins=150, xlim=(6, 50), show=show, folder=folder)

# plot of mass, pt, eta, phi of kp
def kp_plots(show=True, folder=None):
    plotHist_sig(tree_Reco['kp_pt'], xlabel=r"$p_T^{K^+}$ [GeV]",
                 filename="kp_pt.pdf", bins=200, xlim=(-0.1, 20), show=show,
                 folder=folder)
    plotHist_sig(tree_Reco['kp_eta'], xlabel=r"$\eta_{K^+}$", filename="kp_eta.pdf",
                 bins=20, show=show, folder=folder)
    plotHist_sig(tree_Reco['kp_phi'], xlabel=r"$\phi_{K^+}$", filename="kp_phi.pdf",
                 bins=20, show=show, folder=folder)
    plotHist_sig(tree_Reco['kp_e'], xlabel=r"$E_{K^+}$ [GeV]", filename="kp_e.pdf",
                 bins=150, xlim=(0, 25), show=show, folder=folder)

# plot of mass, pt, eta, phi of km
def km_plots(show=True, folder=None):
    plotHist_sig(tree_Reco['km_pt'], xlabel=r"$p_T^{K^-}$ [GeV]",
                 filename="km_pt.pdf", bins=220, xlim=(0, 18),
                 show=show, folder=folder)
    plotHist_sig(tree_Reco['km_eta'], xlabel=r"$\eta_{K^-}$", filename="km_eta.pdf",
                 bins=20, show=show, folder=folder)
    plotHist_sig(tree_Reco['km_phi'], xlabel=r"$\phi_{K^-}$", filename="km_phi.pdf",
                 bins=20, show=show, folder=folder)
    plotHist_sig(tree_Reco['km_e'], xlabel=r"$E_{K^-}$ [GeV]", filename="km_e.pdf",
                 bins=200, xlim=(0, 25), show=show, folder=folder)

# plot of mass, pt, eta, phi of k_big
def k_big_plots(show=True, folder=None):
    plotHist_sig(tree_Reco['k_big_pt'], xlabel=r"$p_T^{K_\mathrm{big}}$ [GeV]",
                 filename="k_big_pt.pdf", bins=200, xlim=(0, 20),
                 show=show, folder=folder)
    plotHist_sig(tree_Reco['k_big_eta'], xlabel=r"$\eta_{K_\mathrm{big}}$",
                 filename="k_big_eta.pdf", bins=20, show=show, folder=folder)
    plotHist_sig(tree_Reco['k_big_phi'], xlabel=r"$\phi_{K_\mathrm{big}}$",
                 filename="k_big_phi.pdf", bins=20, show=show,  folder=folder)
    plotHist_sig(tree_Reco['k_big_e'], xlabel=r"$E_{K_\mathrm{big}}$ [GeV]",
                 filename="k_big_e.pdf", bins=200, xlim=(0, 25),
                 show=show, folder=folder)

# plot of mass, pt, eta, phi of k_small
def k_small_plots(show=True, folder=None):
    plotHist_sig(tree_Reco['k_small_pt'], xlabel=r"$p_T^{K_\mathrm{small}}$ [GeV]",
                 filename="k_small_pt.pdf", bins=200, xlim=(0, 17.5),
                 show=show, folder=folder)
    plotHist_sig(tree_Reco['k_small_eta'], xlabel=r"$\eta_{K_\mathrm{small}}$",
                 filename="k_small_eta.pdf", bins=20,
                 show=show, folder=folder)
    plotHist_sig(tree_Reco['k_small_phi'], xlabel=r"$\phi_{K_\mathrm{small}}$",
                 filename="k_small_phi.pdf", bins=20,
                 show=show, folder=folder)
    plotHist_sig(tree_Reco['k_small_e'], xlabel=r"$E_{K_\mathrm{small}}$ [GeV]",
                 filename="k_small_e.pdf", bins=200, xlim=(0, 20),
                 show=show, folder=folder)

# plot of mass, pt, eta, phi of pi
def pi_plots(show=True, folder=None):
    plotHist_sig(tree_Reco['pi_pt'], xlabel=r"$p_T^\pi$ [GeV]",
                 filename="pi_pt.pdf", bins=200, xlim=(0, 20),
                 show=show, folder=folder)
    plotHist_sig(tree_Reco['pi_eta'], xlabel=r"$\eta_\pi$", filename="pi_eta.pdf",
                 bins=20, show=show, folder=folder)
    bins = {'-1':30, '0':30, '1':30, '2':15, '3':15}
    plotHist_sig(tree_Reco['pi_phi'], xlabel=r"$\phi_\pi$", filename="pi_phi.pdf",
                 bins=20, show=show, folder=folder)
    plotHist_sig(tree_Reco['pi_e'], xlabel=r"$E_\pi$ [GeV]", filename="pi_e.pdf",
                 bins=200, xlim=(0, 25), show=show, folder=folder)


# plot of dR of kp mu
def dR_mu_kp_plot(show=True, folder=None):
    plotHist_sig(p4_Reco['mu'].deltaR(p4_Reco['kp']), xlabel=r"$\Delta R(\mu, K^+)$",
                 filename="dR_mu_kp.pdf", bins=25, show=show, folder=folder)

# plot of dR of km mu
def dR_mu_km_plot(show=True, folder=None):
    plotHist_sig(p4_Reco['mu'].deltaR(p4_Reco['km']), xlabel=r"$\Delta R(\mu, K^-)$",
                 filename="dR_mu_km.pdf", bins=25, show=show, folder=folder)


# plot of dR of k_big mu
def dR_mu_k_big_plot(show=True, folder=None):
    plotHist_sig(p4_Reco['mu'].deltaR(p4_Reco['k_big']),
                 xlabel=r"$\Delta R(\mu, K_\mathrm{big})$",
                 filename="dR_mu_k_big.pdf", bins=25,
                 show=show, folder=folder)

# plot of dR of k_big mu, with log scale
def dR_mu_k_big_logPlot(show=True, folder=None):
    plt.yscale('log')
    plotHist_sig(p4_Reco['mu'].deltaR(p4_Reco['k_big']),
                 xlabel=r"$\Delta R(\mu, K_\mathrm{big})$",
                 filename="dR_mu_k_big_log.pdf", bins=25,
                 show=show, folder=folder)

# plot of dR of k_small mu
def dR_mu_k_small_plot(show=True, folder=None):
    plotHist_sig(p4_Reco['mu'].deltaR(p4_Reco['k_small']),
                 xlabel=r"$\Delta R(\mu, K_\mathrm{small})$",
                 filename="dR_mu_k_small.pdf", bins=25,
                 show=show, folder=folder)

# plot of dR of k_small mu, with log scale
def dR_mu_k_small_logPlot(show=True, folder=None):
    plt.yscale('log')
    plotHist_sig(p4_Reco['mu'].deltaR(p4_Reco['k_small']),
                 xlabel=r"$\Delta R(\mu, K_\mathrm{small})$",
                 filename="dR_mu_k_small_log.pdf", bins=25,
                 show=show, folder=folder)

# plot of dR of pi mu
def dR_mu_pi_plot(show=True, folder=None):
    plotHist_sig(p4_Reco['mu'].deltaR(p4_Reco['pi']), xlabel=r"$\Delta R(\mu, \pi)$",
                 filename="dR_mu_pi.pdf", bins=25, show=show, folder=folder)

# plot of dR of pi mu, with log scale
def dR_mu_pi_logPlot(show=True, folder=None):
    plt.yscale('log')
    plotHist_sig(p4_Reco['mu'].deltaR(p4_Reco['pi']), xlabel=r"$\Delta R(\mu, \pi)$",
                 filename="dR_mu_pi_log.pdf", bins=25,
                 show=show, folder=folder)


def plotAllHist2(show=False, folder=None):
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


    




#########
# ToDos #
#########
# - Plot remaining kinematic variables
# - Check cuts again
# - 

# %%
