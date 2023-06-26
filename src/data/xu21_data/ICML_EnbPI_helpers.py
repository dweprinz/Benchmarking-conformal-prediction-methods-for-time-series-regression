from scipy.linalg import norm
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn import preprocessing
import seaborn as sns
from scipy.stats import beta
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
from numpy.random import choice
import warnings
import os
import matplotlib.cm as cm
import time

'''Helper for ensemble'''


def generate_bootstrap_samples(n, m, B):
    '''
      Return: B-by-m matrix, where row b gives the indices for b-th bootstrap sample
    '''
    samples_idx = np.zeros((B, m), dtype=int)
    for b in range(B):
        sample_idx = np.random.choice(n, m)
        samples_idx[b, :] = sample_idx
    return(samples_idx)


'''Helper for doing online residual'''


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))


'''Get percentile from beta '''

# NOTE: Unused


def beta_percentile(p_vals, alpha):
    p_mean = np.mean(p_vals)
    p_var = np.std(p_vals)**2
    product = (p_mean*(1-p_mean)/p_var-1)
    a_hat = p_mean*product
    b_hat = (1-p_mean)*product
    return beta.ppf(1-alpha, a_hat, b_hat)


'''Helper for Weighted ICP'''


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


"""
For Plotting results: width and coverage plots
"""


def plot_average_new(x_axis, x_axis_name, save=True, Dataname=['Solar_Atl'], two_rows=True):
    """Plot mean coverage and width for different PI methods and regressor combinations side by side,
       over rho or train_size or alpha_ls
       Parameters:
        data_type: simulated (2-by-3) or real data (2-by-2)
        x_axis: either list of train_size, or alpha
        x_axis_name: either train_size or alpha
    """
    ncol = 2
    Dataname.append(Dataname[0])  # for 1D results
    if two_rows:
        fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
    else:
        fig, ax = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
    j = 0
    filename = {'alpha': 'alpha', 'train_size': 'train'}
    one_D = False
    for data_name in Dataname:
        # load appropriate data
        if j == 1 or one_D:
            results = pd.read_csv(f'{data_name}_many_{filename[x_axis_name]}_new_1d.csv')
        else:
            results = pd.read_csv(f'{data_name}_many_{filename[x_axis_name]}_new.csv')
        methods_name = np.unique(results.method)
        cov_together = []
        width_together = []
        # Loop through dataset name and plot average coverage and width for the particular regressor
        muh_fun = np.unique(results[results.method != 'ARIMA']
                            ['muh_fun'])  # First ARIMA, then Ensemble
        for method in methods_name:
            if method == 'ARIMA':
                results_method = results[(results['method'] == method)]
                if data_name == 'Network':
                    method_cov = results_method.groupby(
                        by=[x_axis_name, 'node'], as_index=False).mean().groupby(x_axis_name)['coverage'].describe()  # Column with 50% is median
                    method_width = results_method.groupby(
                        by=[x_axis_name, 'node'], as_index=False).mean().groupby(x_axis_name)['width'].describe()  # Column with 50% is median
                else:
                    method_cov = results_method.groupby(
                        x_axis_name)['coverage'].describe()  # Column with 50% is median
                    method_width = results_method.groupby(
                        x_axis_name)['width'].describe()  # Column with 50% is median
                    method_cov['se'] = method_cov['std']/np.sqrt(method_cov['count'])
                    method_width['se'] = method_width['std']/np.sqrt(method_width['count'])
                    cov_together.append(method_cov)
                    width_together.append(method_width)
            else:
                for fit_func in muh_fun:
                    results_method = results[(results['method'] == method) &
                                             (results['muh_fun'] == fit_func)]
                    if data_name == 'Network':
                        method_cov = results_method.groupby(
                            by=[x_axis_name, 'node'], as_index=False).mean().groupby(x_axis_name)['coverage'].describe()  # Column with 50% is median
                        method_width = results_method.groupby(
                            by=[x_axis_name, 'node'], as_index=False).mean().groupby(x_axis_name)['width'].describe()  # Column with 50% is median
                    else:
                        method_cov = results_method.groupby(
                            x_axis_name)['coverage'].describe()  # Column with 50% is median
                        method_width = results_method.groupby(
                            x_axis_name)['width'].describe()  # Column with 50% is median
                    method_cov['se'] = method_cov['std']/np.sqrt(method_cov['count'])
                    method_width['se'] = method_width['std']/np.sqrt(method_width['count'])
                    cov_together.append(method_cov)
                    width_together.append(method_width)
        # Plot
        # Parameters
        num_method = 1+len(muh_fun)  # ARIMA + EnbPI
        colors = cm.rainbow(np.linspace(0, 1, num_method))
        mtds = np.append('ARIMA', muh_fun)
        # label_names = methods_name
        label_names = {'ARIMA': 'ARIMA', 'RidgeCV': 'EnbPI Ridge',
                       'RandomForestRegressor': 'EnbPI RF', 'Sequential': 'EnbPI NN', 'RNN': 'EnbPI RNN'}
        # if 'ARIMA' in methods_name:
        #     colors = ['orange', 'red', 'blue', 'black']
        # else:
        #     colors = ['red', 'blue', 'black']
        first = 0
        second = 1
        if one_D:
            first = 2
            second = 3
        axisfont = 20
        titlefont = 24
        tickfont = 16
        name = 'mean'
        for i in range(num_method):
            if two_rows:
                # Coverage
                ax[j, first].plot(x_axis, cov_together[i][name], linestyle='-',
                                  marker='o', label=label_names[mtds[i]], color=colors[i])
                ax[j, first].fill_between(x_axis, cov_together[i][name]-cov_together[i]['se'],
                                          cov_together[i][name]+cov_together[i]['se'], alpha=0.35, facecolor=colors[i])
                ax[j, first].set_ylim(0, 1)
                ax[j, first].tick_params(axis='both', which='major', labelsize=tickfont)
                # Width
                ax[j, second].plot(x_axis, width_together[i][name], linestyle='-',
                                   marker='o', label=label_names[mtds[i]], color=colors[i])
                ax[j, second].fill_between(x_axis, width_together[i][name]-width_together[i]['se'],
                                           width_together[i][name]+width_together[i]['se'], alpha=0.35, facecolor=colors[i])
                ax[j, second].tick_params(axis='both', which='major', labelsize=tickfont)
                # Legends, target coverage, labels...
                # Set label
                ax[j, first].plot(x_axis, x_axis, linestyle='-.', color='green')
                # x_ax = ax[j, first].axes.get_xaxis()
                # x_ax.set_visible(False)
                nrow = len(Dataname)
                ax[nrow-1, 0].set_xlabel(r'$1-\alpha$', fontsize=axisfont)
                ax[nrow-1, 1].set_xlabel(r'$1-\alpha$', fontsize=axisfont)
            else:
                # Coverage
                ax[first].plot(x_axis, cov_together[i][name], linestyle='-',
                               marker='o', label=label_names[mtds[i]], color=colors[i])
                ax[first].fill_between(x_axis, cov_together[i][name]-cov_together[i]['se'],
                                       cov_together[i][name]+cov_together[i]['se'], alpha=0.35, facecolor=colors[i])
                ax[first].set_ylim(0, 1)
                ax[first].tick_params(axis='both', which='major', labelsize=tickfont)
                # Width
                ax[second].plot(x_axis, width_together[i][name], linestyle='-',
                                marker='o', label=label_names[mtds[i]], color=colors[i])
                ax[second].fill_between(x_axis, width_together[i][name]-width_together[i]['se'],
                                        width_together[i][name]+width_together[i]['se'], alpha=0.35, facecolor=colors[i])
                ax[second].tick_params(axis='both', which='major', labelsize=tickfont)
                # Legends, target coverage, labels...
                # Set label
                ax[first].plot(x_axis, x_axis, linestyle='-.', color='green')
                # x_ax = ax[j, first].axes.get_xaxis()
                # x_ax.set_visible(False)
                ax[first].set_xlabel(r'$1-\alpha$', fontsize=axisfont)
                ax[second].set_xlabel(r'$1-\alpha$', fontsize=axisfont)
        if two_rows:
            j += 1
        else:
            one_D = True
    if two_rows:
        ax[0, 0].set_title('Coverage', fontsize=axisfont)
        ax[0, 1].set_title('Width', fontsize=axisfont)
    else:
        ax[0].set_title('Coverage', fontsize=axisfont)
        ax[1].set_title('Width', fontsize=axisfont)
        ax[2].set_title('Coverage', fontsize=axisfont)
        ax[3].set_title('Width', fontsize=axisfont)
    if two_rows:
        ax[0, 0].set_ylabel('Multivariate', fontsize=axisfont)
        ax[1, 0].set_ylabel('Unitivariate', fontsize=axisfont)
    else:
        ax[0].set_ylabel('Multivariate', fontsize=axisfont)
        ax[2].set_ylabel('Unitivariate', fontsize=axisfont)
    fig.tight_layout(pad=0)
    if two_rows:
        # ax[0, 1].legend(loc='upper left', fontsize=axisfont-2)
        ax[1, 1].legend(loc='upper center',
                        bbox_to_anchor=(-0.08, -0.18), ncol=3, fontsize=axisfont-2)
    else:
        # ax[1].legend(loc='upper left', fontsize=axisfont-2)
        # ax[3].legend(loc='upper left', fontsize=axisfont-2)
        ax[3].legend(loc='upper center',
                     bbox_to_anchor=(-0.75, -0.18), ncol=5, fontsize=axisfont-2)
    if save:
        if two_rows:
            fig.savefig(
                f'{Dataname[0]}_mean_coverage_width_{x_axis_name}.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0)
        else:
            fig.savefig(
                f'{Dataname[0]}_mean_coverage_width_{x_axis_name}_one_row.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0)


def grouped_box_new(dataname, type, extra_save=''):
    '''First (Second) row contains grouped boxplots for multivariate (univariate) for Ridge, RF, and NN.
       Each boxplot contains coverage and width for all three PI methods over 3 (0.1, 0.3, 0.5) train/total data, so 3*3 boxes in total
       extra_save is for special suffix of plot (such as comparing NN and RNN)'''
    font_size = 18
    label_size = 20
    results = pd.read_csv(f'{dataname}_many_train_new{extra_save}.csv')
    results.sort_values('method', inplace=True, ascending=True)
    results.loc[results.method == 'Ensemble', 'method'] = 'EnbPI'
    results.loc[results.method == 'Weighted_ICP', 'method'] = 'Weighted ICP'
    # TODO: Use actual 1d later, because curren 1d is a copy of multi
    results_1d = pd.read_csv(f'{dataname}_many_train_new_1d{extra_save}.csv')
    results_1d.sort_values('method', inplace=True, ascending=True)
    results_1d.loc[results_1d.method == 'Ensemble', 'method'] = 'EnbPI'
    results_1d.loc[results_1d.method == 'Weighted_ICP', 'method'] = 'Weighted ICP'
    if 'Sequential' in np.array(results.muh_fun):
        results['muh_fun'].replace({'Sequential': 'NeuralNet'}, inplace=True)
        results_1d['muh_fun'].replace({'Sequential': 'NeuralNet'}, inplace=True)
    regrs = np.unique(results.muh_fun)
    regrs_label = {'RidgeCV': 'Ridge', 'LassoCV': 'Lasso', 'RandomForestRegressor': "RF",
                   'NeuralNet': "NN", 'RNN': 'RNN', 'GaussianProcessRegressor': 'GP'}
    # Set up plot
    ncol = 2  # Compare RNN vs NN
    if len(regrs) > 2:
        ncol = 3  # Ridge, RF, NN
        regrs = ['RidgeCV', 'NeuralNet', 'RNN']
    if type == 'coverage':
        f, ax = plt.subplots(2, ncol, figsize=(3*ncol, 6), sharex=True, sharey=True)
    else:
        # all plots in same row share y-axis
        f, ax = plt.subplots(2, ncol, figsize=(3*ncol, 6), sharex=True, sharey=True)
    f.tight_layout(pad=0)
    # Prepare for plot
    d = 20
    results_1d.train_size += d  # for plotting purpose
    tot_data = int(max(results.train_size)/0.278)
    results['ratio'] = np.round(results.train_size/tot_data, 2)
    results_1d['ratio'] = np.round(results_1d.train_size/tot_data, 2)
    j = 0  # column, denote aggregator
    ratios = np.unique(results['ratio'])
    # train_size_for_plot = [ratios[2], ratios[4], ratios[6], ratios[9]] # This was for 4 boxplots in one figure
    train_size_for_plot = ratios
    for regr in regrs:
        mtd = ['EnbPI', 'ICP', 'Weighted ICP']
        mtd_colors = ['red', 'royalblue', 'black']
        color_dict = dict(zip(mtd, mtd_colors))  # specify colors for each box
        # Start plotting
        which_train_idx = [fraction in train_size_for_plot for fraction in results.ratio]
        which_train_idx_1d = [fraction in train_size_for_plot for fraction in results_1d.ratio]
        results_plt = results.iloc[which_train_idx, ]
        results_1d_plt = results_1d.iloc[which_train_idx_1d, ]
        sns.boxplot(y=type, x='ratio',
                              data=results_plt[results_plt.muh_fun == regr],
                              palette=color_dict,
                              hue='method', ax=ax[0, j], showfliers=False)
        sns.boxplot(y=type, x='ratio',
                    data=results_1d_plt[results_1d_plt.muh_fun == regr],
                    palette=color_dict,
                    hue='method', ax=ax[1, j], showfliers=False)
        for i in range(2):
            ax[i, j].tick_params(axis='both', which='major', labelsize=14)
            if type == 'coverage':
                ax[i, j].axhline(y=0.9, color='black', linestyle='dashed')
            # Control legend
            ax[i, j].get_legend().remove()
            # Control y and x-label
            if j == 0:
                # Y-label on
                ax[0, 0].set_ylabel('Multivariate', fontsize=label_size)
                ax[1, 0].set_ylabel('Univariate', fontsize=label_size)
                if i == 1:
                    # X-label on
                    ax[1, j].set_xlabel(r'$\%$ of Total Data', fontsize=label_size)
                else:
                    # X-label off
                    x_axis = ax[i, j].axes.get_xaxis()
                    x_axis.set_visible(False)
            else:
                y_label = ax[i, j].axes.get_yaxis().get_label()
                y_label.set_visible(False)
                if type == 'coverage':
                    # Y-label off
                    y_axis = ax[i, j].axes.get_yaxis()
                    y_axis.set_visible(False)
                if i == 1:
                    # X-label on
                    ax[1, j].set_xlabel(r'$\%$ of Total Data', fontsize=label_size)
                else:
                    # X-label off
                    x_axis = ax[i, j].axes.get_xaxis()
                    x_axis.set_visible(False)
            # Control Title
            if i == 0:
                ax[0, j].set_title(regrs_label[regr], fontsize=label_size)
        j += 1
        # Legend lastly
    # Assign to top middle
    # ax[1, 1].legend(loc='upper center',
    #                 bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=font_size)
    plt.legend(loc='upper center',
               bbox_to_anchor=(-0.15, -0.25), ncol=3, fontsize=font_size)
    plt.savefig(
        f'{dataname}_boxplot_{type}{extra_save}.pdf', dpi=300, bbox_inches='tight',
        pad_inches=0)
