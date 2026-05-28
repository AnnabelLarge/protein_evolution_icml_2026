#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.ticker as mticker
from matplotlib import rc
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = 'Optima'
plt.rcParams["font.size"] = 20
plt.rc('axes', unicode_minus=False)
plt.tight_layout()


def make_plots(log_scale: bool):
    aic_fig, aic_ax = plt.subplots(figsize=[8,6])
    xax_lab = 'Number of parameters'
    
    def add_to_plots(mixture_type,
                     label,
                     color,
                     linestyle = '-',
                     marker = 'o'):
        # extract
        sub_df = df[ df['model_type'] == mixture_type ]
        sub_df = sub_df.sort_values(by='parameters')
        assert len(sub_df) > 0
        
        # add to aic plot
        aic_ax.plot( sub_df['parameters'],
                     sub_df['aic_gain'],
                     marker = marker,
                     linestyle = linestyle,
                     color = color,
                     label = label,
                     linewidth = 3.5,
                     markersize = 10)
        
        del sub_df, label
    
    add_to_plots(mixture_type = 'pairhmm_domain_mix',
                 label = 'Mixture of domain classes',
                 color = 'tab:purple')
    
    add_to_plots(mixture_type = 'pairhmm_fragment_mix',
                 label = 'Mixture of fragment classes',
                 color = 'tab:green')
    
    add_to_plots(mixture_type = 'pairhmm_site_mix',
                 label = 'Mixture of site classes',
                 color = 'tab:orange')
    
    aic_ax.grid()
    aic_ax.legend()
    aic_ax.set_xlabel(xax_lab)
    aic_ax.set_ylabel('$\Delta$AIC (×$10^7$)')
    new_y_tick_labels = [f"{x:.1e}".split("e")[0] for x in aic_ax.get_yticks()]
    aic_ax.set_yticklabels( new_y_tick_labels )
    del new_y_tick_labels
    
    if log_scale:
        aic_ax.set_xscale('log')
    
    aic_fig.savefig(f'AIC_parameters_log_{log_scale}.pdf', 
                    bbox_inches="tight")


def read_file(file):
    df = pd.read_csv(file, sep='\t', index_col=0)
    sub_df = df[ (df['dataset'] == 'data1') &
                 (df['sub_model'] == 'f81') &
                 (df['indel_model'] == 'tkf92') ]
    sub_df = sub_df[['sub_model',
                     'indel_model',
                     'model_type',
                     'parameters',
                     'aic',
                     'bic']]
    sub_df = sub_df.sort_values(by='parameters')
    
    ref = sub_df[ sub_df['model_type']=='pairhmm_reference' ]
    sub_df['aic_gain'] = ref['aic'].item() - sub_df['aic']
    sub_df['bic_gain'] = ref['bic'].item() - sub_df['bic']
    
    sub_df = sub_df.drop( ['aic','bic'], axis=1 )
    sub_df = sub_df[ sub_df['model_type'] !='pairhmm_reference' ]
    return sub_df


if __name__ == '__main__':
    df = read_file('hierarchical_mixture_models_train_set_loglikes.tsv')
    make_plots(log_scale = True)
