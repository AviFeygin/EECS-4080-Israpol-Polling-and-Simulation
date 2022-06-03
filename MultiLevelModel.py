import numpy
import torch
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import keras as ker
import seaborn as sea
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano

import itertools
plt.style.use('bmh')
plt.rcParams["figure.figsize"] = (8,5)

data = pd.read_csv('mlmldata.csv')
data.head()


def abline(slope, intercept, ax=None, **kwargs):
    """Plot a line from slope and intercept"""
    axes = ax if ax else plt.gca()
    x_vals = np.array([0, 7])
    y_vals = intercept + slope * x_vals
    axes.plot(x_vals, y_vals, **kwargs)


# facet with defaults specific to this data
def facetgrid(func):
    """ func is a function that takes the group df, matplotlib axes, group ID
        func must plot using the axes
    """
    fig, ax = plt.subplots(2, 5, figsize=(16, 7),
                           sharex=True, sharey=True,
                           constrained_layout=True)

    groups = data.groupby('schid')  # 10 schools
    grp_ids = list(groups.groups)

    for i, j in itertools.product(range(2), range(5)):
        grp_id = grp_ids[i * 5 + j]
        func(groups.get_group(grp_id), ax[i, j], grp_id)
        ax[i, j].set_title('schid : ' + str(grp_id), fontweight='bold')

    fig.text(0.5, -0.03, 'homework', ha='center', fontsize=16)
    fig.text(-0.02, 0.5, 'math', va='center', rotation='vertical', fontsize=16)
    handles, labels = ax[-1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))


# colors from BMH style
red = '#A60628'
blue = '#0072B2'
green = '#467821'
violet = '#7A68A6'
orange = '#D55E00'
pink = '#CC79A7'