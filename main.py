# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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
import statsmodels.api as stm

import itertools


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
data = 'polls.xls'
polls = pd.read_excel(data)
polls_2021 = pd.read_excel(data, sheet_name=0, index_col=0)
print("polls_2021 shape:")
print(polls_2021.shape)
just_polls = polls_2021.iloc[7:161, 5:25]
print("just polls")

print(just_polls)

covariance = just_polls.corr()
print("covariance")
print(covariance)

heat_map = sea.heatmap(covariance,annot=True, cmap = "RdBu")
plt.show()
correlation_3_day = pd.read_excel(data, sheet_name=1, index_col=0)
correlation_3_poll = pd.read_excel(data, sheet_name=2, index_col=0)
correlation_3_poll = correlation_3_poll.iloc[444:454, 1:19]
correlation_3_day = correlation_3_day.iloc[161:171, 1:19]
heat_map2 = sea.heatmap(correlation_3_day, annot=True, cmap = "RdBu");
plt.show()
heat_map3 = sea.heatmap(correlation_3_poll, annot=True, cmap = "RdBu");
plt.show()
# temp = just_polls[1:1, 1:1]
# start, stop =1;
stm.nonparametric.lowess()


