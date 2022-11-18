from __future__ import division
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from termcolor import colored
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv
import math
import random
import numpy as np
import pandas as pd


from sklearn.gaussian_process.kernels import RBF, WhiteKernel


#=======================[DATA]=========================

#	LOAD DATA

descriptor_list=[
		'tolerance_factor',
		'irA',
		'rsA',
		'rpA',
		'rdA',
		'XcA',
		'eaA',
		'ipA',
		'hoalA',
		'lualA',
		'irB1',
		'rsB1',
		'rpB1',
		'rdB1',
		'XcB1',
		'eaB1',
		'ipB1',
		'hoalB1',
		'lualB1',
		'irB2',
		'rsB2',
		'rpB2',
		'rdB2',
		'XcB2',
		'eaB2',
		'ipB2',
		'hoalB2',
		'lualB2',
		'irX',
		'rsX',
		'rpX',
		'rdX',
		'XcX',
		'eaX',
		'ipX',
		'hoalX',
		'lualX'
        ]

def pearson_coef(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = np.mean(x)
    avg_y = np.mean(y)
    diffprod = 0.0
    xdiff2 = 0.0
    ydiff2 = 0.0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff
    return diffprod / np.sqrt(xdiff2 * ydiff2)



df = pd.read_csv('mixed_single_double_perovskite.csv')
X = df[descriptor_list].astype(float)
Y = df['Eg'].astype(float)

print(X)

# Sec. 2: Calculate the Pearson correlation coefficients
feature = []
pc = []
j=0
for i in descriptor_list: 
    X2 = np.array(df[i])
    feature.append(i)
    pc.append(np.abs(pearson_coef(X2, Y)))
    j += 1



plt.clf()
y_pos = np.arange(len(feature))
plt.bar(y_pos, pc, align='center',alpha=0.5)
plt.xticks(y_pos,feature, rotation=90, fontsize=6)
plt.yticks(fontsize=6)
plt.title('Pearson correlation coefficient', fontsize=10)
plt.ylabel('coefficient')
plt.tight_layout()
ax=plt.axes()
ax.tick_params(axis='both', direction='in')
plt.savefig('pearson_mix_Data.pdf')





