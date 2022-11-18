import numpy as np
from matplotlib import *
import matplotlib.pyplot as plt
from data_preparation import *
from sklearn import preprocessing	
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import csv
from sklearn import cross_validation
import time
from sklearn import linear_model
from sklearn.linear_model import lasso_path, enet_path

###############################################################################
# Read Data
Header, A, extraction = read_data() 
y = np.log10(extraction)
X_scaled=preprocessing.scale(A)
X = X_scaled
#print Header.shape

##############################################################################
selector = linear_model.Lasso(alpha=0.01, max_iter=100000) # at 0.01 only 16 features survive
selector.fit(X,y)

sorted_idx = np.argsort(np.abs(selector.coef_))[::-1]
sc=np.abs(selector.coef_[sorted_idx])

Header = Header[sorted_idx]
print("Top ten features are:")
for i in range(0,10):
	print Header[i]

sort_coef = np.abs(selector.coef_[sorted_idx])
A = np.array(A.T[sorted_idx]).T
X = np.array(X.T[sorted_idx]).T
