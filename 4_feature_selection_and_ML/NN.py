#from sklearn.neural_network import MLPRegressor
# TRY TO REPLAFE MLP REGRESSOR WITH TENSOR FLOW
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from termcolor import colored
import matplotlib.pyplot as plt
import csv
import math
import random
import numpy as np
import pandas as pd


from sklearn.gaussian_process.kernels import RBF, WhiteKernel


#=======================[DATA]=======================================================================================
# Load data
df1 = pd.read_excel('data.xlsx', sheet_name='random')
X = df1[['B',
	'X',
	'OCT-PC1',
	'OCT-PC2',
	'OCT-PC3',
	'A-PC1',
	'A-PC2',
	'A-PC3',
	'Angles_mean',
	'Angles.std'	
	]].astype(float)
Y = df1['ed'].astype(float)

df2 = pd.read_excel('data.xlsx', sheet_name='must_have')
X_train_2 = df2[['B',
	'X',
	'OCT-PC1',
	'OCT-PC2',
	'OCT-PC3',
	'A-PC1',
	'A-PC2',
	'A-PC3',
	'Angles_mean',
	'Angles.std'	
	]].astype(float)
Y_train_2 = df2['ed'].astype(float)

#print 'X.shape',X.shape
#print 'X_train_2',X_train_2.shape

#print 'X.shape',X.shape
#print 'X_train_2',X_train_2.shape

#=======================[/DATA]=======================================================================================


total_run_no=10


#	Create empty list
train_ratio_csv=[]
run_no=[]
ose_csv=[]
ise_csv=[]
cver_csv=[]
pred_csv=[]
ba_csv=[]
bg_csv=[]
ose_mean_csv=[]
ise_mean_csv=[]
ose_std_csv=[]
ise_std_csv=[]


for ts in np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95]):
#for ts in np.array([0.8]):
	#Create empty list

	OSE_csv=[]
	ISE_csv=[]

	
	for run in range(total_run_no):
		print '===================================================================================================='
		X_train_1, X_test, Y_train_1, Y_test = train_test_split(X, Y, train_size=ts)
		train_total_ratio=float((ts*len(X)+len(X_train_2))/333)

#	add "must have data"


		frames1 = [X_train_1, X_train_2]
		X_train = pd.concat(frames1)

		frames2 = [Y_train_1, Y_train_2]
		Y_train = pd.concat(frames2)


#	print 'X_train',X_train.shape


# Train model
#	param_grid = {"alpha": np.logspace(-5, 5, 11),
#               "gamma": np.logspace(-5, 5, 11)}
#	model = GridSearchCV(KernelRidge(kernel='rbf',alpha=0.1,gamma=0.1), cv=5,
#  	param_grid=param_grid)


#	param_grid = {"alpha": [l
#	                       for l in np.logspace(-2, 2, 11)]}
#	model = GridSearchCV(MLPRegressor(
#				hidden_layer_sizes=(12,),
#				activation='tanh',
#				solver='lbfgs',
#				learning_rate='constant',
#				tol=0.0001,
#				learning_rate_init = 0.0001,
#				max_iter=500), cv=5, param_grid=param_grid)
#	model.fit(X_train, Y_train)	
	
#	model = tf.keras.models.Sequential([
#	model.add(Dense(12, input_dim=X_train.shape[1], activation='Sigmoid'))
#	model.add(Dense(5, activation='Sigmoid'))
#	model.add(Dense(1))
#	model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
#	model.fit(X_train, Y_train)




		print 'input_dim', X_train.shape[1]



#	BUILD MODEL
		model = tf.keras.models.Sequential([
		layers.Dense(8,input_dim=X_train.shape[1], activation=tf.nn.tanh),
		layers.Dense(8,activation=tf.nn.tanh),
		layers.Dense(1)
		])

		model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_absolute_error','mean_squared_error'])
	
#	model.summary()

#	TRAIN THE MODEL

		EPOCHS = 2000

		model.fit(X_train, Y_train, epochs=EPOCHS,validation_split=0.2, verbose=0)
	


#	param_grid = {"alpha": [l
#                                         for l in np.logspace(-13, 100, 10)],
#                              "kernel": [RBF(l)
#                                         for l in np.logspace(-5, 100, 10)]}
#        model = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)





# Get best parameter
#	cv_score=model.best_score_
#	best_alpha=model.best_model_.alpha
#	best_gamma=model.best_model_.gamma

# Get the prediction for train and test	
		Y_test_predict = model.predict(X_test)
		Y_train_predict= model.predict(X_train)
#	print Y_test
#	print Y_test_predict


# Get the prediction for the INPUT	
#	prediction=model.predict(INPUT)

# calculate train error
        	mse1 = mean_squared_error(Y_train, Y_train_predict)
        	ISE = np.sqrt(mse1)
# calculate test error
        	mse2 = mean_squared_error(Y_test, Y_test_predict)
        	OSE = np.sqrt(mse2)

#	print 'best alpha=', best_alpha
#	print 'best gamma=', best_gamma

# print information to screen
		if OSE < ISE:
			print colored('                            OSE < ISE','blue')
		print 'ISE=',colored(ISE,'red'),'OSE=',colored(OSE,'red'),colored(abs(OSE-ISE),'blue')
		print "predict on input"
#	print colored(prediction,'green')



# add data to the list
		OSE_csv.append(OSE)
		ISE_csv.append(ISE)
#	cver_csv.append(cv_score)
#	pred_csv.append(prediction)




	OSE_mean=np.mean(OSE_csv)
	ISE_mean=np.mean(ISE_csv)
	OSE_std=np.std(OSE_csv)
	ISE_std=np.std(ISE_csv)


	

# 	Add value to the list
	
	
	train_ratio_csv.append(train_total_ratio)
	ose_mean_csv.append(OSE_mean)
	ise_mean_csv.append(ISE_mean)
	ose_std_csv.append(OSE_std)
	ise_std_csv.append(ISE_std)
	print 'ratio:',train_total_ratio,"-----------------------------------------------"
	print colored('ose_mean','red'),colored(OSE_mean,'red'),'ose_std:',OSE_std,'ise_mean',ISE_mean,'iSE_std:',ISE_std

#	Make csv file.

rows = zip(train_ratio_csv,ose_mean_csv,ose_std_csv,ise_mean_csv,ise_std_csv)
with open('learning_curve_ed.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for row in rows:
        wr.writerow(row)



