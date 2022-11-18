from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from termcolor import colored
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import sklearn
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

#=======================[DATA]=======================================================================================
# Load data
df1 = pd.read_excel('dataHSE_addGGA.xlsx', sheet_name='random')
X = df1[['B',
	'X',
	'OCT-PC1',
	'OCT-PC2',
	'OCT-PC3',
	'A-PC1',
	'A-PC2',
	'A-PC3',
	'Angles_mean',
	'Angles.std',
	'GGA',
	]].astype(float)
Y = df1['eg'].astype(float)

df2 = pd.read_excel('dataHSE_addGGA.xlsx', sheet_name='must_have')
X_train_2 = df2[['B',
	'X',
	'OCT-PC1',
	'OCT-PC2',
	'OCT-PC3',
	'A-PC1',
	'A-PC2',
	'A-PC3',
	'Angles_mean',
	'Angles.std',
	'GGA',
	]].astype(float)
Y_train_2 = df2['eg'].astype(float)

#print 'X.shape',X.shape
#print 'X_train_2',X_train_2.shape

#print 'X.shape',X.shape
#print 'X_train_2',X_train_2.shape

#=======================[/DATA]=======================================================================================


total_run_no=10


#	Create empty list

run_no=[]
ose_csv=[]
ise_csv=[]
cver_csv=[]
pred_csv=[]
ba_csv=[]
bg_csv=[]

	
for x in range(total_run_no):
	print ('=================================================================')
	X_train_1, X_test, Y_train_1, Y_test = train_test_split(X, Y, test_size=0.2)

#	add "must have data"


	frames1 = [X_train_1, X_train_2]
	X_train = pd.concat(frames1)

	frames2 = [Y_train_1, Y_train_2]
	Y_train = pd.concat(frames2)


#	print 'X_train',X_train.shape


# Train model
#	param_grid = {"alpha": np.logspace(-5, 5, 11),
#               "gamma": np.logspace(-5, 5, 11)}
#	estimator = GridSearchCV(KernelRidge(kernel='rbf',alpha=0.1,gamma=0.1), cv=5,
#  	param_grid=param_grid)





	kernel=RBF(length_scale=5)
	param_grid = {"alpha": np.logspace(-5, 5, 15),
               "gamma": np.logspace(-5, 5, 15)}
	estimator = GridSearchCV(KernelRidge(kernel='rbf'), cv=5,
  	param_grid=param_grid)
	estimator.fit(X_train, Y_train)	

# Get best parameter
	cv_score=estimator.best_score_
	best_alpha=estimator.best_estimator_.alpha
	best_gamma=estimator.best_estimator_.gamma



#==============DO THE PREDICTION AGAIN=================================================

	estimator = KernelRidge(kernel=kernel,alpha=best_alpha,gamma=best_gamma)
	estimator.fit(X_train, Y_train)

	print (estimator.dual_coef_)
	duel_coef=estimator.dual_coef_
	dists = cdist(X_test, X_train,metric='sqeuclidean')
	print ('duel_coef', duel_coef)




	print (estimator.score(X_train, Y_train))
	print (sklearn.gaussian_process.kernels)

#============ CALCULATE KERNEL ========================================================
	Y_test_predict = estimator.predict(X_test)
	print ('prediction by test', Y_test_predict)


	dists = cdist(X_test, X_train, metric='euclidean')
	K = np.exp(-0.5*dists**2)
	K = np.squeeze(np.asarray(K))
	duel_coef = np.squeeze(np.asarray(duel_coef))
	print ('prediction by hand', np.dot(K, duel_coef))


# Get the prediction for train and test	
	Y_test_predict = estimator.predict(X_test)
	Y_train_predict= estimator.predict(X_train)
#	print Y_test
#	print Y_test_predict



    
# Get the prediction for the INPUT	
#	prediction=estimator.predict(INPUT)

# calculate train error and test error
	mse1 = mean_squared_error(Y_train, Y_train_predict)
	mse2 = mean_squared_error(Y_test, Y_test_predict)
	ISE = np.sqrt(mse1)
	OSE = np.sqrt(mse2)

	print ('best alpha=', best_alpha)
	print ('best gamma=', best_gamma)

# print information to screen
	if OSE < ISE:
		print(colored('OSE < ISE','blue'))
	print ('ISE=',colored(ISE,'red'),'OSE=',colored(OSE,'red'),colored(abs(OSE-ISE),'blue'))
	print ("predict on input")
#	print colored(prediction,'green')



# add data to the list
	ose_csv.append(OSE)
	ise_csv.append(ISE)
	cver_csv.append(cv_score)
#	pred_csv.append(prediction)
	run_no.append(x)
	ba_csv.append(best_alpha)
	bg_csv.append(best_gamma)


#	Make plot file for each run
	train_prediction=estimator.predict(X_train)
	fig = plt.figure()
	plt.scatter(Y_train, Y_train_predict,marker='^',color='b',facecolors='b',label='train data',s=23)
	plt.scatter(Y_test, Y_test_predict,marker='o',color='r',facecolors='r',label='test data',s=23)
#	xline = np.linspace(-5.5, -3.5, 1000)
	xline = np.linspace(1, 7, 1000)
	plt.plot(xline,xline,color='black')
	plt.legend(loc='upper left')
	plt.text(3, 1,'train error=')
	plt.text(4.5, 1, ISE)
	plt.text(3, 1.5,'test error=')
	plt.text(4.5, 1.5, OSE)
	plt.xlabel('real (DFT)')
	plt.ylabel('predicted ')
	filename=str(x)+".pdf"	
	fig.savefig(filename)

# Make file record what data point has been use
	np.savetxt(str(x)+"duel_coef.csv", duel_coef, delimiter=",")
	X_train.to_csv(str(x)+'X_train.csv')
	Y_train.to_csv(str(x)+'Y_train.csv')
	X_test.to_csv(str(x)+'X_test.csv')
	Y_test.to_csv(str(x)+'Y_test.csv')
	np.savetxt(str(x)+"Y_test_predicted.csv", Y_test_predict, delimiter=",")

	

#	Make csv file.


#rows = zip(run_no,ba_csv,ose_csv,ise_csv,cver_csv,pred_csv)
rows = zip(run_no,ba_csv,bg_csv,ose_csv,ise_csv,cver_csv)
with open('result.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(("runo","best alpha","best gamma","ose","ise","cve"))
    for row in rows:
        wr.writerow(row)



