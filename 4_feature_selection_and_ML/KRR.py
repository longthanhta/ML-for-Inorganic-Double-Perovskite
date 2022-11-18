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
from sklearn import preprocessing
from sklearn import linear_model
#=============================================================================

#=======================[DATA]=======================================================================================
# Load data

descriptor_list=[
		'tolerance_factor',
#		'irA',
#		'rsA',
#		'rpA',
#		'rdA',
#		'XcA',
#		'eaA',
#		'ipA',
#		'hoalA',
#		'lualA',
#		'irB1',
		'rsB1',
#		'rpB1',
		'rdB1',
		'XcB1',
		'eaB1',
#		'ipB1',
		'hoalB1',
		'lualB1',
		'irB2',
		'rsB2',
		'rpB2',
		'rdB2',
		'XcB2',
		'eaB2',
		'ipB2',
#		'hoalB2',
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

df1 = pd.read_csv('data_and_descriptors_final.csv')
X = df1[descriptor_list].astype(float)
Y = df1['Eg'].astype(float)

#print(X1_in)
#print(X2_in)


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

# Data preprocessing

for col in X.columns:
    max=X[col].max()
    for i in range(0,X.shape[0]):
        if X[col][i] != 0:
            X[col][i]=X[col][i]/max
#print(X_in)

	
for x in range(total_run_no):
    print ('=================================================================')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    param_grid={"alpha":np.logspace(-5,5,10),"gamma":np.logspace(-5,5,10)}
    Y_average=np.mean(Y)
    kernel=RBF(length_scale=1)
    estimator=GridSearchCV(KernelRidge(kernel=kernel),cv=5,
                           param_grid=param_grid)
    estimator.fit(X_train,Y_train)	

# Get best parameter
    cv_score=estimator.best_score_
    best_alpha=estimator.best_estimator_.alpha
    best_gamma=estimator.best_estimator_.gamma


    Y_test_predict = estimator.predict(X_test)
#    print ('prediction by test', Y_test_predict)



# Get the prediction for train and test	
    Y_test_predict = estimator.predict(X_test)
    Y_train_predict= estimator.predict(X_train)
#	print Y_test
#	print Y_test_predict

    
    
# calculate train error and test error
    mse1 = mean_squared_error(Y_train, Y_train_predict)
    mse2 = mean_squared_error(Y_test, Y_test_predict)
    ISE = np.sqrt(mse1)
    OSE = np.sqrt(mse2)

    print ('best alpha=', best_alpha)
    print ('best gamma=', best_gamma)

# print information to screen
    if OSE < ISE:
        print('OSE < ISE')
        print ('ISE=',ISE,'OSE=',OSE)
#	print ("predict on input")
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
    xline = np.linspace(0, 7, 1000)
    plt.plot(xline,xline,color='black')
    plt.legend(loc='upper left')
    plt.text(3, 1,'train error=')
    plt.text(4.5, 1, ISE)
    plt.text(3, 1.5,'test error=')
    plt.text(4.5, 1.5, OSE)
    plt.xlabel('real (DFT)')
    plt.ylabel('predicted ')
    filename=str(x+1)+".pdf"	
    fig.savefig(filename)

# Make file record what data point has been use
#	X_train.to_csv(str(x)+'X_train.csv')
#	Y_train.to_csv(str(x)+'Y_train.csv')
#	X_test.to_csv(str(x)+'X_test.csv')
#	Y_test.to_csv(str(x)+'Y_test.csv')
#	np.savetxt(str(x)+"Y_test_predicted.csv", Y_test_predict, delimiter=",")

	

#	Make csv file.


#rows = zip(run_no,ba_csv,ose_csv,ise_csv,cver_csv,pred_csv)
rows = zip(run_no,ba_csv,bg_csv,ose_csv,ise_csv,cver_csv)
with open('result.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(("runo","best alpha","best gamma","ose","ise","cve"))
    for row in rows:
        wr.writerow(row)



