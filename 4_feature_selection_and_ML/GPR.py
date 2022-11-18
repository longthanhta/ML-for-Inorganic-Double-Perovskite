from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from termcolor import colored
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import sklearn
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn import preprocessing
from sklearn import linear_model
#=============================================================================
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
		'ipB1',
		'hoalB1',
		'lualB1',
		'irB2',
		'rsB2',
#		'rpB2',
		'rdB2',
		'XcB2',
		'eaB2',
		'ipB2',
		'hoalB2',
#		'lualB2',
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
X1_in = df1[descriptor_list].astype(float)
Y1_in = df1['Eg'].astype(float)

df2 = pd.read_csv('single_perovskite_data_and_descriptors_final.csv')
X2_in = df2[descriptor_list].astype(float)
Y2_in = df2['Eg'].astype(float)

#print(X1_in)
#print(X2_in)


# Data preprocessing
frames1 = [X1_in, X2_in]
X_in = pd.concat(frames1,keys=descriptor_list,ignore_index=True, sort=False)
frames2 = [Y1_in, Y2_in]
Y_in = pd.concat(frames2,ignore_index=True, sort=False)

for col in X_in.columns:
    max=X_in[col].max()
    for i in range(0,X_in.shape[0]):
        if X_in[col][i] != 0:
            X_in[col][i]=X_in[col][i]/max
#print(X_in)

tr_l=X1_in.shape[0]

#print('train_length=',tr_l)

X_train_data = pd.DataFrame()
X_predict = pd.DataFrame()
X_train_data = X_in[:tr_l]
X_predict = X_in[tr_l:]



Y_train_data = pd.DataFrame()
Y_predict = pd.DataFrame()
Y_train_data = Y_in[:tr_l]
Y_predict = Y_in[tr_l:]


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
    X_train, X_test, Y_train, Y_test = train_test_split(X_train_data, Y_train_data, test_size=0.2)
    param_grid = {"alpha": np.logspace(-11, 11, 20)}
    Y_average=np.mean(Y_train)
    kernel = Y_average*RBF(length_scale=1) + WhiteKernel(noise_level=0.03)
    estimator = GridSearchCV(GaussianProcessRegressor(kernel=kernel), cv=5,
                         param_grid=param_grid)
    estimator.fit(X_train, Y_train)
# Get best parameter
    cv_score=estimator.best_score_
    best_alpha=estimator.best_estimator_.alpha


    Y_test_predict = estimator.predict(X_test)
#    print ('prediction by test', Y_test_predict)



# Get the prediction for train and test	
    Y_test_predict = estimator.predict(X_test)
    Y_train_predict= estimator.predict(X_train)
#	print Y_test
#	print Y_test_predict



    
# Get the prediction for the INPUT	
    prediction = estimator.predict(X_predict)
    error = abs(prediction - Y_predict)    
    rows = zip(df2['formula'],df2['A'],df2['B1'],df2['B2'],df2['X'],prediction,Y_predict,error)
    with open('predict_result_'+str(x+1)+'.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(("formula",'A','B1','B2','X',"prediction","real data",'error'))
        for row in rows:
           wr.writerow(row)
    
    
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
rows = zip(run_no,ba_csv,ose_csv,ise_csv,cver_csv)
with open('result.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(("runo","best alpha","ose","ise","cve"))
    for row in rows:
        wr.writerow(row)



