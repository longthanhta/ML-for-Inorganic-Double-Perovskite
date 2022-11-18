# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:03:42 2019

@author: Long
"""
import xlsxwriter
import pandas as pd
import csv


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

df_in = pd.read_excel('data.xlsx', sheet_name='data')


#Creat library of compound name
A_lib=['Cs']
B_lib=['']

formula_list=[]
A_list=[]
B1_list=[]
B2_list=[]
X_list=[]
Eg_list=[]
Has_bs_list=[]
for i in range(0,len(df_in['Formula'])):
    f=df_in['Formula'][i]
 
    if 'Cl' in f or 'Br' in f or "I" in f or "F" in f:
        if not 'O' in f:
            if '2' in f:
                formula_list.append(f)
                Eg_list.append(df_in['Band Gap (eV)'][i])
                Has_bs_list.append(df_in['Has Bandstructure'][i])
                
                
df_out = pd.DataFrame.from_dict({'System':formula_list,'formula':formula_list,'Band_Gap':Eg_list,'Has Bandstructure':Has_bs_list})
df_out.to_excel('data_2.xlsx', header=True, index=False)    
#    if len(formula) <= 10:
#        continue
#    formula_list.append(formula)

'''    
#    print (formula)
    A=formula[0:2]
    A_list.append(A)
#    print (i,formula[-4:])
    
    
    if not hasNumbers(df_in['System'][i][3:5]):
        B1_list.append(df_in['System'][i][3:5])
    elif not hasNumbers(df_in['System'][i][3:4]):
        B1_list.append(df_in['System'][i][3:4])
        
    if not hasNumbers(df_in['System'][i][6:8]):
        B2_list.append(df_in['System'][i][6:8])
    elif not hasNumbers(df_in['System'][i][5:7]):
        B2_list.append(df_in['System'][i][5:7])
    elif not hasNumbers(df_in['System'][i][6:7]):
        B2_list.append(df_in['System'][i][6:7])
    elif not hasNumbers(df_in['System'][i][5:6]):
        B2_list.append(df_in['System'][i][5:6])        

    print(df_in['System'][i][-4:][0:2])
    if not hasNumbers(df_in['System'][i][-4:][0:2]):
        X_list.append(df_in['System'][i][-4:][0:2])
    elif not hasNumbers(df_in['System'][i][-3:][0:1]):
        X_list.append(df_in['System'][i][-3:][0:1])
        

#print (X_list)
#for i in range (0,len(A_list)):
#    print (A_list[i],B1_list[i],B2_list[i],X_list[i])
#rows = zip(formula_list,A_list,B1_list,B2_list,X_list)
#with open('result.csv', 'w') as myfile:
#    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#    wr.writerow(("formula","A","B1","B2","X",))
#    for row in rows:
#        wr.writerow(row)

df_out = pd.DataFrame.from_dict({'System':formula_list,'A':A_list,'B1':B1_list,'B2':B1_list,'B2':B2_list,'X':X_list})
df_out.to_excel('result.xlsx', header=True, index=False)
'''