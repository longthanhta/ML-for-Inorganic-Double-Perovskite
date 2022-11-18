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

df_in = pd.read_csv('NPJ2019.csv')

#Creat library of compound name
A_lib=['Cs']
B_lib=['']

formula_list=[]
A_list=[]
B1_list=[]
B2_list=[]
X_list=[]
Eg_list=[]
for i in range(0,len(df_in['id'])):
    A_list.append(df_in['a_atom'][i])
    B1_list.append(df_in['b1_atom'][i])
    B2_list.append(df_in['b2_atom'][i])
    X_list.append(df_in['x_atom'][i])
    formula=df_in['a_atom'][i]+'2'+df_in['b1_atom'][i]+df_in['b2_atom'][i]+df_in['x_atom'][i]+'6'
    formula_list.append(formula)
    Eg_list.append(df_in['ind_gap'][i])
df_out = pd.DataFrame.from_dict({'System':formula_list,'A':A_list,'B1':B1_list,'B1':B1_list,'B2':B2_list,'X':X_list,'Band_gap':Eg_list})
df_out.to_excel('result.xlsx', header=True, index=False)
