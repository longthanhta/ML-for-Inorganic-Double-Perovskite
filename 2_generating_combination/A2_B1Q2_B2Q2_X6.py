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
import math
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

#=======================[DATA]=================================================


#print (len(formula_input))

formula_list=[]
A_list=[]
B1_list=[]
B2_list=[]
X_input=[]

ir_A_list=[]
rs_A_list=[]
rp_A_list=[]
rd_A_list=[]
Xc_A_list=[]
e_a_A_list=[]
i_p_A_list=[]
h_o_a_l_A_list=[]
l_u_a_l_A_list=[]


ir_B1Q2_list=[]
rs_B1Q2_list=[]
rp_B1Q2_list=[]
rd_B1Q2_list=[]
Xc_B1Q2_list=[]
e_a_B1Q2_list=[]
i_p_B1Q2_list=[]
h_o_a_l_B1Q2_list=[]
l_u_a_l_B1Q2_list=[]


ir_B2Q2_list=[]
rs_B2Q2_list=[]
rp_B2Q2_list=[]
rd_B2Q2_list=[]
Xc_B2Q2_list=[]
e_a_B2Q2_list=[]
i_p_B2Q2_list=[]
h_o_a_l_B2Q2_list=[]
l_u_a_l_B2Q2_list=[]


ir_X_list=[]
rs_X_list=[]
rp_X_list=[]
rd_X_list=[]
Xc_X_list=[]
e_a_X_list=[]
i_p_X_list=[]
h_o_a_l_X_list=[]
l_u_a_l_X_list=[]


# Loat features data==========================================================

df_A_features = pd.read_excel('descriptors_final.xlsx', sheet_name='A')
df_B1Q2_features = pd.read_excel('descriptors_final.xlsx', sheet_name='B_Q2')
df_B2Q2_features = pd.read_excel('descriptors_final.xlsx', sheet_name='B_Q2')
df_X_features = pd.read_excel('descriptors_final.xlsx', sheet_name='X')


# For A BQ1 BQ3 X6============================================================
# Get the list of Element
A_list=[]
B1Q2_list=[]
B2Q2_list=[]
X_list=[]


# Prepare features for A
for index, row in df_A_features.iterrows():
    A_list.append(format(row['Symb']))
    ir_A_list.append(format(row['i_r']))
    rs_A_list.append(format(row['rs']))
    rp_A_list.append(format(row['rp']))
    rd_A_list.append(format(row['rd']))
    Xc_A_list.append(format(row['Xc']))
    e_a_A_list.append(format(row['e_a']))
    i_p_A_list.append(format(row['i_p']))
    h_o_a_l_A_list.append(format(row['h_o_a_l']))
    l_u_a_l_A_list.append(format(row['l_u_a_l']))


# Prepare features for B1Q2
for index, row in df_B1Q2_features.iterrows():
    B1Q2_list.append(format(row['Symb']))
    ir_B1Q2_list.append(format(row['i_r']))
    rs_B1Q2_list.append(format(row['rs']))
    rp_B1Q2_list.append(format(row['rp']))
    rd_B1Q2_list.append(format(row['rd']))
    Xc_B1Q2_list.append(format(row['Xc']))
    e_a_B1Q2_list.append(format(row['e_a']))
    i_p_B1Q2_list.append(format(row['i_p']))
    h_o_a_l_B1Q2_list.append(format(row['h_o_a_l']))
    l_u_a_l_B1Q2_list.append(format(row['l_u_a_l']))
    
# Prepare features for B2Q2
    
for index, row in df_B2Q2_features.iterrows():
    B2Q2_list.append(format(row['Symb']))
    ir_B2Q2_list.append(format(row['i_r']))
    rs_B2Q2_list.append(format(row['rs']))
    rp_B2Q2_list.append(format(row['rp']))
    rd_B2Q2_list.append(format(row['rd']))
    Xc_B2Q2_list.append(format(row['Xc']))
    e_a_B2Q2_list.append(format(row['e_a']))
    i_p_B2Q2_list.append(format(row['i_p']))
    h_o_a_l_B2Q2_list.append(format(row['h_o_a_l']))
    l_u_a_l_B2Q2_list.append(format(row['l_u_a_l']))

# Prepare features for X

for index, row in df_X_features.iterrows():
    X_list.append(format(row['Symb']))
    ir_X_list.append(format(row['i_r']))
    rs_X_list.append(format(row['rs']))
    rp_X_list.append(format(row['rp']))
    rd_X_list.append(format(row['rd']))
    Xc_X_list.append(format(row['Xc']))
    e_a_X_list.append(format(row['e_a']))
    i_p_X_list.append(format(row['i_p']))
    h_o_a_l_X_list.append(format(row['h_o_a_l']))
    l_u_a_l_X_list.append(format(row['l_u_a_l']))    


#Genrating database============================================================    
    
    
formula_list=[]
tf_list_final=[]
A_list_final=[]
ir_A_list_final=[]
rs_A_list_final=[]
rp_A_list_final=[]
rd_A_list_final=[]
Xc_A_list_final=[]
e_a_A_list_final=[]
i_p_A_list_final=[]
h_o_a_l_A_list_final=[]
l_u_a_l_A_list_final=[]

B1Q2_list_final=[]
ir_B1Q2_list_final=[]
rs_B1Q2_list_final=[]
rp_B1Q2_list_final=[]
rd_B1Q2_list_final=[]
Xc_B1Q2_list_final=[]
e_a_B1Q2_list_final=[]
i_p_B1Q2_list_final=[]
h_o_a_l_B1Q2_list_final=[]
l_u_a_l_B1Q2_list_final=[]


B2Q2_list_final=[]
ir_B2Q2_list_final=[]
rs_B2Q2_list_final=[]
rp_B2Q2_list_final=[]
rd_B2Q2_list_final=[]
Xc_B2Q2_list_final=[]
e_a_B2Q2_list_final=[]
i_p_B2Q2_list_final=[]
h_o_a_l_B2Q2_list_final=[]
l_u_a_l_B2Q2_list_final=[]

X_list_final=[]
ir_X_list_final=[]
rs_X_list_final=[]
rp_X_list_final=[]
rd_X_list_final=[]
Xc_X_list_final=[]
e_a_X_list_final=[]
i_p_X_list_final=[]
h_o_a_l_X_list_final=[]
l_u_a_l_X_list_final=[]


print(len(X_list))


for i in range(0,len(A_list)):
    for j in range(0,len(B1Q2_list)):
        for k in range(0,len(B2Q2_list)):
            for l in range(0,len(X_list)):
                formula=A_list[i]+'2'+B1Q2_list[j]+B2Q2_list[k]+X_list[l]+'6'
                #print(formula)
                formula_list.append(formula)
                
                A_list_final.append(A_list[i])
                ir_A_list_final.append(ir_A_list[i])
                rs_A_list_final.append(rs_A_list[i])
                rp_A_list_final.append(rp_A_list[i])
                rd_A_list_final.append(rd_A_list[i])
                Xc_A_list_final.append(Xc_A_list[i])
                e_a_A_list_final.append(e_a_A_list[i])
                i_p_A_list_final.append(i_p_A_list[i])
                h_o_a_l_A_list_final.append(h_o_a_l_A_list[i])
                l_u_a_l_A_list_final.append(l_u_a_l_A_list[i])

                

                B1Q2_list_final.append(B1Q2_list[j])
                ir_B1Q2_list_final.append(ir_B1Q2_list[j])
                rs_B1Q2_list_final.append(rs_B1Q2_list[j])
                rp_B1Q2_list_final.append(rp_B1Q2_list[j])
                rd_B1Q2_list_final.append(rd_B1Q2_list[j])
                Xc_B1Q2_list_final.append(Xc_B1Q2_list[j])
                e_a_B1Q2_list_final.append(e_a_B1Q2_list[j])
                i_p_B1Q2_list_final.append(i_p_B1Q2_list[j])
                h_o_a_l_B1Q2_list_final.append(h_o_a_l_B1Q2_list[j])
                l_u_a_l_B1Q2_list_final.append(l_u_a_l_B1Q2_list[j])
                
                B2Q2_list_final.append(B2Q2_list[k])                
                ir_B2Q2_list_final.append(ir_B2Q2_list[k])
                rs_B2Q2_list_final.append(rs_B2Q2_list[k])
                rp_B2Q2_list_final.append(rp_B2Q2_list[k])
                rd_B2Q2_list_final.append(rd_B2Q2_list[k])
                Xc_B2Q2_list_final.append(Xc_B2Q2_list[k])
                e_a_B2Q2_list_final.append(e_a_B2Q2_list[k])
                i_p_B2Q2_list_final.append(i_p_B2Q2_list[k])
                h_o_a_l_B2Q2_list_final.append(h_o_a_l_B2Q2_list[k])
                l_u_a_l_B2Q2_list_final.append(l_u_a_l_B2Q2_list[k])

                X_list_final.append(X_list[l])                
                ir_X_list_final.append(ir_X_list[l])
                rs_X_list_final.append(rs_X_list[l])
                rp_X_list_final.append(rp_X_list[l])
                rd_X_list_final.append(rd_X_list[l])
                Xc_X_list_final.append(Xc_X_list[l])
                e_a_X_list_final.append(e_a_X_list[l])
                i_p_X_list_final.append(i_p_X_list[l])
                h_o_a_l_X_list_final.append(h_o_a_l_X_list[l])
                l_u_a_l_X_list_final.append(l_u_a_l_X_list[l])
                
                
                
                tf=float((float(ir_A_list[i])+float(ir_X_list[l])))/float((math.sqrt(2)*(float(float(ir_B1Q2_list[j])/2)+float(float(ir_B2Q2_list[k])/2)+float(ir_X_list[l]))))
                tf_list_final.append(tf)
                

rows = zip(formula_list,A_list_final,B1Q2_list_final,B2Q2_list_final,X_list_final,tf_list_final,ir_A_list_final,rs_A_list_final,
           rp_A_list_final,rd_A_list_final,Xc_A_list_final,e_a_A_list_final,i_p_A_list_final,
           h_o_a_l_A_list_final,l_u_a_l_A_list_final,ir_B1Q2_list_final,rs_B1Q2_list_final,
           rp_B1Q2_list_final,rd_B1Q2_list_final,Xc_B1Q2_list_final,e_a_B1Q2_list_final,
           i_p_B1Q2_list_final,h_o_a_l_B1Q2_list_final,l_u_a_l_B1Q2_list_final,
           ir_B2Q2_list_final,rs_B2Q2_list_final,rp_B2Q2_list_final,rd_B2Q2_list_final,
           Xc_B2Q2_list_final,e_a_B2Q2_list_final,i_p_B2Q2_list_final,h_o_a_l_B2Q2_list_final,
           l_u_a_l_B2Q2_list_final,ir_X_list_final,rs_X_list_final,rp_X_list_final,
           rd_X_list_final,Xc_X_list_final,e_a_X_list_final,i_p_X_list_final,h_o_a_l_X_list_final,l_u_a_l_X_list_final)
with open('result_A2_B1Q2_B2Q2_X6.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(('formula','A','B1','B2','X','tolerance_factor','irA','rsA',
           'rpA','rdA','XcA','eaA','ipA',
           'hoalA','lualA','irB1','rsB1',
           'rpB1','rdB1','XcB1','eaB1',
           'ipB1','hoalB1','lualB1',
           'irB2','rsB2','rpB2','rdB2',
           'XcB2','eaB2','ipB2','hoalB2',
           'lualB2','irX','rsX','rpX',
           'rdX','XcX','eaX','ipX','hoalX','lualX'))
    for row in rows:
        wr.writerow(row)
