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
import time
from tqdm import tqdm

#=======================[DATA]=================================================
# Load data
df_input = pd.read_excel('Dawn_single_perovskite_data.xlsx', sheet_name='Sheet1')
Eg=df_input[['Band_gap']]
formula_list=[]
Eg_list=[]
A_list=[]
B1_list=[]
B2_list=[]
X_list=[]

for index, row in df_input.iterrows():
    formula_list.append(row['System'])
    A_list.append(row['A'])
    B1_list.append(row['B1'])    
    B2_list.append(row['B2'])
    X_list.append(row['X'])    
    Eg_list.append(row['Band_gap'])
    
for i in range (0,len(formula_list)):
    print (formula_list[i],Eg_list[i])
    
# Load Descriptor data

df_features = pd.read_csv('final_A2_B1_B2_X6.csv')



# Append Descriptors


formula_list_final=[]
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

B1_list_final=[]
ir_B1_list_final=[]
rs_B1_list_final=[]
rp_B1_list_final=[]
rd_B1_list_final=[]
Xc_B1_list_final=[]
e_a_B1_list_final=[]
i_p_B1_list_final=[]
h_o_a_l_B1_list_final=[]
l_u_a_l_B1_list_final=[]


B2_list_final=[]
ir_B2_list_final=[]
rs_B2_list_final=[]
rp_B2_list_final=[]
rd_B2_list_final=[]
Xc_B2_list_final=[]
e_a_B2_list_final=[]
i_p_B2_list_final=[]
h_o_a_l_B2_list_final=[]
l_u_a_l_B2_list_final=[]

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

Eg_list_final=[]


for index, row in df_features.iterrows():
#    print(str(index)+'/'+str(df_features.shape[0]))
    for i in range(0,len(formula_list)):
#        print (format(row['A']),A_list[i],format(row['B1']),B1_list[i],format(row['B2']),B2_list[i],format(row['X']),X_list[i])
        if format(row['A'])==A_list[i] and format(row['B1'])==B1_list[i] and format(row['B2'])==B2_list[i] and format(row['X'])==X_list[i] :
            formula=A_list[i]+'2'+B1_list[i]+X_list[i]+'3'
            formula_list_final.append(formula)
            A_list_final.append(format(row['A']))
            ir_A_list_final.append(format(row['irA']))
            rs_A_list_final.append(format(row['rsA']))
            rp_A_list_final.append(format(row['rpA']))
            rd_A_list_final.append(format(row['rdA']))
            Xc_A_list_final.append(format(row['XcA']))
            e_a_A_list_final.append(format(row['eaA']))
            i_p_A_list_final.append(format(row['ipA']))
            h_o_a_l_A_list_final.append(format(row['hoalA']))
            l_u_a_l_A_list_final.append(format(row['lualA']))

                

            B1_list_final.append(format(row['B1']))
            ir_B1_list_final.append(format(row['irB1']))
            rs_B1_list_final.append(format(row['rsB1']))
            rp_B1_list_final.append(format(row['rpB1']))
            rd_B1_list_final.append(format(row['rdB1']))
            Xc_B1_list_final.append(format(row['XcB1']))
            e_a_B1_list_final.append(format(row['eaB1']))
            i_p_B1_list_final.append(format(row['ipB1']))
            h_o_a_l_B1_list_final.append(format(row['hoalB1']))
            l_u_a_l_B1_list_final.append(format(row['lualB1']))
                
            B2_list_final.append(format(row['B2']))                
            ir_B2_list_final.append(format(row['irB2']))
            rs_B2_list_final.append(format(row['rsB2']))
            rp_B2_list_final.append(format(row['rpB2']))
            rd_B2_list_final.append(format(row['rdB2']))
            Xc_B2_list_final.append(format(row['XcB2']))
            e_a_B2_list_final.append(format(row['eaB2']))
            i_p_B2_list_final.append(format(row['ipB2']))
            h_o_a_l_B2_list_final.append(format(row['hoalB2']))
            l_u_a_l_B2_list_final.append(format(row['lualB2']))

            X_list_final.append(format(row['X']))                
            ir_X_list_final.append(format(row['irX']))
            rs_X_list_final.append(format(row['rsX']))
            rp_X_list_final.append(format(row['rpX']))
            rd_X_list_final.append(format(row['rdX']))
            Xc_X_list_final.append(format(row['XcX']))
            e_a_X_list_final.append(format(row['eaX']))
            i_p_X_list_final.append(format(row['ipX']))
            h_o_a_l_X_list_final.append(format(row['hoalX']))
            l_u_a_l_X_list_final.append(format(row['lualX']))
            tf_list_final.append(format(row['tolerance_factor']))
            Eg_list_final.append(Eg_list[i])
            
            
            
print(len(formula_list_final))
print(len(A_list_final))
print(len(B1_list_final))
print(len(B2_list_final))
print(len(X_list_final))
print(len(tf_list_final))
print(len(Eg_list_final))
print(len(ir_A_list_final))
print(len(rs_A_list_final))
print(len(rp_A_list_final))
print(len(rd_A_list_final))
print(len(Xc_A_list_final))
print(len(e_a_A_list_final))
print(len(i_p_A_list_final))
print(len(h_o_a_l_A_list_final))
print(len(l_u_a_l_A_list_final))
print(len(ir_B1_list_final))
print(len(rs_B1_list_final))
print(len(rp_B1_list_final))
print(len(rd_B1_list_final))
print(len(Xc_B1_list_final))
print(len(e_a_B1_list_final))
print(len(i_p_B1_list_final))
print(len(h_o_a_l_B1_list_final))
print(len(l_u_a_l_B1_list_final))
print(len(ir_B2_list_final))
print(len(rs_B2_list_final))
print(len(rp_B2_list_final))
print(len(rd_B2_list_final))
print(len(Xc_B2_list_final))
print(len(e_a_B2_list_final))
print(len(i_p_B2_list_final))
print(len(h_o_a_l_B2_list_final))
print(len(l_u_a_l_B2_list_final))
print(len(ir_X_list_final))
print(len(rs_X_list_final))
print(len(rp_X_list_final))
print(len(rd_X_list_final))
print(len(Xc_X_list_final))
print(len(e_a_X_list_final))
print(len(i_p_X_list_final))
print(len(h_o_a_l_X_list_final))
print(len(l_u_a_l_X_list_final))
            
rows = zip(formula_list_final,A_list_final,B1_list_final,B2_list_final,X_list_final,tf_list_final,Eg_list_final,ir_A_list_final,rs_A_list_final,
           rp_A_list_final,rd_A_list_final,Xc_A_list_final,e_a_A_list_final,i_p_A_list_final,
           h_o_a_l_A_list_final,l_u_a_l_A_list_final,ir_B1_list_final,rs_B1_list_final,
           rp_B1_list_final,rd_B1_list_final,Xc_B1_list_final,e_a_B1_list_final,
           i_p_B1_list_final,h_o_a_l_B1_list_final,l_u_a_l_B1_list_final,
           ir_B2_list_final,rs_B2_list_final,rp_B2_list_final,rd_B2_list_final,
           Xc_B2_list_final,e_a_B2_list_final,i_p_B2_list_final,h_o_a_l_B2_list_final,
           l_u_a_l_B2_list_final,ir_X_list_final,rs_X_list_final,rp_X_list_final,
           rd_X_list_final,Xc_X_list_final,e_a_X_list_final,i_p_X_list_final,h_o_a_l_X_list_final,l_u_a_l_X_list_final)
with open('single_perovskite_data_and_descriptors_final.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(('formula','A','B1','B2','X','tolerance_factor','Eg','irA','rsA',
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
            

    
