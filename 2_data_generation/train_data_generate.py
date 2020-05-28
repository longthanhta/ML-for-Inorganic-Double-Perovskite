import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt


# Generate descriptor list
MP_des_list=["formula","A","B1","B2","X","Eg","F_E","density","volume"]
element_des_list=['q','label_IP','label_HOMO','label_preodict','i_r','i_r_s','i_r_p','i_r_d','i_r_f','EN','EA','sum_s_p_r','r_s','r_p','r_d','r_f','IP','IP_1','HOMO','LUMO','L','S','J','val_e','weight']
element_des_list_A=[des + '_A' for des in element_des_list]
element_des_list_B1=[des + '_B1' for des in element_des_list]
element_des_list_B2=[des + '_B2' for des in element_des_list]
element_des_list_X=[des + '_X' for des in element_des_list]
structure_des_list=['B1X_bond','B2X_bond','AX_bond','heat_formation','Elasticity_bulk','Elasticity_shear']

# Get material project data
df1_input = pd.read_csv('1_mp_yaocai_eg_fe_ehull_elastic_dens_vol.csv')

# Get bond energies data
df_be=pd.read_csv('Bond_energy.csv')
BE=df_be['Energy']

# Get Heat formation data
df_hf=pd.read_csv('Heat_formation.csv')
# Get elasticity data
df_el=pd.read_csv('MP_bulk_double.csv')

A_hf=df_hf['a_atom']
B1_hf=df_hf['b1_atom']
B2_hf=df_hf['b2_atom']
X_hf=df_hf['x_atom']
HF=df_hf['heat_of_formation']


A_el=df_el['A']
B1_el=df_el['B1']
B2_el=df_el['B2']
X_el=df_el['X']


MP_data= np.array(df1_input[MP_des_list])

# Get element symbol
MP_A_symbol= np.array(df1_input['A'])
MP_B1_symbol= np.array(df1_input['B1'])
MP_B2_symbol= np.array(df1_input['B2'])
MP_X_symbol= np.array(df1_input['X'])



# Get element data

irA_input = pd.read_excel('element_descriptors.xlsx', sheet_name='A')
A_symbol = np.array(irA_input['Symbol'])
A_des= np.array(irA_input[element_des_list])

irB3_input = pd.read_excel('element_descriptors.xlsx', sheet_name='B_Q3')
B_symbol = np.array(irB3_input['Symbol'])
B3_des= np.array(irB3_input[element_des_list])

X_input = pd.read_excel('element_descriptors.xlsx', sheet_name='X')
X_symbol = np.array(X_input['Symbol'])
X_des= np.array(X_input[element_des_list])

print('A symbol',A_symbol)
print('B symbol',B_symbol)
print('X symbol',X_symbol)


data_length = len(df1_input)
B1X_be_lst=[]
B2X_be_lst=[]
heat_formation=[]
final_frame=[]
for i in range(data_length):
    #if i==3: break
    MP_data_entry=MP_data[i]
    # Find index that match the symbol
    ir_A_index = np.where(A_symbol == MP_A_symbol[i])[0]
    ir_B1_index = np.where(A_symbol == MP_B1_symbol[i])[0]
    ir_B2_index = np.where(B_symbol == MP_B2_symbol[i])[0]
    ir_X_index = np.where(X_symbol == MP_X_symbol[i])[0]

    if len(ir_A_index) == 0 or len(ir_B1_index) == 0 or len(ir_B2_index) == 0 or len(ir_X_index) == 0 :continue

    #print(ir_B2_index)

    A_elmt=MP_A_symbol[i]
    B1_elmt=MP_B1_symbol[i]
    B2_elmt=MP_B2_symbol[i]
    X_elmt=MP_X_symbol[i]




    B1X_bond=B1_elmt+'-'+X_elmt
    B1X_bond_i=X_elmt+'-'+B1_elmt
    if df_be['Bond'].str.contains(B1X_bond).any():
        try:
            B1X_be=(float(BE[df_be['Bond']==B1X_bond]))
        except:
            B1X_be=0
    elif df_be['Bond'].str.contains(B1X_bond_i).any():
        try:
            B1X_be=(float(BE[df_be['Bond']==B1X_bond_i]))
        except:
            B1X_be=0
    else:
        B1X_be=0
    

    B2X_bond=B2_elmt+'-'+X_elmt
    B2X_bond_i=X_elmt+'-'+B2_elmt
    if df_be['Bond'].str.contains(B2X_bond).any():
        try:
            B2X_be=(float(BE[df_be['Bond']==B2X_bond]))
        except:
            B2X_be=0
    elif df_be['Bond'].str.contains(B2X_bond_i).any():
        try:
            B2X_be=(float(BE[df_be['Bond']==B2X_bond_i]))
        except:
            B2X_be=0
    else:
        B2X_be=0



    AX_bond=A_elmt+'-'+X_elmt
    AX_bond_i=X_elmt+'-'+A_elmt
    if df_be['Bond'].str.contains(AX_bond).any():
        try:
            AX_be=(float(BE[df_be['Bond']==AX_bond]))
        except:
            AX_be=0
    elif df_be['Bond'].str.contains(AX_bond_i).any():
        try:
            AX_be=(float(BE[df_be['Bond']==AX_bond_i]))
        except:
            AX_be=0
    else:
        AX_be=0

    #Get heat formation energy:
    heat_formation='NA'
    for j in range(0,len(df_hf)):
        if A_hf[j]==MP_A_symbol[i] and B1_hf[j]==MP_B1_symbol[i] and B2_hf[j]==MP_B2_symbol[i] and X_hf[j]==MP_X_symbol[i]:
            heat_formation=df_hf['heat_of_formation'][j]
            break
    elasticity_bulk='NA'
    elasticity_shear='NA'
    for j in range(0,len(df_el)):
        if A_el[j]==MP_A_symbol[i] and B1_el[j]==MP_B1_symbol[i] and B2_el[j]==MP_B2_symbol[i] and X_el[j]==MP_X_symbol[i]:
            elasticity_bulk=df_el['Elasticity_bulk'][j]
            elasticity_shear=df_el['Elasticity_shear'][j]
            break

    #Additional property
    added_information=np.array([B1X_be,B2X_be,AX_be,heat_formation,elasticity_bulk,elasticity_shear])


    # If match, then append it to final frame
    if len(ir_A_index)>0 and len(ir_B1_index)>0 and len(ir_B2_index)>0 and len(ir_X_index)>0:
        ionA_des_ls=list(A_des[ir_A_index[0]])
        ionB1_des_ls=list(A_des[ir_B1_index[0]])
        ionB2_des_ls=list(B3_des[ir_B2_index[0]])
        ionX_des_ls=list(X_des[ir_X_index[0]])
    try:
        row = np.concatenate((MP_data_entry,ionA_des_ls,ionB1_des_ls,ionB2_des_ls,ionX_des_ls,added_information))
    except:
        continue
    if 'nan' not in str(row):
        #print(row)
        final_frame.append(row)


final_frame = np.array(final_frame)
final_header=MP_des_list+element_des_list_A+element_des_list_B1+element_des_list_B2+element_des_list_X+structure_des_list



with open('2_origin_descriptors_double.csv', 'w',newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow((final_header))
    for row in final_frame:
        wr.writerow(row)
