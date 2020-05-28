from pymatgen import MPRester
import pymatgen.core.bonds as cb
import pymatgen.analysis.elasticity.elastic as py_elastic
import numpy as np
import pandas as pd
import csv
import math
from ase.io import read,write
from ase import Atoms

A_list=['Ag2','Au2','Cs2','Cu2','Fr2','K2','Li2','Na2','Rb2','Hg2','Tl2']
B1_list=['Ag','Au','Cs','Cu','Fr','K','Li','Na','Rb','Hg','Tl']
B2_list=['Al','As','Bi','Ce','Co','Cr','Dy','Er','Eu','Fe','Ga','Gd','In','La','Lu','Mn','Mo','Nb','Nd','Ni','Pr','Rh','Ru','Sb','Sc','Sm','Ta','Tb','Ti','Tm','V','Y']
X_list=['F6','Br6','Cl6','I6']
def Average(lst): 
	return sum(lst) / len(lst) 

def sort_none_bracket_str(string_formula):
	A=''
	B1=''
	B2=''
	X=''
	print('original formula',string_formula)
	# Get A	
	for elmt in A_list:
		if str(elmt) in string_formula:
			A=elmt[:-1]
			print('A is',A)
			string_formula=string_formula.replace(A,'')
			break
	print('string_formula',string_formula)

	# Get X
	for elmt in X_list:
		if elmt in string_formula:
			X=elmt[:-1]
			print('X is',X)
			string_formula=string_formula.replace(X,'')
			break
	print('string_formula',string_formula)


	# Get B1
	for elmt in B1_list:
		if elmt in string_formula:
			B1=elmt
			print('B1 is',B1)
			string_formula=string_formula.replace(B1,'')
			break
	print('string_formula',string_formula)


	# Get B2
	for elmt in B2_list:
		if elmt in string_formula:
			B2=elmt
			print('B2 is',B2)
			string_formula=string_formula.replace(B2,'')
			break
	print('string_formula',string_formula)

	if len(string_formula) == 2:
		correct_formula=[]
		correct_formula.append(A)
		correct_formula.append(B1)
		correct_formula.append(B2)
		correct_formula.append(X)
		return correct_formula
	else:
		return None


if __name__ == "__main__":
	MAPI_KEY = None  # You must change this to your Materials API key! (or set MAPI_KEY env variable)
	QUERY = "mp-1110834"  # change this to the mp-id of your correct_formula of interest
	# QUERY = "TiO"  # change this to a formula of interest
	# QUERY = "Ti-O"  # change this to a chemical system of interest

#	mpr = MPRester('0bwL1euo5ydiIRfTZ') 
#		bs=m.get_bandstructure_by_material_id(QUERY)
#		bs.get_bandgap()
#	print(mpr.query({'material_id': 'mp-555599'}, ['band_gap']))


df_input = pd.read_csv('1_MP_ID_updated_only_A2BBX6.csv')
ID_list=df_input['Material ID']
m = MPRester('0bwL1euo5ydiIRfTZ')


# Creat empty list to append
final_frame=[]


data_length=len(ID_list)
for i in range(data_length):
	#if i>3: break
	print(i+1,'/',data_length) 
	info = m.get_data(ID_list[i]) #get data from material project API
	print("ID_list[i]",ID_list[i]) #get ID list from material project (could be updated)
	formula=info[0]['pretty_formula'] #get string formula
	correct_formula = sort_none_bracket_str(formula) #get correct formula
	if correct_formula is None: continue
	Eg = info[0]['band_gap'] #get band gap
	Ehull= info[0]['e_above_hull'] #get e_above_hull
	ela_id_dict = dict(m.get_data(ID_list[i], data_type="pred", prop="elastic_moduli")[0]) # get ML prediction of MP
	ela_dict=ela_id_dict.get("elastic_moduli")
	elastic_K = ela_dict.get("K") # Bulk modulus
	elastic_G=ela_dict.get("G") # Shear modulus
	formation_energy = info[0]['formation_energy_per_atom'] #Formation energy
	cif = info[0]['cif']
	with open('cif/'+formula+".cif", "w") as text_file:
		text_file.write(cif)

	atom = read('cif/'+formula+".cif",index=None,format='cif',parallel=False)
	print(atom)
	symb = atom.get_chemical_symbols()
	print(symb)
	print(atom[0])

	M = atom.get_cell()
	Mi = np.linalg.inv(M)
	coor=atom.get_scaled_positions(wrap=True)

	dist_A_X_l=[]
	dist_B1_X_l=[]
	dist_B2_X_l=[]
	A_pos_l=[]
	B1_pos_l=[]
	B2_pos_l=[]
	X_pos_l=[]

	for i in range(0,len(symb)):
		if symb[i]==correct_formula[0]:
			A_pos=coor[i]
			A_pos_l.append(A_pos)
		if symb[i]==correct_formula[1]:
			B1_pos=coor[i]
			B1_pos_l.append(B1_pos)
		if symb[i]==correct_formula[2]:
			B2_pos=coor[i]
			B2_pos_l.append(B2_pos)
		if symb[i]==correct_formula[3]:
			X_pos=coor[i]
			X_pos_l.append(X_pos)

	A_pos_l=np.array(A_pos_l)
	B1_pos_l=np.array(B1_pos_l)
	B2_pos_l=np.array(B2_pos_l)
	X_pos_l=np.array(X_pos_l)


	#	Project to other directions
		# ADD ATOM IN SIX DIRECTION for A
		
	A_pos_l_1 = A_pos_l + [1,0,0]
	A_pos_l_2 = A_pos_l + [0,1,0]
	A_pos_l_3 = A_pos_l + [0,0,1]

	A_pos_l_4 = A_pos_l + [-1,0,0]
	A_pos_l_5 = A_pos_l + [0,-1,0]
	A_pos_l_6 = A_pos_l + [0,0,-1]

	A_pos_l_7 = A_pos_l + [1,1,0]
	A_pos_l_8 = A_pos_l + [0,1,1]
	A_pos_l_9 = A_pos_l + [1,0,1]


	A_pos_l_10 = A_pos_l + [-1,-1,0]
	A_pos_l_11 = A_pos_l + [0,-1,-1]
	A_pos_l_12 = A_pos_l + [-1,0,-1]


	A_pos_l_13 = A_pos_l + [-1,1,0]
	A_pos_l_14 = A_pos_l + [0,-1,1]
	A_pos_l_15 = A_pos_l + [-1,0,1]


	A_pos_l_16 = A_pos_l + [1,-1,0]
	A_pos_l_17 = A_pos_l + [0,1,-1]
	A_pos_l_18 = A_pos_l + [1,0,-1]


	A_pos_l_19 = A_pos_l + [1,1,1]
	A_pos_l_20 = A_pos_l + [-1,-1,-1]


	A_pos_l_21 = A_pos_l + [-1,1,1]
	A_pos_l_22 = A_pos_l + [1,-1,1]
	A_pos_l_23 = A_pos_l + [1,1,-1]



	A_pos_l_24 = A_pos_l + [1,-1,-1]
	A_pos_l_25 = A_pos_l + [-1,1,-1]
	A_pos_l_26 = A_pos_l + [-1,-1,1]

	A_pos_l_27 = A_pos_l + [-1,-1,0]

	A_pos_l=np.concatenate((A_pos_l,A_pos_l_1,A_pos_l_2,A_pos_l_3,A_pos_l_4,A_pos_l_5,A_pos_l_6,A_pos_l_7,A_pos_l_8,A_pos_l_9,
				A_pos_l_10,A_pos_l_11,A_pos_l_12,A_pos_l_13,A_pos_l_14,A_pos_l_15,A_pos_l_16,A_pos_l_17,A_pos_l_18,
				A_pos_l_19,A_pos_l_20,A_pos_l_21,A_pos_l_22,A_pos_l_23,A_pos_l_24,A_pos_l_25,A_pos_l_26,A_pos_l_27))

		# B1DD B1TOM IN SIX DIRECTION for B1

	B1_pos_l_1 = B1_pos_l + [1,0,0]
	B1_pos_l_2 = B1_pos_l + [0,1,0]
	B1_pos_l_3 = B1_pos_l + [0,0,1]

	B1_pos_l_4 = B1_pos_l + [-1,0,0]
	B1_pos_l_5 = B1_pos_l + [0,-1,0]
	B1_pos_l_6 = B1_pos_l + [0,0,-1]

	B1_pos_l_7 = B1_pos_l + [1,1,0]
	B1_pos_l_8 = B1_pos_l + [0,1,1]
	B1_pos_l_9 = B1_pos_l + [1,0,1]


	B1_pos_l_10 = B1_pos_l + [-1,-1,0]
	B1_pos_l_11 = B1_pos_l + [0,-1,-1]
	B1_pos_l_12 = B1_pos_l + [-1,0,-1]


	B1_pos_l_13 = B1_pos_l + [-1,1,0]
	B1_pos_l_14 = B1_pos_l + [0,-1,1]
	B1_pos_l_15 = B1_pos_l + [-1,0,1]


	B1_pos_l_16 = B1_pos_l + [1,-1,0]
	B1_pos_l_17 = B1_pos_l + [0,1,-1]
	B1_pos_l_18 = B1_pos_l + [1,0,-1]


	B1_pos_l_19 = B1_pos_l + [1,1,1]
	B1_pos_l_20 = B1_pos_l + [-1,-1,-1]


	B1_pos_l_21 = B1_pos_l + [-1,1,1]
	B1_pos_l_22 = B1_pos_l + [1,-1,1]
	B1_pos_l_23 = B1_pos_l + [1,1,-1]



	B1_pos_l_24 = B1_pos_l + [1,-1,-1]
	B1_pos_l_25 = B1_pos_l + [-1,1,-1]
	B1_pos_l_26 = B1_pos_l + [-1,-1,1]

	B1_pos_l_27 = B1_pos_l + [-1,-1,0]


	B1_pos_l=np.concatenate((B1_pos_l,B1_pos_l_1,B1_pos_l_2,B1_pos_l_3,B1_pos_l_4,B1_pos_l_5,B1_pos_l_6,B1_pos_l_7,B1_pos_l_8,B1_pos_l_9,
				B1_pos_l_10,B1_pos_l_11,B1_pos_l_12,B1_pos_l_13,B1_pos_l_14,B1_pos_l_15,B1_pos_l_16,B1_pos_l_17,B1_pos_l_18,
				B1_pos_l_19,B1_pos_l_20,B1_pos_l_21,B1_pos_l_22,B1_pos_l_23,B1_pos_l_24,B1_pos_l_25,B1_pos_l_26,B1_pos_l_27))

		# B2DD B2TOM IN SIX DIRECTION for B2

	B2_pos_l_1 = B2_pos_l + [1,0,0]
	B2_pos_l_2 = B2_pos_l + [0,1,0]
	B2_pos_l_3 = B2_pos_l + [0,0,1]

	B2_pos_l_4 = B2_pos_l + [-1,0,0]
	B2_pos_l_5 = B2_pos_l + [0,-1,0]
	B2_pos_l_6 = B2_pos_l + [0,0,-1]

	B2_pos_l_7 = B2_pos_l + [1,1,0]
	B2_pos_l_8 = B2_pos_l + [0,1,1]
	B2_pos_l_9 = B2_pos_l + [1,0,1]


	B2_pos_l_10 = B2_pos_l + [-1,-1,0]
	B2_pos_l_11 = B2_pos_l + [0,-1,-1]
	B2_pos_l_12 = B2_pos_l + [-1,0,-1]


	B2_pos_l_13 = B2_pos_l + [-1,1,0]
	B2_pos_l_14 = B2_pos_l + [0,-1,1]
	B2_pos_l_15 = B2_pos_l + [-1,0,1]


	B2_pos_l_16 = B2_pos_l + [1,-1,0]
	B2_pos_l_17 = B2_pos_l + [0,1,-1]
	B2_pos_l_18 = B2_pos_l + [1,0,-1]


	B2_pos_l_19 = B2_pos_l + [1,1,1]
	B2_pos_l_20 = B2_pos_l + [-1,-1,-1]


	B2_pos_l_21 = B2_pos_l + [-1,1,1]
	B2_pos_l_22 = B2_pos_l + [1,-1,1]
	B2_pos_l_23 = B2_pos_l + [1,1,-1]



	B2_pos_l_24 = B2_pos_l + [1,-1,-1]
	B2_pos_l_25 = B2_pos_l + [-1,1,-1]
	B2_pos_l_26 = B2_pos_l + [-1,-1,1]

	B2_pos_l_27 = B2_pos_l + [-1,-1,0]


	B2_pos_l=np.concatenate((B2_pos_l,B2_pos_l_1,B2_pos_l_2,B2_pos_l_3,B2_pos_l_4,B2_pos_l_5,B2_pos_l_6,B2_pos_l_7,B2_pos_l_8,B2_pos_l_9,
				B2_pos_l_10,B2_pos_l_11,B2_pos_l_12,B2_pos_l_13,B2_pos_l_14,B2_pos_l_15,B2_pos_l_16,B2_pos_l_17,B2_pos_l_18,
				B2_pos_l_19,B2_pos_l_20,B2_pos_l_21,B2_pos_l_22,B2_pos_l_23,B2_pos_l_24,B2_pos_l_25,B2_pos_l_26,B2_pos_l_27))

		# XDD XTOM IN SIX DIRECTION for X

	X_pos_l_1 = X_pos_l + [1,0,0]
	X_pos_l_2 = X_pos_l + [0,1,0]
	X_pos_l_3 = X_pos_l + [0,0,1]

	X_pos_l_4 = X_pos_l + [-1,0,0]
	X_pos_l_5 = X_pos_l + [0,-1,0]
	X_pos_l_6 = X_pos_l + [0,0,-1]

	X_pos_l_7 = X_pos_l + [1,1,0]
	X_pos_l_8 = X_pos_l + [0,1,1]
	X_pos_l_9 = X_pos_l + [1,0,1]


	X_pos_l_10 = X_pos_l + [-1,-1,0]
	X_pos_l_11 = X_pos_l + [0,-1,-1]
	X_pos_l_12 = X_pos_l + [-1,0,-1]


	X_pos_l_13 = X_pos_l + [-1,1,0]
	X_pos_l_14 = X_pos_l + [0,-1,1]
	X_pos_l_15 = X_pos_l + [-1,0,1]


	X_pos_l_16 = X_pos_l + [1,-1,0]
	X_pos_l_17 = X_pos_l + [0,1,-1]
	X_pos_l_18 = X_pos_l + [1,0,-1]


	X_pos_l_19 = X_pos_l + [1,1,1]
	X_pos_l_20 = X_pos_l + [-1,-1,-1]


	X_pos_l_21 = X_pos_l + [-1,1,1]
	X_pos_l_22 = X_pos_l + [1,-1,1]
	X_pos_l_23 = X_pos_l + [1,1,-1]



	X_pos_l_24 = X_pos_l + [1,-1,-1]
	X_pos_l_25 = X_pos_l + [-1,1,-1]
	X_pos_l_26 = X_pos_l + [-1,-1,1]

	X_pos_l_27 = X_pos_l + [-1,-1,0]


	X_pos_l=np.concatenate((X_pos_l,X_pos_l_1,X_pos_l_2,X_pos_l_3,X_pos_l_4,X_pos_l_5,X_pos_l_6,X_pos_l_7,X_pos_l_8,X_pos_l_9,
				X_pos_l_10,X_pos_l_11,X_pos_l_12,X_pos_l_13,X_pos_l_14,X_pos_l_15,X_pos_l_16,X_pos_l_17,X_pos_l_18,
				X_pos_l_19,X_pos_l_20,X_pos_l_21,X_pos_l_22,X_pos_l_23,X_pos_l_24,X_pos_l_25,X_pos_l_26,X_pos_l_27))

	# CONVERT TO CARTASIAN COORDINATE

	A_pos=A_pos_l.dot(M)
	B1_pos=B1_pos_l.dot(M)
	B2_pos=B2_pos_l.dot(M)
	X_pos=X_pos_l.dot(M)



	for i in range (0,len(X_pos)):
		for j in range (0,len(A_pos)):
			dist_A_X_l.append(np.linalg.norm(A_pos[j]-X_pos[i]))
		for j in range (0,len(B1_pos)):
			dist_B1_X_l.append(np.linalg.norm(B1_pos[j]-X_pos[i]))
		for j in range (0,len(B2_pos)):
			dist_B2_X_l.append(np.linalg.norm(B2_pos[j]-X_pos[i]))

	dist_A_X=min(dist_A_X_l)
	dist_B1_X=min(dist_B1_X_l)
	dist_B2_X=min(dist_B2_X_l)
	print(dist_A_X)
	print(dist_B1_X)
	print(dist_B2_X)

	if (len(correct_formula)==4): # Only select double perovskite
		volume = info[0]['volume']
		density = info[0]['density']
		row = np.concatenate((formula,correct_formula[0],correct_formula[1],correct_formula[2],correct_formula[3],
							  Eg,formation_energy,
							  Ehull,elastic_K,elastic_G,
							  density,volume,dist_A_X,dist_B1_X,dist_B2_X), axis=None)
#							  density,volume), axis=None)

		final_frame.append(row)

final_frame = np.array(final_frame)
print("final_frame",np.shape(final_frame))
with open('1_mp_yaocai_eg_fe_ehull_elastic_dens_vol.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	wr.writerow(("formula","A","B1","B2","X",
				 "Eg","F_E",
				 "Ehull","Elasticity_bulk","Elasticity_shear",
				 "density","volume",'AX_dist','B1X_dist','B2X_dit'))
	for row in final_frame:
		wr.writerow(row)
