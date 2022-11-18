import os
import matplotlib.pyplot as plt
import ase.db
import ase
import csv
from ase.db import connect
import pandas as pd
import re

def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

db = connect('organometal.db')
number_of_structures=240
#print (db.get_atoms)


#	Creat list
#	The list below is all of key in db file
symmetry=[]
gllbsc_ind_gap=[]
energy=[]
calculator_parameters=[]
numbers=[]
mtime=[]
ctime=[]
gllbsc_disc=[]
cell=[]
name=[]
positions=[]
space_group=[]
gllbsc_dir_gap=[]
pbc=[]
initial_magmoms=[]
calculator=[]
unique_id=[]
user=[]
idl=[]

#	Creat folder CIF and cd to this

#os.mkdir('CIF')
#os.chdir('CIF')

#	Extract data


for i in range(1,number_of_structures+1):
	row = db.get(id=i)
	formula=row['formula']

#	Only choose HIOPs with ABX3 structure

	if 'Cl' or 'Br' or 'I' or 'F' in formula:
#		symmetry.append(row['symmetry'])
		gllbsc_dir_gap.append(row['gllbsc_dir_gap'])
		name.append(formula)
#		gllbsc_dir_gap.append(row['GLLB_ind'])
#pi		idl.append(int(i))
#		a = db.get_atoms(id=i)
#		filename=str(i)+".cif"
#		filename=formula+".cif"
#		ase.io.write(filename,a,format='cif')
#	cd 


#	Write data

df_out = pd.DataFrame.from_dict({'formula':name,'gllbsc_dir_gap':gllbsc_dir_gap})
df_out.to_excel('result.xlsx', header=True, index=False)




