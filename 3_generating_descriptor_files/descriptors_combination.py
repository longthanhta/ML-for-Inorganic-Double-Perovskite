import numpy as np
import csv, random 


def read_data():    
    ifile  = open('extr.csv', "rb")
    reader = csv.reader(ifile)
    csvdata=[]
    for row in reader:
        csvdata.append(row)   
    ifile.close()

    raw_header=csvdata[0]
    raw_header=raw_header[2:]
    
    csvdata=csvdata[1:]
    #cvadata=random.shuffle(csvdata) #<<<<<<<<<<<<<<<<<<<
    extr=[]
    compounds=[]
    raw_data=[]
    for item in range(len(csvdata)):
        features=csvdata[item]
        extr.append(features[1])      # target
        compounds.append(features[0])
        raw_data.append(features[2:]) # descriptors
             
    numrow=len(raw_data)
    numcol=len(raw_data[0]) 
    A = np.array(raw_data,dtype=float).reshape(numrow,numcol)
    Header = np.array(raw_header).reshape(numcol)
    nrow=len(extr)
    extraction = np.array(extr,dtype=float).reshape(nrow)
   
        
    return Header, A, extraction

