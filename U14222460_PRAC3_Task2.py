# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:39:46 2020

@author: Lefa Raleting
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import csv


import pandas as pd


#This part of the code. opens up a specified csv file, then adds it to a turple
""""
with open('data1.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    line= [tuple(row) for row in reader]
csvFile.close()


database=line

#function to split datainto chunks

def chunks(l,n):
    for i in range(0,len(l),n):
        yield l[i:i+n]
#print(list(chunks(database,90))[0][89])
#This part of the code then goes on to create a data base of the the data collected
#counting ever instances and classifying it in a dictionary

#database=collections.Counter()
#database.update(line)




"""

#these are my helper dictionaries to better help navigate my functions
indexmoves= {
           0:'RRRR', 1:'RRRP', 2:'RRRS', 3:'RRPR', 4:'RRPP', 5:'RRPS', 6:'RRSR',
           7:'RRSP', 8:'RRSS', 9: 'RPRR', 10:'RPRP',11:'RPRS', 12:'RPPR', 13:'RPPP',
           14:'RPPS',15:'RPSR',16:'RPSP',17:'RPSS', 18:'RSRR', 19:'RSRP',20: 'RSRS',
           21: 'RSPR', 22: 'RSPP', 23: 'RSPS', 24: 'RSSR', 25: 'RSSP', 26: 'RSSS',
           27: 'PRRR', 28: 'PRRP', 29: 'PRRS', 30: 'PRPR', 31: 'PRPP', 32: 'PRPS',
           33: 'PRSR', 34: 'PRSP', 35: 'PRSS', 36: 'PPRR', 37: 'PPRP', 38: 'PPRS',
           39: 'PPPR', 40: 'PPPP', 41: 'PPPS', 42: 'PPSR', 43: 'PPSP', 44: 'PPSS',
           45: 'PSRR', 46: 'PSRP', 47: 'PSRS', 48: 'PSPR', 49: 'PSPP', 50: 'PSPS',
           51: 'PSSR', 52: 'PSSP', 53: 'PSSS', 54: 'SRRR', 55: 'SRRP', 56: 'SRRS',
           57: 'SRPR', 58: 'SRPP', 59: 'SRPS', 60: 'SRSR', 61: 'SRSP', 62: 'SRSS',
           63: 'SPRR', 64: 'SPRP', 65: 'SPRS', 66: 'SPPR', 67: 'SPPP', 68: 'SPPS',
           69: 'SPSR', 70: 'SPSP', 71: 'SPSS', 72: 'SSRR', 73: 'SSRP', 74: 'SSRS',
           75: 'SSPR', 76: 'SSPP', 77: 'SSPS', 78: 'SSSR', 79: 'SSSP', 80: 'SSSS'}
plays={0:'R',1:'P',2:'S'}
players={'R':0,'P':1,'S':2}
outputEncoder={'R':[1,0,0],'P':[0,1,0],'S':[0,0,1]}



movesIndex={
            "RRRR": 0,"RRRP": 1,"RRRS": 2,"RRPR": 3,"RRPP": 4,"RRPS": 5,"RRSR": 6,
            "RRSP": 7,"RRSS": 8,"RPRR":9,"RPRP": 10,"RPRS":11,"RPPR": 12,"RPPP": 13,
            "RPPS":14,"RPSR":15,"RPSP":16,"RPSS":17,"RSRR":18,"RSRP":19,"RSRS":20,
            "RSPR":21,"RSPP":22,"RSPS":23,"RSSR":24,"RSSP":25,"RSSS":26,"PRRR":27,
            "PRRP":28,"PRRS":29,"PRPR":30,"PRPP":31,"PRPS":32,"PRSR":33,"PRSP":34,
            "PRSS":35,"PPRR":36,"PPRP":37,"PPRS":38,"PPPR":39,"PPPP":40,"PPPS":41,
            "PPSR":42,"PPSP":43,"PPSS":44,"PSRR":45,"PSRP":46,"PSRS":47,"PSPR":48,
            "PSPP":49,"PSPS":50,"PSSR":51,"PSSP":52,"PSSS":53,"SRRR":54,"SRRP":55,
            "SRRS":56,"SRPR":57,"SRPP":58,"SRPS":59,"SRSR":60,"SRSP":61,"SRSS":62,
            "SPRR":63,"SPRP":64,"SPRS":65,"SPPR":66,"SPPP":67,"SPPS":68,"SPSR":69,
            "SPSP":70,"SPSS":71,"SSRR":72,"SSRP":73,"SSRS":74,"SSPR":75,"SSPP":76,
            "SSPS":77,"SSSR":78,"SSSP":79,"SSSS":80
            }


encoder={
            "RRRR": 0.0,"RRRP": 0.01,"RRRS": 0.02,"RRPR": 0.03,"RRPP": 0.04,"RRPS": 0.05,
            "RRSR": 0.06,"RRSP": 0.07,"RRSS": 0.08,"RPRR": 0.09,"RPRP": 0.10,"RPRS":0.11,
            "RPPR": 0.12,"RPPP": 0.13, "RPPS":0.14,"RPSR": 0.15,"RPSP": 0.16,"RPSS":0.17,
            "RSRR": 0.18,"RSRP": 0.19,"RSRS": 0.20,"RSPR": 0.21,"RSPP": 0.22,"RSPS":0.23,
            "RSSR": 0.24,"RSSP": 0.25,"RSSS": 0.26,"PRRR": 0.27,"PRRP": 0.28,"PRRS":0.29,
            "PRPR": 0.30,"PRPP": 0.31,"PRPS": 0.32,"PRSR": 0.33,"PRSP": 0.34, "PRSS":0.35,
            "PPRR": 0.36,"PPRP": 0.37,"PPRS": 0.38,"PPPR": 0.39,"PPPP": 0.40,"PPPS":0.41,
            "PPSR": 0.42,"PPSP": 0.43,"PPSS": 0.44,"PSRR": 0.45,"PSRP": 0.46,"PSRS":0.47,
            "PSPR": 0.48,"PSPP": 0.49,"PSPS": 0.50,"PSSR": 0.51,"PSSP": 0.52,"PSSS":0.53,
            "SRRR": 0.54,"SRRP": 0.55,"SRRS": 0.56,"SRPR": 0.57,"SRPP": 0.58,"SRPS":0.59,
            "SRSR": 0.60,"SRSP": 0.61,"SRSS": 0.62,"SPRR": 0.63,"SPRP": 0.64,"SPRS":0.65,
            "SPPR": 0.66,"SPPP": 0.67,"SPPS": 0.68,"SPSR": 0.69,"SPSP": 0.70,"SPSS":0.71,
            "SSRR": 0.72,"SSRP": 0.73,"SSRS": 0.74,"SSPR": 0.75,"SSPP": 0.76,"SSPS":0.77,
            "SSSR": 0.78,"SSSP": 0.79,"SSSS": 0.80
            }



#_______________Global parameters______________________________\

Input_number = 1 # This the number of nodes in the input layer
Hidden_layers = 2 #This is the number hidden layers
Hidden_number1= 3 #number of neurons in the first hidden layer
Hidden_number2= 3 #Number of neurons in the second hidden layer
Output_number= 3 #Number of nodes in output layer

#__________________Intializations______________________________

#________________________Matrix's_______________________________

#Matrix for between input and hidden layer 1
#Weight_IH= np.array([[-1.30256608, -0.63358578 , 0.46809963]])#np.random.randn(Input_number,Hidden_number1)
#Weight_IH= np.array([[ 0.03439296, -0.84921801, -0.67630459]])
Weight_IH=np.array([[0.98546655, 1.0, 0.68011305]]) #np.array([[ 0.62028726, -0.23636479, -0.56722029]])

#matrix for hidden layer 1
Bias_H1 = np.array([[0.65685711, 0.6665443,  0.45332548]])#np.random.rand(1,Hidden_number1)
#Matrix for between input and hidden layer 2 
#Weight_HH= np.array([[-1.82436288e-05,  1.11140185e-03, -2.23581046e-05],[ 6.63464877e-02, -3.05224354e-01,  2.11662861e-01],[ 8.98090548e-02, -4.11788163e-01,  2.86453940e-01]])# np.random.randn(Hidden_number1,Hidden_number2)
#Weight_HH=np.array([[-1.85954332,  0.99382871,  0.19574764],[-0.32298423,  0.14265085, -0.03934919],[-1.48797382,  0.83164407,  0.2485742 ]])
Weight_HH=np.array([[0.98546655, 0.33667067, 0.38093468],
           [1.0 ,     0.34163581, 0.38655262],
           [0.68011305, 0.23235098, 0.26289948]])
#matrix for hidden layer 2
Bias_H2= np.array([[0.3889918,  0.13289353, 0.1503658 ]])#np.random.rand(1,Hidden_number2)
#Weight Matrix for hidden layer 2 to output
#Weight_HO= np.array([[ 0.3024957 , -0.39158243, -0.38557635],[-0.30391214, -0.63549297,  0.11393838],[-0.1940526,  -0.70688921 ,-0.70417734]])#np.random.randn(Hidden_number2,Output_number)
#Weight_HO=np.array([[ 0.36541192, -0.77949451, -0.85524656],[-1.1897309,  -0.59789276,  0.93715327],[-0.22587574, -0.26511212, -0.71704524]])
Weight_HO=np.array([[0.80960078, 1.0,         0.92964603],
                    [0.27658862, 0.34163581, 0.31760038],
                    [0.3129533,  0.38655262, 0.35935711]])
#Bias matrix for output layer 
Bias_O =np.array([[0.93448987, 0.47523067, 0.48086404]])#np.random.rand(1,Output_number)


#Activation Functions 

#Leaky Relu 
def Relu(x,a=0.1):
    #if x is less than zero
    #return the multiplication of the value and small gradient
    #else return the value as is
    x[x<0]=a*x[x<0]

    return x

def DerivativeRelu(x,a=0.1):
    x[x<=0]=a #let the diravitive equal to the slope which is a for x<=0
    x[x>0]=1 #slope equal 1 for x>0
    return x
#Soft max function
def Softmax(x):
#    return np.exp(x)/np.exp(x).sum(axis=0)
    return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)),axis=0)
#softmax derivative
def Softmaxderv(x):
    #soft= x.reshape(-1,1)
    #return np.diagflat(soft) - np.dot(soft, soft.T)
    return (1-Softmax(x[0]))
#This Function is for forward propergation. Explained in the report 
def Foward_Propagation(Inputs,Weight_IH,Weight_HH,Weight_HO,a=0.1):
    S1= np.dot(Inputs,Weight_IH)+ Bias_H1
    A1= Relu(S1[0],a)    
    S2= np.dot(A1,Weight_HH)+ Bias_H2
    A2= Relu(S2[0],a)
    Y=  np.dot(A2,Weight_HO)+ Bias_O
    Out= Softmax(Y[0])
    
    return S1,A1,S2,A2,Y,Out



#This function is for backpopagation and it works as explained in the report    
def Back_Propagation(Out,S1,A1,S2,A2,Y,LearningRate,Target,inputs,a=0.1):    
    global Weight_HO
    global Weight_HH
    global Weight_IH
    
    Alpha1= np.multiply(-(Target-Out),Softmaxderv(Y)) #np.dot(np.array((Target-Out)),Softmaxderv(Y)) #3by3 matrix
   
    Link_H2O = np.dot(np.array([A2]).T, np.array([Alpha1])) #gradient calculation
    Alpha2= np.dot(Alpha1,Weight_HO.T)*DerivativeRelu(S2,a)
    Link_H1H2= np.dot(np.array([A1]).T,Alpha2)
    Alpha3=np.dot(Alpha2,Weight_HH.T)*DerivativeRelu(S1[0],a)
    Link_IH1=np.dot(np.array([inputs]).T,Alpha3)#
    
    Weight_HO=Weight_HO-LearningRate*Link_H2O
    Weight_HO=Weight_HO/np.max(Weight_HO)
    Weight_HH=Weight_HH-LearningRate*Link_H1H2
    Weight_HH=Weight_HH/np.max(Weight_HH)
    Weight_IH=Weight_IH-LearningRate*Link_IH1
    Weight_IH=Weight_IH/np.max(Weight_IH)
    
    return Weight_HO,Weight_HH,Weight_IH #Link_H2O,Link_H1H2,Link_IH1

#loss function to help corret the error in training
def Loss_function(EOut,Out):
    L=0.5*sum((EOut-Out)**2) 
    return L
#_________________________TASK1________________________________________
   
gama=0.05
number_of_iterations=10 #Epoch
Loss=[]


def train(Inputs,Target,IH,HH,HO,LearningRate,Epoch,alpha):
   # DataBase_length= len(database)
    
    error=0
    for i in range(Epoch):#[encoder[Inputs[k][0]]] outputEncoder[Target[k][1]]
        for k in Inputs: #range(0,1000000):#DataBase_length
            S1,A1,S2,A2,Y,Out=Foward_Propagation([encoder[k[0]]],IH,HH,HO,alpha)
            HO,HH,IH=Back_Propagation(Out,S1,A1,S2,A2,Y,LearningRate,outputEncoder[k[1]],[encoder[k[0]]] ,alpha)
            #arrayOut.append(Out)
            #print(Y)
            
        #L=Loss_functionDB(Target,arrayOut )
        L=Loss_function(outputEncoder[k[1]],Out)
        Loss.append(L)
        if(i>1):
            if(((Loss[i-1]-Loss[i])/Loss[i-1])<0.10 and (Loss[i-1]>=Loss[i]) ):
                error+=1
                if(error==3):
                    break
            else:
                error=0
#    plt.plot(range(i+1),Loss)
#    plt.title('Loss Function')
#    plt.xlabel('Epoch')
#    plt.ylabel('Loss')
  
    
#    print("After\n")
#    print("Weight_IH \n")
#    print(Weight_IH)
  
#    print("Weight_HH \n")
#    print(Weight_HH)
    
#    print("Weight_HO")
#    print(Weight_HO)      
        
           

    return HO,HH,IH,Loss            



#Task 2


#timea=time.time()
#train(database,database,Weight_IH,Weight_HH,Weight_HO,gama,number_of_iterations)
#timeb=time.time()
#print("Time:", timeb-timea)



#This is a helper function to determine output
def move(x):
    b=list(x)
    pos=b.index(max(b))
    if(pos==0):
        return 'R'
    elif(pos==1):
        return 'P'
    else:
        return 'S'
       

if input == "":
    previous=np.random.choice(["R", "P", "S"]) 
    moves=""
    matches=0
    temp=""
    alpha=0.05

else:
    if(matches<2):
        moves+=previous
        moves+=input
        previous=np.random.choice(["R", "P", "S"])
        matches+=1
    else:
        temp=previous
        S1,A1,S2,A2,Y,previous=Foward_Propagation(encoder[moves],Weight_IH,Weight_HH,Weight_HO,alpha)
        previous=move(previous)
        moves+=temp
        moves+=input
        moves=moves[2:]
    
output=previous
        
  
