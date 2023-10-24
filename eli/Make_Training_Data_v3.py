# -*- coding: utf-8 -*-
"""


THis is meant to be a streamlined version of "Demixing Michaelis-Menten Reaction.ipynb"


Created on Tue Oct 17 15:17:34 2023

@author: Eli.Kinigstein
"""

#%%

import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.integrate as I
import scipy.linalg as LA
import time as tt
import pickle

%matplotlib qt5
#%%
# Here make a function to define the ODE using the kinetic coefficients as parameters. 

# note that C (shown below) is a vector of concentrations C=[C1,C2,C3,C4].T (from the definitions above)
# we change the indice to conform with python. Thus:

# C[0]=C1
# C[1]=C2
# C[2]=C3
# C[3]=C4

# k1,k2,k3 (shown below) have the meanings the same as above.

def MM_ODE(C,t,k1,k2,k3):
    
    #here I define each line o the ODE 
    # note C0 here corresponds to C1 above
    # C1 corresponds to C2 above...etc
    dC0_dt=  -C[0]*C[1]*k1 + k2*C[2]
    dC1_dt=  -C[0]*C[1]*k1 + k2*C[2] + k3*C[2]
    dC2_dt=   C[0]*C[1]*k1 - k2*C[2] - k3*C[2]
    dC3_dt=                            k3*C[2]
    
    #This makes the 4x1 array of dC_dt
    return np.array([dC0_dt , dC1_dt , dC2_dt , dC3_dt ])  

#%%

# Here I make a function to integrate the ODE and return the solutions!

# note that C_dot (Below) is the vector returned by the function MM_ODE()
# this is dC_dt
# C0 are the initial conditions. 

def Integrate_ODE(Kin_Coeff, C0):
    #here I define the range of time delays and the time sampling
    time=np.linspace(0,500, 501)
    
    #here I simply make a touple out of the kientic coefficients.
    Kinetic_Coefficents      =  (Kin_Coeff[0],Kin_Coeff[1],Kin_Coeff[2])
    #NOTE:Kinetic_Coefficents=  (k1          ,k2          ,k3)
    #k1,k2,k3 are the parameters for MM_ODE and are defined in the markdown cell above.
    
    #here I integrate the 
    C_sim=I.odeint(MM_ODE,y0=C0,t=time, args=  Kinetic_Coefficents  )
    return C_sim
#%% This is a function that gets the simulated weights to simulate the D.T*D

def Get_A(C_N, OL_S, Tr_D):
    
    #Note that C_N is not normalized.
    
    OL_C=np.zeros(OL_S.shape)
    
    #here I make the overlap matrix of a single concentration simulation vectors.
    for i in range(OL_S.shape[0]):
        for j in range(OL_S.shape[1]):
            OL_C[i,j]=C_N[:,i].T@C_N[:,j]                    

    #here I find the eignevectors of the combined overlap matrix for one simulation
    
    L,E=np.linalg.eig(OL_C*OL_S) # note, elementwise nultiply
    lam=np.reshape(L,(4,1))
    cont=True
    
    while(cont):
        XX=(np.random.random((4,1))-.5)/.5
        x=XX/np.linalg.norm(XX)
        z= np.sqrt(Tr_D)*np.copy(x)/(np.linalg.norm(x)*np.sqrt(lam))
        A=E@z
        if (A>0).all():
            cont=False
        else:
            cont=True
    return(A)
#%% this is a function that makes D.T*D

def Make_DT_D_5(C_sim, S_overlap, Tr_D, AA=[]):
    #print(AA.any())
    if AA.any():
        OA_Mat=AA*S_overlap
    else:
        #t_st4=tt.time()
        X=np.random.random((4,1)) #very important!!         
        x=X/np.linalg.norm(X)
        A= Get_A_3(C_sim,S_overlap, Tr_D)
        AA=np.outer(A,A)
        OA_Mat=AA*S_overlap
        #dt4=tt.time()-t_st4
        #print('making A takes:')
        #print(dt4)
        #print("\n")
    ddt=C_sim@OA_Mat@C_sim.T
    return ddt,AA
#%% This is a function that makes the R matrix from the real simulations and the SVD

def Make_R_Mat(C_svd, C_sim):
    R_mat=np.zeros((C_sim.shape[1],C_svd.shape[1]))
        
    for i in range(C_sim.shape[1]):
        for j in range(C_svd.shape[1]):
            R_mat[i,j]=C_sim[:,i].T@C_svd[:,j]
            
    return R_mat


#%%
def make_S(size):
    
    Keep_Looking=True
    
    while(Keep_Looking):
        
        #we need a size x size matrx. Thus we need a set of (size x 1) vectors with size total vector in the set.
        # we obtain the set of normlaized vectors to generate below
        s=[]
        for i in range(size):
            #this generates an arbitrary spectral vector with components between -1 and 1
            x=(np.random.random((size,1))-.5)/.5
            #the simulated spectra is normalzed.
            s.append(np.copy(x)/np.linalg.norm(x))
        Spectra=np.concatenate(s, axis=1)    
        # Now test to see how close the matrix is to sngular (below).
        # the condition number (np.linalg.cond) is a metric for how close the matrix is to singular.
        # Dispite the random nature of the generation process, this happends once in every 1000 calls to this function. 
        
        if np.linalg.cond(Spectra)>1e6:
            #print("The spectral overlap matrix is close to singular!")
            Keep_Looking=True
        else: 
            Keep_Looking=False
    return s
#%%
def make_A(size):

    #here I take a random vector with entries in the range 0.2 to 0.8
    return np.random.random((size,1))*.9 +.1

#%% This is a function that makes a random spetral overlap matrix.



def Make_Sim_Data_Matrix(s, C, A):   
    size=A.shape[0]
    D=np.zeros((C.shape[0],size))  
    
    
    for i in range(size):
        D+=np.outer(C[:,i], s[i].T)*A[i]

    #the line below simply scales the data such that the norm of the Dtaa matrix is one.
    # the idea is that we should normalize the input data in the same way, becasue we want the training data to be similar to the experimental data
    # here we enforce one particlar metric of similarity: that the matrix norm is always equalt ot that of the experimental data. 
    # however, I think that we can go further and only use training data that has the same eignevlaue spectrum as our data.
    D_norm=D/np.linalg.norm(D, ord='fro')
    [SVD_Concentration_Vectors,SVD_Values,SVD_Spectral_Vectors]=np.linalg.svd(D_norm, full_matrices=False)   
        
    return [SVD_Concentration_Vectors,SVD_Values,SVD_Spectral_Vectors]  


#%% NOw This is a function that puts together all of the steps to generate one simulated SVD from a random sampling of input parameters (k, c(t=0), SOM....)

def Make_Sim_Data(k1,k2,k3,c0_0,c1_0,c2_0,c3_0, A, S):
    #here we take in all of the parameters.

    #here I integrate to get one 
    C_sim=Integrate_ODE( [k1,k2,k3], [c0_0,c1_0,c2_0,c3_0])
    
    #here I make a simulated SVD with that concentration dynamics simulation
    [C_svd,w_svd,S_svd]=Make_Sim_Data_Matrix(S, C_sim, A)
    # note that S are the simulated 'spectra' and A are the simluated spectral weights.
        
    #Using the simulated SVD vectors (eigenvectors of DT@D) I can now make the R matrix (the answer)
    #note that I have added a transpose to the definition such that:
    # C_sim=SVD_Vectors@R
    R= Make_R_Mat(C_svd[:,:3], C_sim).T
                   
    Data_List={"NN_eVector_Input":C_svd[:,:3], "NN_eValue_Input":w_svd[:3], "NN_Prediction":R, "Conc_Dynamics":C_sim, "Spectral_Amplitudes":A, "Simulated Spectra":S }

    return Data_List

#%%
# this generates a random sample in a given range

def generate_log_uniform_sample(k1_min,k1_max):
    A=k1_min
    sam=np.random.random()
    q1=np.log((k1_max-k1_min)/A +1)
    k1_sam=(np.exp(q1*sam) -1)*A +k1_min
    
    return k1_sam

 
#%% Here I put it all togehter and generate the data files:
    
    
def Make_Data_File(
        File_Name,
        File_num ,
        kk1_min,
        kk1_max,
        kk2_min,
        kk2_max,
        kk3_min,
        kk3_max,
        cc0_min,
        cc0_max,
        cc1_min,
        cc1_max, Static_S=False, Static_A=False ):

    #this is the number of pieces of data that will go into one file
    Num_points=100000
    Data_Points=[]
    
    if Static_S:
        SS=make_S(size=4)  
    if Static_A:
        AA=make_A(size=4)    

        
    
    for i in range(Num_points):
        #here I generate all of the "randomly" sampled components
        
        K1=generate_log_uniform_sample(kk1_min,kk1_max)
        K2=generate_log_uniform_sample(kk2_min,kk2_max)
        K3=generate_log_uniform_sample(kk3_min,kk3_max)
        C0=generate_log_uniform_sample(cc0_min,cc0_max)
        C1=generate_log_uniform_sample(cc1_min,cc1_max)
        
        if not Static_S:
            SS=make_S(size=4)    
        if not Static_A:
            AA=make_A(size=4)   
                          #Make_Sim_Data(k1      ,k2       ,k3       ,c0_0      ,c1_0      ,c2_0,  c3_0,  A,S) 
        Data_Points.append(Make_Sim_Data(k1=K1,k2=K2 ,k3=K3 ,c0_0=C0,c1_0=C1,c2_0=0,c3_0=0,A=AA,S=SS))
       # NOw i'll need to generate the data, it i already mixed up!


    FN=File_Name+str(File_num)+".p"
    file_Z=open(FN, "wb")
    pickle.dump(Data_Points, file_Z)
    file_Z.close()
    #print(FN)
    return "Done with " + FN+"!"


#%% This is with variation of all parameters, but less than the previous dataset

fldr="E:\\DeMixing_SVD_Data\\Data_Set_3\\F1\\"
DTA_File="MM_Data_set_1_10_18_23_"
dta=[]
for i in range(200):
    print(Make_Data_File(fldr+DTA_File,i ,.01,.05,.01,.05,.01,.05,.3,1,.3,1, Static_S=False, Static_A=False ))

fldr="E:\\DeMixing_SVD_Data\\Data_Set_3\\F2\\"
DTA_File="MM_Data_set_2_10_18_23_"
dta=[]
for i in range(200):
    print(Make_Data_File(fldr+DTA_File,i ,.01,.05,.01,.05,.01,.05,.3,1,.3,1, Static_S=True, Static_A=False ))

fldr="E:\\DeMixing_SVD_Data\\Data_Set_3\\F3\\"
DTA_File="MM_Data_set_3_10_18_23_"
dta=[]
for i in range(200):
    print(Make_Data_File(fldr+DTA_File,i ,.01,.05,.01,.05,.01,.05,.3,1,.3,1, Static_S=False, Static_A=True ))


fldr="E:\\DeMixing_SVD_Data\\Data_Set_3\\F4\\"
DTA_File="MM_Data_set_4_10_18_23_"
dta=[]
for i in range(200):
    print(Make_Data_File(fldr+DTA_File,i ,.045,.05,.045,.05,.045,.05,.9,1,.9,1, Static_S=False, Static_A=False ))

fldr="E:\\DeMixing_SVD_Data\\Data_Set_3\\F5\\"
DTA_File="MM_Data_set_5_10_18_23_"
dta=[]
for i in range(200):
    print(Make_Data_File(fldr+DTA_File,i ,.01,.05,.045,.05,.045,.05,.9,1,.9,1, Static_S=True, Static_A=True ))

fldr="E:\\DeMixing_SVD_Data\\Data_Set_3\\F6\\"
DTA_File="MM_Data_set_6_10_18_23_"
dta=[]
for i in range(200):
    print(Make_Data_File(fldr+DTA_File,i ,.01,.05,.01,.05,.01,.05,.3,1,.3,1, Static_S=True, Static_A=True ))
    

fldr="E:\\DeMixing_SVD_Data\\Data_Set_3\\F7\\"
DTA_File="MM_Data_set_7_10_18_23_"
dta=[]
for i in range(200):
    print(Make_Data_File(fldr+DTA_File,i ,.045,.05,.045,.05,.045,.05,.9,1,.1,1, Static_S=True, Static_A=True ))

fldr="E:\\DeMixing_SVD_Data\\Data_Set_3\\F8\\"
DTA_File="MM_Data_set_8_10_18_23_"
dta=[]
for i in range(200):
    print(Make_Data_File(fldr+DTA_File,i ,.045,.05,.045,.05,.045,.05,.1,1,.9,1, Static_S=True, Static_A=True ))
    
    
    
    
#%% Here I run the above code and generate the data files.

x=Make_Data_File(fldr,i ,.01,.05,.01,.05,.01,.05,.3,1,.3,1, Static_S=False, Static_A=False )[0]

R=x["NN_Prediction"]
SVD_Vectors=x["NN_eVector_Input"]
Dyanmics=x["Conc_Dynamics"]
#Reconstructed_Dynamics= (R@SVD_Vectors.T).T
Reconstructed_Dynamics= SVD_Vectors@R

    
fig,ax=plt.subplots(4)    
ax[0].plot(Dyanmics[:,0]-Reconstructed_Dynamics[:,0])
ax[1].plot(Dyanmics[:,1]-Reconstructed_Dynamics[:,1])
ax[2].plot(Dyanmics[:,2]-Reconstructed_Dynamics[:,2])
ax[3].plot(Dyanmics[:,3]-Reconstructed_Dynamics[:,3])
        
#note, this is the Moore-Penrose pseudo inverse
print(R@np.linalg.pinv(R))
    