import numpy as np
import scipy.integrate as spi
import pickle
import os

class MichaelisMentenSimulation:
    def __init__(self, time_frame=(0, 500), num_time_points=501):
        self.time = np.linspace(time_frame[0], time_frame[1], num_time_points)

    def mm_ode(self, C, t, k1, k2, k3):
        """Defines the ODE for the Michaelis-Menten kinetics."""
        E, S, ES, P = C  # Unpack concentrations
        dE_dt = -k1 * E * S + k2 * ES
        dS_dt = -k1 * E * S + (k2 + k3) * ES
        dES_dt = k1 * E * S - (k2 + k3) * ES
        dP_dt = k3 * ES
        return [dE_dt, dS_dt, dES_dt, dP_dt]

    def integrate_ode(self, kinetic_coeff, initial_conditions):
        """Integrates the ODE to simulate the reaction over time."""
        res = spi.odeint(self.mm_ode, y0=initial_conditions, t=self.time, args=kinetic_coeff)
        return res

    @staticmethod
    def save_simulation(simulation_data, directory, file_name):
        """Saves the simulation data to a file."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, file_name), "wb") as file:
            pickle.dump(simulation_data, file)



def make_S(size):
    # Spectral vectors of species. 4 vectors since we have 4 species. 

    Keep_Looking=True
    
    while(Keep_Looking):
        
        #we need a size x size matrx. Thus we need a set of (size x 1) vectors with size total vector in the set.
        # we obtain the set of normlaized vectors to generate below
        s=[]
        for i in range(size):
            #this generates an arbitrary spectral vector with components between -1 and 1

            x=(np.random.random((size,1))-.5)/.5

            # normalise and append it
            s.append(np.copy(x)/np.linalg.norm(x))

        Spectra=np.concatenate(s, axis=1)   
        
        # We want to make sure it's not linearly independent. 
        # As no species have same spectra.
        # Now test to see how close the matrix is to sngular (below).
        # the condition number (np.linalg.cond) is a metric for how close the matrix is to singular.
        # Dispite the random nature of the generation process, this happends once in every 1000 calls to this function. 
        
        # Keep generating until the conition of linearly independence is met
        if np.linalg.cond(Spectra)>1e6:
            #print("The spectral overlap matrix is close to singular!")
            Keep_Looking=True
        else: 
            Keep_Looking=False
    return s


def make_A(size):
    # Amplitudes of the spectral vectors
    #here I take a random vector with entries in the range 0.2 to 0.8
    return np.random.random((size,1))*.9 +.1




def Make_Sim_Data_Matrix(s, C, A):   
    size=A.shape[0]
    D=np.zeros((C.shape[0],size))  
    
    
    for i in range(size):
        D+=np.outer(C[:,i], s[i].T)*A[i]

    #the line below simply scales the data such that the norm of the Dtaa matrix is one.
    # the idea is that we should normalize the input data in the same way, becasue we want the training data to be similar to the experimental data
    D_norm=D/np.linalg.norm(D, ord='fro')
    [SVD_Concentration_Vectors,SVD_Values,SVD_Spectral_Vectors]=np.linalg.svd(D_norm, full_matrices=False)   
        
    return [SVD_Concentration_Vectors,SVD_Values,SVD_Spectral_Vectors]  


def Make_R_Mat(C_svd, C_sim):
    R_mat=np.zeros((C_sim.shape[1],C_svd.shape[1]))
        
    for i in range(C_sim.shape[1]):
        for j in range(C_svd.shape[1]):
            R_mat[i,j]=C_sim[:,i].T@C_svd[:,j]
            
    return R_mat


def main():
    # Initialize the simulator
    sim = MichaelisMentenSimulation()


    # Sample randomly
    # Use uniform distribution between .01 and .05
    k1 = 0.02
    k2 = 0.01
    k3 = 0.03

    # Initial concentrations
    # Sample randomly
    # Use uniform distribution e.g. between .3 and .1
    # Keep ES0 and P0 to zero
    E0 = 1.0  # Enzyme
    S0 = 2.0  # Substrate
    ES0 = 0.0  # Enzyme-substrate complex
    P0 = 0.0  # Product

    initial_conditions = [E0, S0, ES0, P0]

    # Integrate the ODE to get the simulation data
    c_sim = sim.integrate_ode((k1, k2, k3), initial_conditions)


    # Simulated SVD with that concentration dynamics simulation
    # We assume values of ground truth for spectral vector S and spectral amplitude A
    # Let's make S and A
    # Each species has one spectra, so we need to generate 4
    s = make_S(4)
    a = make_A(4)
    
    [c_svd,w_svd,s_svd] = Make_Sim_Data_Matrix(s, c_sim, a)

    # note that S are the simulated 'spectra' and A are the simluated spectral weights.
    #Using the simulated SVD vectors (eigenvectors of DT@D) I can now make the R matrix (the answer)
    #note that I have added a transpose to the definition such that:
    # C_sim=SVD_Vectors@R
    R= Make_R_Mat(c_svd[:,:3], c_sim).T
            



    Data_List={"NN_eVector_Input":c_svd[:,:3], "NN_eValue_Input":w_svd[:3], "NN_Prediction":R, "Conc_Dynamics":c_sim, "Spectral_Amplitudes":a, "Simulated Spectra":s }


    # features: NN_eVector_Input x NN_eValue_Input
    # target: NN_Prediction

    # Prepare the data dictionary to save
    simulation_data = {
        'time': sim.time,
        'data': data,
        'parameters': {'k1': k1, 'k2': k2, 'k3': k3},
        'initial_conditions': initial_conditions
    }

    # Define directory and file_name
    directory = "simulations"
    file_name = "simulation_data.pkl"

    # Save the simulation data
    sim.save_simulation(simulation_data, directory, file_name)

    print(f"Simulation data saved in {os.path.join(directory, file_name)}")

if __name__ == "__main__":
    main()
