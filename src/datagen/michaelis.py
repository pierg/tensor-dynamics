import numpy as np
import scipy.integrate as spi
import os, sys
import pickle

class MichaelisMentenSimulation:
    def __init__(self, time_frame=(0, 500), num_time_points=501):
        """Initializes the simulation settings."""
        # Define the timeframe for the simulation
        self.time = np.linspace(time_frame[0], time_frame[1], num_time_points)

    @staticmethod
    def _mm_ode(C, t, k1, k2, k3):
        """Defines the system of ODEs for Michaelis-Menten kinetics."""
        E, S, ES, P = C  # Unpack concentrations
        # Define the differential equations for each species
        dE_dt = -k1 * E * S + k2 * ES
        dS_dt = -k1 * E * S + (k2 + k3) * ES
        dES_dt = k1 * E * S - (k2 + k3) * ES
        dP_dt = k3 * ES
        return [dE_dt, dS_dt, dES_dt, dP_dt]

    def run_simulation(self, kinetic_coeff, initial_conditions):
        """Runs the reaction simulation with the given parameters."""
        # Solve the system of ODEs
        return spi.odeint(self._mm_ode, y0=initial_conditions, t=self.time, args=kinetic_coeff)

    @staticmethod
    def save_results(simulation_data, directory, file_name):
        """Saves the simulation data to a file."""
        if not os.path.exists(directory):
            os.makedirs(directory)  # Ensure the directory exists
        with open(os.path.join(directory, file_name), "wb") as file:
            pickle.dump(simulation_data, file)  # Write the data to a file



