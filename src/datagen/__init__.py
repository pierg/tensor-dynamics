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

def main():
    # Initialize the simulator
    sim = MichaelisMentenSimulation()

    # Parameters for the simulation
    k1 = 0.02
    k2 = 0.01
    k3 = 0.03

    # Initial concentrations
    E0 = 1.0  # Enzyme
    S0 = 2.0  # Substrate
    ES0 = 0.0  # Enzyme-substrate complex
    P0 = 0.0  # Product
    initial_conditions = [E0, S0, ES0, P0]

    # Integrate the ODE to get the simulation data
    data = sim.integrate_ode((k1, k2, k3), initial_conditions)

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
