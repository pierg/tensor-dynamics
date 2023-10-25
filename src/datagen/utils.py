import numpy as np
import sys




def generate_random_parameters(low, high, size=None):
    """Generates random values within a specified range."""
    return np.random.uniform(low, high, size)

def generate_spectral_data(size, spectra_range, amplitude_range):
    """Creates spectral vectors and amplitudes with certain conditions."""
    spectral_vectors = []
    while True:
        # Generate normalized spectral vectors
        for _ in range(size):
            vec = np.random.uniform(spectra_range[0], spectra_range[1], (size, 1))
            spectral_vectors.append(vec / np.linalg.norm(vec))

        spectra = np.concatenate(spectral_vectors, axis=1)

        # Check condition to ensure spectra are not linearly dependent
        if np.linalg.cond(spectra) < 1/sys.float_info.epsilon:
            break

    # Generate amplitudes for the spectral data
    amplitudes = generate_random_parameters(amplitude_range[0], amplitude_range[1], (size, 1))
    return spectral_vectors, amplitudes

def simulate_svd(s, C, A):
    """Performs an SVD on the generated spectral data."""
    # Simulate the data matrix D using concentration and spectral info
    D = sum(np.outer(C[:, i], s[i].T) * A[i] for i in range(A.shape[0]))
    # Normalize the data matrix
    D_norm = D / np.linalg.norm(D, ord='fro')
    # Perform singular value decomposition
    return np.linalg.svd(D_norm, full_matrices=False)

def calculate_r_matrix(C_svd, C_sim):
    """Calculates the R matrix for assessing the simulation accuracy."""
    # R matrix represents the correlation between simulated and decomposed data
    return np.dot(C_sim.T, C_svd)


def generate_parameters(kinetic_coeff_range, initial_concentration_range, spectra_config, amplitude_config):
    """
    Generate random parameters for the simulation and spectral data based on the provided ranges.

    :param kinetic_coeff_range: Dictionary with ranges for the kinetic coefficients (k1, k2, k3).
    :param initial_concentration_range: Dictionary with ranges for the initial concentrations (E0, S0).
    :param spectra_config: Tuple defining the range for spectral vector components.
    :param amplitude_config: Tuple defining the range for spectral amplitudes.
    :return: Dictionary containing the generated parameters.
    """
    spectral_vectors, amplitudes = generate_spectral_data(4, spectra_config, amplitude_config)
    # Generate random kinetic coefficients and initial concentrations
    parameters = {
        'k1': generate_random_parameters(*kinetic_coeff_range['k1']),
        'k2': generate_random_parameters(*kinetic_coeff_range['k2']),
        'k3': generate_random_parameters(*kinetic_coeff_range['k3']),
        'E0': generate_random_parameters(*initial_concentration_range['E0']),
        'S0': generate_random_parameters(*initial_concentration_range['S0']),
        'spectral_vectors': spectral_vectors,
        'amplitudes': amplitudes
    }

    return parameters



def generate_datapoint(parameters):
    """
    Generate a datapoint by running the simulation with the provided parameters.
    """
    # Extract parameters for clarity
    k1, k2, k3 = parameters['k1'], parameters['k2'], parameters['k3']
    E0, S0 = parameters['E0'], parameters['S0']
    spectral_vectors, amplitudes = parameters['spectral_vectors'], parameters['amplitudes']

    # Initial conditions - assuming ES0 and P0 are zero
    initial_conditions = [E0, S0, 0.0, 0.0]  # E0, S0, ES0, P0

    # Initialize and run the Michaelis-Menten simulation
    simulator = MichaelisMentenSimulation()
    conc_simulation = simulator.run_simulation((k1, k2, k3), initial_conditions)

    # Perform SVD on the simulated spectral data
    U_svd, sigma_svd, Vt_svd = simulate_svd(spectral_vectors, conc_simulation, amplitudes)

    # Calculate the R matrix for comparison
    R_matrix = calculate_r_matrix(U_svd[:, :3], conc_simulation).T

    # Package the results for saving or further analysis
    datapoint = {
        "NN_eVector_Input": U_svd[:, :3],
        "NN_eValue_Input": sigma_svd[:3],
        "NN_Prediction": R_matrix,
        "Conc_Dynamics": conc_simulation,  # Adding concentration dynamics for completeness
        "parameters": parameters  # Storing the parameters used for this simulation
    }

    return datapoint
