"""
Author: Piergiuseppe Mallozzi
Date: November 2023
"""

import sys

import numpy as np
import scipy.integrate as spi


class MichaelisMentenSimulation:
    def __init__(self, time_frame=(0, 500), num_time_points=501):
        """
        Initializes the Michaelis-Menten kinetics simulation settings.

        Args:
            time_frame (tuple): Time frame for the simulation.
            num_time_points (int): Number of time points to use in the simulation.
        """
        self.time = np.linspace(time_frame[0], time_frame[1], num_time_points)

    @staticmethod
    def _mm_ode(C, t, k1, k2, k3):
        """
        Defines the system of ordinary differential equations (ODEs) for Michaelis-Menten kinetics.

        Args:
            C (list): List of concentrations [E, S, ES, P].
            t (float): Time variable for the ODEs.
            k1, k2, k3 (float): Kinetic coefficients.

        Returns:
            list: List of differential equations for each species.
        """
        E, S, ES, P = C  # Unpack concentrations
        dE_dt = -k1 * E * S + k2 * ES
        dS_dt = -k1 * E * S + (k2 + k3) * ES
        dES_dt = k1 * E * S - (k2 + k3) * ES
        dP_dt = k3 * ES
        return [dE_dt, dS_dt, dES_dt, dP_dt]

    def run_simulation(self, kinetic_coeff, initial_conditions):
        """
        Runs the reaction simulation with given parameters.

        Args:
            kinetic_coeff (tuple): Kinetic coefficients (k1, k2, k3).
            initial_conditions (list): Initial concentrations [E0, S0, ES0, P0].

        Returns:
            numpy.ndarray: Simulation results over the specified time points.
        """
        return spi.odeint(
            self._mm_ode, y0=initial_conditions, t=self.time, args=kinetic_coeff
        )


def generate_random_parameters(low, high, size=None):
    """
    Generates random parameter values within a specified range.

    Args:
        low, high (float): The lower and upper bounds of the range.
        size (int, optional): Number of values to generate.

    Returns:
        numpy.ndarray: Randomly generated parameter values.
    """
    return np.random.uniform(low, high, size)


def generate_spectral_data(size, spectra_range, amplitude_range):
    """
    Creates spectral vectors and amplitudes based on specified conditions.

    Args:
        size (int): Number of spectral vectors to generate.
        spectra_range (tuple): Range for spectral data generation.
        amplitude_range (tuple): Range for amplitude values.

    Returns:
        tuple: Spectral vectors and amplitudes.
    """
    spectral_vectors = []
    while True:
        for _ in range(size):
            vec = np.random.uniform(spectra_range[0], spectra_range[1], (size, 1))
            spectral_vectors.append(vec / np.linalg.norm(vec))

        spectra = np.concatenate(spectral_vectors, axis=1)
        if np.linalg.cond(spectra) < 1 / sys.float_info.epsilon:
            break

    amplitudes = generate_random_parameters(
        amplitude_range[0], amplitude_range[1], (size, 1)
    )
    return spectral_vectors, amplitudes


def simulate_svd(s, C, A):
    """
    Performs Singular Value Decomposition (SVD) on the generated spectral data.

    Args:
        s (numpy.ndarray): Spectral vectors.
        C (numpy.ndarray): Concentration data from the simulation.
        A (numpy.ndarray): Amplitudes for spectral data.

    Returns:
        tuple: Decomposed matrices from SVD.
    """
    D = sum(np.outer(C[:, i], s[i].T) * A[i] for i in range(A.shape[0]))
    D_norm = D / np.linalg.norm(D, ord="fro")
    return np.linalg.svd(D_norm, full_matrices=False)


def calculate_r_matrix(C_svd, C_sim):
    """
    Calculates the R matrix for assessing simulation accuracy.

    Args:
        C_svd (numpy.ndarray): Matrix from SVD decomposition.
        C_sim (numpy.ndarray): Simulated concentration data.

    Returns:
        numpy.ndarray: R matrix representing the correlation between simulated and decomposed data.
    """
    return np.dot(C_sim.T, C_svd)


def generate_range_config(param_config, keys):
    """
    Generates a dictionary of parameter ranges based on configuration.

    Args:
        param_config (dict): Parameter configuration dictionary.
        keys (list): List of keys to include in the range configuration.

    Returns:
        dict: Dictionary of parameter ranges.
    """
    range_config = {}
    for key in keys:
        min_key = f"{key}_min"
        max_key = f"{key}_max"
        range_config[key] = (param_config[min_key], param_config[max_key])
    return range_config


def generate_parameters(param_config):
    """
    Generates random parameters for the simulation and spectral data.

    Args:
        param_config (dict): Configuration dictionary containing parameter ranges.

    Returns:
        dict: Dictionary of generated parameters.
    """
    kinetic_keys = ["k1", "k2", "k3"]
    concentration_keys = ["E0", "S0"]
    kinetic_coeff_range = generate_range_config(param_config, kinetic_keys)
    initial_concentration_range = generate_range_config(
        param_config, concentration_keys
    )

    spectra_config = (param_config["spectra_min"], param_config["spectra_max"])
    amplitude_config = (param_config["amplitude_min"], param_config["amplitude_max"])

    spectral_vectors, amplitudes = generate_spectral_data(
        4, spectra_config, amplitude_config
    )

    parameters = {
        key: generate_random_parameters(*kinetic_coeff_range[key])
        for key in kinetic_keys
    }
    parameters.update(
        {
            key: generate_random_parameters(*initial_concentration_range[key])
            for key in concentration_keys
        }
    )
    parameters["spectral_vectors"] = spectral_vectors
    parameters["amplitudes"] = amplitudes

    return parameters


def generate_datapoint_dictionary(parameters):
    """
    Generates a datapoint by running the simulation with provided parameters.

    Args:
        parameters (dict): Parameters for the simulation.

    Returns:
        dict: A dictionary containing the results of the simulation and data processing.
    """
    k1, k2, k3 = parameters["k1"], parameters["k2"], parameters["k3"]
    E0, S0 = parameters["E0"], parameters["S0"]
    spectral_vectors, amplitudes = (
        parameters["spectral_vectors"],
        parameters["amplitudes"],
    )
    initial_conditions = [E0, S0, 0.0, 0.0]  # E0, S0, ES0, P0

    simulator = MichaelisMentenSimulation()
    conc_simulation = simulator.run_simulation((k1, k2, k3), initial_conditions)

    U_svd, sigma_svd, Vt_svd = simulate_svd(
        spectral_vectors, conc_simulation, amplitudes
    )
    R_matrix = calculate_r_matrix(U_svd[:, :3], conc_simulation).T

    datapoint = {
        "NN_eVector_Input": U_svd[:, :3],
        "NN_eValue_Input": sigma_svd[:3],
        "NN_Prediction": R_matrix,
        "Conc_Dynamics": conc_simulation,
        "parameters": parameters,
    }

    return datapoint


def generate_datapoint(parameters) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a data point with features and labels based on given parameters.

    Args:
        parameters (dict): Parameters for generating the data point.

    Returns:
        tuple: Features and labels for the data point.
    """
    datapoint = generate_datapoint_dictionary(parameters)

    # Assuming that operations are numpy-based and "features" need an additional dimension
    features = datapoint["NN_eValue_Input"] * datapoint["NN_eVector_Input"]
    if features.ndim == 2:
        features = features[:, :, np.newaxis]

    label = datapoint["NN_Prediction"].flatten()
    return features, label


def collect_data_points(num_samples, param_config):
    """
    Collects multiple data points based on the provided number of samples and parameter configuration.

    Args:
        num_samples (int): Number of data points to generate.
        param_config (dict): Configuration dictionary for parameter generation.

    Returns:
        tuple: Arrays of features and labels for all generated data points.
    """
    results_features = []
    results_labels = []

    for _ in range(num_samples):
        parameters = generate_parameters(param_config)
        features, labels = generate_datapoint(parameters)
        results_features.append(features.flatten())
        results_labels.append(labels)

    return np.array(results_features), np.array(results_labels)
