import numpy as np
import sys
import scipy.integrate as spi


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
        return spi.odeint(
            self._mm_ode, y0=initial_conditions, t=self.time, args=kinetic_coeff
        )


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
        if np.linalg.cond(spectra) < 1 / sys.float_info.epsilon:
            break

    # Generate amplitudes for the spectral data
    amplitudes = generate_random_parameters(
        amplitude_range[0], amplitude_range[1], (size, 1)
    )
    return spectral_vectors, amplitudes


def simulate_svd(s, C, A):
    """Performs an SVD on the generated spectral data."""
    # Simulate the data matrix D using concentration and spectral info
    D = sum(np.outer(C[:, i], s[i].T) * A[i] for i in range(A.shape[0]))
    # Normalize the data matrix
    D_norm = D / np.linalg.norm(D, ord="fro")
    # Perform singular value decomposition
    return np.linalg.svd(D_norm, full_matrices=False)


def calculate_r_matrix(C_svd, C_sim):
    """Calculates the R matrix for assessing the simulation accuracy."""
    # R matrix represents the correlation between simulated and decomposed data
    return np.dot(C_sim.T, C_svd)


def generate_range_config(param_config, keys):
    """Generate a dictionary of parameter ranges based on configuration and keys."""
    range_config = {}
    for key in keys:
        min_key = f"{key}_min"
        max_key = f"{key}_max"
        range_config[key] = (param_config[min_key], param_config[max_key])
    return range_config


def generate_parameters(param_config):
    """
    Generate random parameters for the simulation and spectral data based on the provided ranges.
    """
    # Define keys for ranges
    kinetic_keys = ["k1", "k2", "k3"]
    concentration_keys = ["E0", "S0"]  # ES0 and P0 are assumed to be zero

    # Generate range configurations
    kinetic_coeff_range = generate_range_config(param_config, kinetic_keys)
    initial_concentration_range = generate_range_config(
        param_config, concentration_keys
    )

    # Simplified retrieval of config data
    spectra_config = (param_config["spectra_min"], param_config["spectra_max"])
    amplitude_config = (param_config["amplitude_min"], param_config["amplitude_max"])

    # Generate spectral data
    spectral_vectors, amplitudes = generate_spectral_data(
        4, spectra_config, amplitude_config
    )

    # Create a dictionary for parameters and generate random values
    parameters = {}
    for key in kinetic_keys:
        parameters[key] = generate_random_parameters(*kinetic_coeff_range[key])
    for key in concentration_keys:
        parameters[key] = generate_random_parameters(*initial_concentration_range[key])

    # Add spectral vectors and amplitudes to parameters
    parameters.update({"spectral_vectors": spectral_vectors, "amplitudes": amplitudes})

    return parameters


def generate_datapoint_dictionary(parameters):
    """
    Generate a datapoint by running the simulation with the provided parameters.
    """
    # Extract parameters for clarity
    k1, k2, k3 = parameters["k1"], parameters["k2"], parameters["k3"]
    E0, S0 = parameters["E0"], parameters["S0"]
    spectral_vectors, amplitudes = (
        parameters["spectral_vectors"],
        parameters["amplitudes"],
    )

    # Initial conditions - assuming ES0 and P0 are zero
    initial_conditions = [E0, S0, 0.0, 0.0]  # E0, S0, ES0, P0

    # Initialize and run the Michaelis-Menten simulation
    simulator = MichaelisMentenSimulation()
    conc_simulation = simulator.run_simulation((k1, k2, k3), initial_conditions)

    # Perform SVD on the simulated spectral data
    U_svd, sigma_svd, Vt_svd = simulate_svd(
        spectral_vectors, conc_simulation, amplitudes
    )

    # Calculate the R matrix for comparison
    R_matrix = calculate_r_matrix(U_svd[:, :3], conc_simulation).T

    # Package the results for saving or further analysis
    datapoint = {
        "NN_eVector_Input": U_svd[:, :3],
        "NN_eValue_Input": sigma_svd[:3],
        "NN_Prediction": R_matrix,
        "Conc_Dynamics": conc_simulation,  # Adding concentration dynamics for completeness
        "parameters": parameters,  # Storing the parameters used for this simulation
    }

    return datapoint


def normalize_data(data, mean, std):
    """
    Normalize data using the provided mean and standard deviation, flattening and reshaping as needed.

    Args:
    data (list/np.array): The data to be normalized, expected in the form of a list or a NumPy array.
    mean (float/list/np.array): The mean value(s) used for normalization. Could be a list, a single float, or a NumPy array.
    std (float/list/np.array): The standard deviation value(s) used for normalization. Could be a list, a single float, or a NumPy array.

    Returns:
    np.array: The normalized data, reshaped to its original dimensions.
    """

    # Ensure the inputs are NumPy arrays
    data_np = np.array(data) if not isinstance(data, np.ndarray) else data
    mean_np = np.array(mean) if not isinstance(mean, np.ndarray) else mean
    std_np = np.array(std) if not isinstance(std, np.ndarray) else std

    # Record the original shape of the data
    original_shape = data_np.shape

    # Flatten the data if it's not already a 1D array
    data_flattened = data_np.flatten() if len(original_shape) > 1 else data_np

    # Normalize the flattened data. If mean and std are single values, they'll broadcast.
    # If they're arrays, their length should match the number of elements in the flattened data.
    normalized_data_flattened = (data_flattened - mean_np) / std_np

    # Reshape the data back to its original shape
    normalized_data = normalized_data_flattened.reshape(original_shape)

    return normalized_data


def quantize_data(data, q_min=-128, q_max=127):
    """Quantize the normalized data."""
    # Assuming data is already normalized.
    # You might want to adjust the range depending on your exact needs.
    quantized = np.round(data * q_max)  # or use other factors for quantization
    # Ensure values are within the desired range
    quantized = np.clip(quantized, q_min, q_max)
    return quantized.astype(np.int32)  # or np.uint8 or other as needed


def generate_datapoint(parameters) -> tuple[np.ndarray, np.ndarray]:
    datapoint = generate_datapoint_dictionary(parameters)

    # Assuming that the operations are numpy-based and "features" need an additional dimension
    features = datapoint["NN_eValue_Input"] * datapoint["NN_eVector_Input"]

    # Check the current shape of your features and decide if you need to add an extra dimension.
    # This is typically necessary when preparing data for convolutional networks which expect data in a 3D format.
    if features.ndim == 2:
        # The data is 2D, and we need to add an additional dimension to it.
        # np.newaxis is used to increase the dimension of the existing array by one more dimension,
        # when used once. Thus, 2D becomes 3D.
        features = features[:, :, np.newaxis]

    # For the label, we're flattening it as it seems to be a single dimensional output per data point
    label = datapoint["NN_Prediction"].flatten()

    return features, label


def collect_data_points(num_samples, param_config):
    results_features = []
    results_labels = []

    for _ in range(num_samples):
        # Generate random parameters based on your configuration
        parameters = generate_parameters(param_config)

        # Generate data points
        features, labels = generate_datapoint(parameters)

        # Store the results
        results_features.append(
            features.flatten()
        )  # flatten if the data is multi-dimensional
        results_labels.append(labels)

    return np.array(results_features), np.array(results_labels)
