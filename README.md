# Biochemically-Informed Neural Networks
**Configurable Neural Networks with Biochemically-Informed Data Generation**

<p align="center">
  <img src="image.png" alt="Project Image">
</p>

This repository presents a flexible platform for building and training neural networks using TensorFlow. It uses physics models as foundation for generating synthetic datasets. Key features of the project include:

- **TOML-Based Configuration**: Easy customization of neural networks and data generation settings through TOML files.
- **TensorFlow Integration**: Utilizes TensorFlow for robust model building and training.
- **Dynamic Data Generation**: Features a data generation module capable of simulating complex biochemical reactions, including *Michaelis-Menten kinetics*, for creating synthetic datasets. 
- **Efficient Data Management**: Employs TensorFlow's Dataset API for optimized data handling and preprocessing.
- **High-Performance Computing (HPC) Support**: Includes scripts for HPC integration, compatible with Slurm-managed systems, leveraging TensorFlow's `MirroredStrategy` for distributed training.

This project is designed to harness the power of modern computational tools, offering a comprehensive solution for exploring neural networks in various scientific and research contexts.



This repository presents a flexible platform for building and training neural networks using TensorFlow. It uses physics models as foundation for generating synthetic datasets. Key features of the project include:

- **TOML-Based Configuration**: Easy customization of neural networks and data generation settings through TOML files.
- **TensorFlow Integration**: Utilizes TensorFlow for robust model building and training.
- **Dynamic Data Generation**: Features a data generation module capable of simulating complex biochemical reactions, including *Michaelis-Menten kinetics*, for creating synthetic datasets. 
- **Efficient Data Management**: Employs TensorFlow's Dataset API for optimized data handling and preprocessing.
- **High-Performance Computing (HPC) Support**: Includes scripts for HPC integration, compatible with Slurm-managed systems, leveraging TensorFlow's `MirroredStrategy` for distributed training.

This project is designed to harness the power of modern computational tools, offering a comprehensive solution for exploring neural networks in various scientific and research contexts.


## Prerequisites

To utilize this repository, ensure you have:

- **Python 3.11+**: Required for running the project. The latest version ensures compatibility and optimal performance.
- **TensorFlow**: Essential for building and training neural network models.
- **Poetry**: Recommended for efficient dependency management. [See Poetry Documentation](https://python-poetry.org/docs/).
- **Alternatively**: Use `requirements.txt` for managing Python dependencies without Poetry.

## Installation & Setup

1. **Clone the Repository**
   
   Get a local copy of the repository:
   ```bash
   git clone https://github.com/pierg/neural_networks.git
   cd neural_networks
   ```

2. **Install Dependencies**

   - With Poetry:
     ```bash
     poetry config virtualenvs.in-project true
     poetry install
     poetry shell
     pip install tensorflow
     ```
   
   - Without Poetry:
     Install using `pip` from `requirements.txt`.

3. **Configuration Setup**
   
   Configure your neural network, dataset, and parameters in `config/configurations.toml` and `config/data.toml`. This includes defining network architecture, data generation parameters, and training settings.

4. **Run the Project**

   Start the project using Poetry:
   ```bash
   poetry run python src/main.py
   ```

## Running with Docker

For streamlined deployment, use Docker. Run predefined configurations or specify custom ones.

- **Default Run**:
  ```bash
  docker run -it \
      -v <LOCAL_RESULTS_PATH>:/app/results \
      pmallozzi/neural_networks:latest
  ```

- **Specific Configuration**:
  ```bash
  docker run -it \
      -v <LOCAL_RESULTS_PATH>:/app/results \
      pmallozzi/neural_networks:latest CONFIGS='E'
  ```

- **Multiple Configurations**:
  ```bash
  docker run -it \
      -v <LOCAL_RESULTS_PATH>:/app/results \
      -v <LOCAL_CONFIG_PATH>:/app/config \
      pmallozzi/neural_networks:latest CONFIGS='E A'
  ```


## Modules and Features

### `deepnn` - Deep Neural Network Module
This module contains the core functionalities for constructing and managing neural networks. Key components include:

- **NeuralNetwork Class**: Handles the creation, training, and evaluation of neural network models.
- **Metrics**: Implements custom metrics, like `R_squared`, for model evaluation.
- **Callbacks**: Provides custom callbacks like `CustomSaveCallback` for advanced training control.

### `dataset` - Dataset Generation and Management
Manages data handling, including generation, preprocessing, and statistics calculation. It features:
  
- **Dataset Generators**: Create and manipulate datasets for training and evaluation.
- **Statistics Generator**: Calculates and maintains running statistics for datasets, aiding in insightful data analysis.

### `shared` - Shared Resources and Utilities
Includes shared paths and utility functions used across the project. It centralizes common resources for better maintainability.

### `config` - Configuration Management
Allows configuring neural network architectures, training parameters, and dataset characteristics through TOML files, providing a highly flexible and user-friendly interface.

## Configuration Files

The `configurations.toml` and `data.toml` files are central to the flexibility and user-friendliness of this project. They allow users to define and modify the neural network architecture, dataset parameters, and training settings without altering the codebase.

### `configurations.toml` Overview

The `configurations.toml` file is a central piece in customizing the neural network models within this framework. It allows users to define various aspects of neural networks, such as their architecture, compilation, and training parameters. This file supports the creation of multiple distinct configurations, each tailored to specific requirements or experimental setups.

#### Key Configurable Elements:

- **Structure**: Outlines the architecture of the neural network. Users can specify the types of layers (e.g., Conv2D, Dense), along with their respective parameters like filters, kernel size, units, and activation functions.

- **Compilation**: Defines how the model should be compiled. This includes setting the optimizer type (e.g., Adam, SGD), the loss function (e.g., mean squared error), and the evaluation metrics (e.g., R_squared, MeanAbsoluteError).

- **Training**: Configures various training aspects. This includes the number of epochs, batch size, and other parameters crucial for the training process. It also allows specifying data generation settings and normalization statistics.

#### Example Configuration:

```toml
# Configuration 'E': A sample neural network setup
[E]
# Network structure with layers defined sequentially
[E.structure]
layers = [
  { type = "Conv2D", filters = 32, kernel_size = [3, 1], activation = "relu" },
  { type = "Flatten" },
  { type = "Dense", units = 128, activation = "relu" },
  { type = "Dense", units = 12, activation = "linear" },
]
# Compilation settings including optimizer, loss, and metrics
[E.compile]
optimizer = "adam"
loss = "mean_squared_error"
metrics = ["R_squared", "MeanSquaredError", "MeanAbsoluteError", "MeanAbsolutePercentageError", "RootMeanSquaredError"]
# Dataset selection and statistics usage for normalization
[E.dataset]
training = "dbig" # Reference to dataset defined in data.toml
use_stats_of = 300000 # Use existing statistics from 300000 data points, if available
# Training parameters like epochs, save intervals, etc.
[E.training]
epochs = 200 # Indicative number of epochs, subject to early stopping
save_interval = 3 # Save model and results every 3 epochs
```

This file acts as a blueprint for neural network configurations, making the setup process intuitive and highly customizable. Users can easily experiment with different architectures and training regimes by adjusting the parameters in this TOML file.


### `data.toml` Overview

The `data.toml` file plays a crucial role in the dynamic generation of datasets based on biochemical models, specifically the Michaelis-Menten kinetics. It allows users to configure various parameters for data generation, offering a flexible approach to creating diverse datasets for neural network training and evaluation.

#### Data Generation Based on Michaelis-Menten Kinetics

- **Michaelis-Menten Kinetics**: This biochemical model describes the rate of enzymatic reactions. It is particularly used in this project to generate synthetic datasets that mimic real-world biochemical data.
- **Dynamic Generation**: Data points are generated on-the-fly during runtime, ensuring a fresh and varied dataset for each training session. This approach is beneficial for testing the robustness of the neural network against diverse data samples.

#### Configurable Parameters in `data.toml`

Each dataset configuration, such as `[dtest]`, contains parameters and shape settings that dictate how data will be generated:

- **Kinetic Parameters**: `k1`, `k2`, `k3`, `E0`, `S0` represent reaction rates and initial concentrations. Their ranges (`_min` and `_max`) can be specified, allowing the creation of diverse reaction conditions.
  
- **Spectra Parameters**: Control the characteristics of spectral data, a common type of output in biochemical analyses. This includes `spectra_min`, `spectra_max`, `amplitude_min`, and `amplitude_max`.

- **Shape Parameters**: Define the overall structure and size of the dataset.
  - `seed`: Ensures reproducibility in data generation.
  - `n_samples`: Total number of samples in the dataset.
  - `n_shards`: Divides the dataset into manageable parts, useful for distributed processing.
  - `batch_size`: Number of samples processed together, important for neural network training.
  - `shuffle_buffer_size`: Determines the extent of shuffling, which adds randomness to the data feeding process.
  - `splits`: Ratio of division of data into training, validation, and test sets.

#### Example:

```toml
[dtest]
[dtest.parameters]
k1_min = 0.01
k1_max = 0.05
k2_min = 0.01
k2_max = 0.05
k3_min = 0.01
k3_max = 0.05
E0_min = 0.1
E0_max = 0.3
S0_min = 0.1
S0_max = 0.3
spectra_min = -1
spectra_max = 1
amplitude_min = 0.2
amplitude_max = 0.8
[dtest.shape]
seed = 42
n_samples = 5000
n_shards = 10
batch_size = 50
shuffle_buffer_size = 100
splits = [0.7, 0.15, 0.15]
```

This configuration demonstrates how users can tailor the data generation process to fit specific research needs or experimental conditions. It empowers users to experiment with various biochemical scenarios and observe how well their neural network models perform under different data conditions.

### Running on High-Performance Computing (HPC) with Slurm

For users with access to High-Performance Computing (HPC) resources managed by Slurm Workload Manager, this project includes a dedicated `slurm` directory. This directory contains scripts optimized for efficiently managing and running neural network jobs on HPC clusters. These scripts are particularly beneficial for executing computationally intensive tasks, leveraging the advanced processing capabilities and GPU resources typically available in HPC environments.

#### Overview of the `slurm` Directory

- **`parallel-gpu.sh`**: This script automates the submission of multiple neural network training jobs to the Slurm scheduler. It reads configuration identifiers from the `configurations.toml` file and submits a Slurm job for each configuration, enabling parallel processing of multiple neural network architectures or datasets.

- **`submit-gpu.sh`**: This script is responsible for submitting an individual job to the Slurm scheduler. It configures job specifications such as GPU allocation, number of tasks, and CPUs per task. The script can be executed directly for specific configurations or invoked by `parallel-gpu.sh` for batch submissions.

#### How to Run Jobs on HPC

1. **Prepare Your Configurations**: Define your neural network architectures, datasets, and training parameters in the `config/configurations.toml` and `config/data.toml` files.

2. **Access the Slurm Environment**: Log in to your HPC cluster where Slurm is set up and navigate to the project directory.

3. **Submit Jobs Using `parallel-gpu.sh`**:
   - Ensure the `parallel-gpu.sh` and `submit-gpu.sh` scripts are executable (`chmod +x`).
   - Run `./parallel-gpu.sh` to submit jobs for all configurations defined in `configurations.toml`.
   - The script automatically queues jobs in Slurm, utilizing the specified resources.

4. **Monitor Job Status**: Use Slurm commands like `squeue`, `sinfo`, or `sacct` to monitor the status of your submitted jobs.

5. **Access Results**: After the jobs are completed, results can be found in the specified output directory, typically within the project's `results` folder.

#### Advantages of Using Slurm for HPC

- **Efficient Resource Utilization**: Slurm ensures optimal use of the available HPC resources, managing job queues and resource allocation effectively.
  
- **Scalability**: Easily scale your neural network training across multiple nodes and GPUs, facilitating the handling of large-scale data and complex models.

- **Parallel Processing**: Run multiple configurations simultaneously, significantly reducing the time required for extensive experimental studies.

These Slurm scripts are designed to help researchers and practitioners leverage HPC resources for advanced neural network experiments, making the process streamlined and user-friendly.

