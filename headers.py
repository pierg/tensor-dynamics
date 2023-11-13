import os
from pathlib import Path

def add_header_to_file(file_path, description):
    """
    Adds a header to the specified Python file.

    Args:
    - file_path (Path): Path to the Python file.
    - description (str): Description to be added to the header.
    """
    header = (
        f"'''\n"
        f"Author: Piergiuseppe Mallozzi\n"
        f"Date: November 2023\n"
        f"Description: {description}\n"
        f"'''\n\n"
    )

    with open(file_path, 'r+') as file:
        original_content = file.read()
        file.seek(0, 0)
        file.write(header + original_content)

def add_headers_to_files(directory_path, file_descriptions):
    """
    Adds headers to Python files based on provided descriptions.

    Args:
    - directory_path (Path): Path to the directory containing Python files.
    - file_descriptions (dict): Dictionary with file relative paths as keys and descriptions as values.
    """
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.py'):
                file_path = Path(root) / filename
                relative_path = file_path.relative_to(directory_path)

                # Add header if description exists in the dictionary
                if str(relative_path) in file_descriptions:
                    description = file_descriptions[str(relative_path)]
                    add_header_to_file(file_path, description)

# Dictionary mapping file paths to their descriptions
file_descriptions = {
    "deepnn/model.py": "Defines the NeuralNetwork class, encompassing model initialization, training, evaluation, and saving functionalities.",
    "deepnn/datasets.py": "Dataclass Datasets for organizing and managing training, validation, and test datasets, including their statistics.",
    "deepnn/metrics.py": "Custom TensorFlow metrics implementation, including the R_squared metric class for model performance evaluation.",
    "shared/__init__.py": "Initialization for shared module, defining key directory paths for configuration, results, and datasets.",
    "dataset/statistics_generator.py": "Defines RunningStats and RunningStatsDatapoints classes for calculating and maintaining running statistics of datasets.",
    "dataset/michaeli.py": "Functions and classes for simulating Michaelis-Menten kinetics, generating random parameters, and processing spectral data.",
    # Add more file paths and descriptions as needed
}


# Update this path to the directory containing your Python files
directory_path = Path(__file__).parent / "src"
add_headers_to_files(directory_path, file_descriptions)
