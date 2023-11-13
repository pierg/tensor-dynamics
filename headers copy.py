import os

from pathlib import Path

AUTHOR = "Piergiuseppe Mallozzi"
DATE = "November 2023"

def add_header_to_file(file_path, description):
    """
    Adds a header to the specified Python file.

    Args:
    - file_path (str): Path to the Python file.
    - description (str): Description to be added to the header.
    """
    header = (
        f"'''\n"
        f"Author: {AUTHOR}\n"
        f"Date: {DATE}\n"
        f"Description: {description}\n"
        f"'''\n\n"
    )

    with open(file_path, 'r+') as file:
        original_content = file.read()
        file.seek(0, 0)
        file.write(header + original_content)

def add_headers_to_directory(directory_path):
    """
    Adds headers to all Python files in the specified directory.

    Args:
    - directory_path (str): Path to the directory containing Python files.
    """
    for filename in os.listdir(directory_path):
        if filename.endswith('.py'):
            file_path = os.path.join(directory_path, filename)
            
            # Define the description based on the filename
            description = "Description for this file"
            if filename == 'model.py':
                description = "Defines the NeuralNetwork class, encompassing model initialization, training, evaluation, and saving functionalities."
            elif filename == 'datasets.py':
                description = "Dataclass Datasets for organizing and managing training, validation, and test datasets, including their statistics."
            elif filename == 'metrics.py':
                description = "Custom TensorFlow metrics implementation, including the R_squared metric class for model performance evaluation."
            elif filename == 'statistics_generator.py':
                description = "Defines RunningStats and RunningStatsDatapoints classes for calculating and maintaining running statistics of datasets."
            elif filename == 'michaeli.py':
                description = "Functions and classes for simulating Michaelis-Menten kinetics, generating random parameters, and processing spectral data."
            # Add more elif blocks for other filenames as needed

            add_header_to_file(file_path, description)

# Update this path to the directory containing your Python files
directory_path = Path(__file__).parent / "src"
add_headers_to_directory(directory_path)
