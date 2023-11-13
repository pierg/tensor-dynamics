import os
from pathlib import Path

# Constants for author and date
AUTHOR = "Piergiuseppe Mallozzi"
DATE = "November 2023"

def header_exists(file_path, author_name):
    """
    Checks if the header already exists in the file.

    Args:
        file_path (Path): Path to the Python file.
        author_name (str): The name of the author to check in the header.

    Returns:
        bool: True if the header exists, False otherwise.
    """
    try:
        with open(file_path, 'r') as file:
            # Read the first few lines of the file
            first_lines = ''.join([next(file) for _ in range(3)])
            return author_name in first_lines
    except (IOError, StopIteration):
        return False

def add_header_to_file(file_path, author_name, date, description=None):
    """
    Adds a header to the specified Python file.

    Args:
        file_path (Path): Path to the Python file.
        author_name (str): Name of the author.
        date (str): Date to be added to the header.
        description (str, optional): Description to be added to the header.
    """
    # Construct the header with or without a description
    header = f"'''\nAuthor: {author_name}\nDate: {date}\n"
    if description:
        header += f"Description: {description}\n"
    header += "'''\n\n"

    with open(file_path, 'r+') as file:
        original_content = file.read()
        file.seek(0, 0)
        file.write(header + original_content)
        print(f"Header added to {file_path}")

def add_headers_to_files(directory_path, file_descriptions):
    """
    Adds headers to Python files based on provided descriptions.

    Args:
        directory_path (Path): Path to the directory containing Python files.
        file_descriptions (dict): Dictionary with file relative paths as keys and descriptions as values.
    """
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.py'):
                file_path = Path(root) / filename
                relative_path = str(file_path.relative_to(directory_path))

                # Check if the file needs a description in the header
                description = file_descriptions.get(relative_path)

                # Add header if it does not already exist
                if not header_exists(file_path, AUTHOR):
                    add_header_to_file(file_path, AUTHOR, DATE, description)

# Dictionary mapping file paths to their descriptions
file_descriptions = {
    "deepnn/model.py": "Defines the NeuralNetwork class, encompassing model initialization, training, evaluation, and saving functionalities.",
    "deepnn/datasets.py": "Dataclass Datasets for organizing and managing training, validation, and test datasets, including their statistics.",
    "deepnn/metrics.py": "Custom TensorFlow metrics implementation, including the R_squared metric class for model performance evaluation.",
    "shared/__init__.py": "Initialization for shared module, defining key directory paths for configuration, results, and datasets.",
    "dataset/statistics_generator.py": "Defines RunningStats and RunningStatsDatapoints classes for calculating and maintaining running statistics of datasets.",
    "dataset/michealis.py": "Functions and classes for simulating Michaelis-Menten kinetics, generating random parameters, and processing spectral data.",
    # Add more file paths and descriptions as needed
}

# Update this path to the directory containing your Python files
directory_path = Path(__file__).parent / "src"
add_headers_to_files(directory_path, file_descriptions)
