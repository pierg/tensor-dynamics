import argparse
import os
import subprocess
from urllib.parse import urlparse, urlunparse
from pathlib import Path
import sys
import json

# -------------------------
# ARGUMENT PARSING FUNCTIONS
# -------------------------


def parse_arguments():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Process data and configurations.")
    parser.add_argument("configs", nargs="*", help="Optional configurations")
    return parser.parse_args()


# -------------------------
# DIRECTORY UTILS FUNCTIONS
# -------------------------


def validate_directory(path):
    """Validates if the provided path is a directory."""
    if not path.is_dir():
        print(f"Provided path is not a directory: {path}", file=sys.stderr)
        sys.exit(1)


def is_directory_empty(path):
    """Check if a directory is empty or does not exist."""
    if not os.path.exists(path):
        return True
    return len(os.listdir(path)) == 0


def save_dict_to_json_file(data_dict: dict, file_path: Path) -> None:
    """
    Save a dictionary into a formatted JSON file.

    Args:
    data_dict (Dict): The dictionary to save.
    file_path (Path): The file path where the dictionary should be saved.
    """
    # Ensure the parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(data_dict, file, ensure_ascii=False, indent=4)
        print(f"Data saved successfully to {file_path}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
    except json.JSONDecodeError as e:
        print(f"An error occurred with JSON encoding/decoding: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise  # Re-throw the exception after logging it


def format_section(title, dict_data):
    """
    Formats a dictionary into a sectioned string with a title.

    Args:
        title (str): The title of the section.
        dict_data (dict): The dictionary to format.

    Returns:
        str: A string representation of the dictionary formatted as a section.
    """
    formatted = f"{title}:\n"
    separator = "-" * len(title)
    formatted += f"{separator}\n"

    for key, value in dict_data.items():
        if isinstance(
            value, dict
        ):  # if value itself is a dictionary, format as a subsection
            value = "\n" + format_section(f"  {key}", value)
        formatted += f"{key.capitalize()}: {value}\n"

    return formatted + "\n"  # Add a newline for separation between sections


def pretty_print_dict(data, indent=0):
    """
    Recursively prints nested dictionaries.

    Args:
        data (dict): The dictionary to print.
        indent (int, optional): The current indentation level. Defaults to 0.
    """
    for key, value in data.items():
        # Print the key with the current indentation
        print("    " * indent + str(key), end="")

        if isinstance(value, dict):
            # If the value is another dictionary, print a colon and continue the recursion
            print(":")
            pretty_print_dict(
                value, indent + 1
            )  # Increase the indentation level for nested dictionary
        else:
            # If the value is not a dictionary, print a colon and the value itself
            print(f": {value}")


# -------------------------
# GIT OPERATIONS FUNCTIONS
# -------------------------


def clone_private_repo(repo_url, local_path):
    """Clones a private repository to the local path."""
    try:
        # Ensure the GitHub token is available
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GitHub token is not provided")

        # Parse the provided URL
        parsed_url = urlparse(repo_url)

        # Prepare the new netloc with the token
        # The format is: TOKEN@hostname (The '@' is used to separate the token from the hostname)
        new_netloc = f"{token}@{parsed_url.netloc}"

        # Construct the new URL components with the modified netloc
        new_url_components = (
            parsed_url.scheme,
            new_netloc,
            parsed_url.path,
            parsed_url.params,
            parsed_url.query,
            parsed_url.fragment,
        )

        # Reconstruct the full URL with the token included
        url_with_token = urlunparse(new_url_components)

        # Perform the clone operation
        subprocess.run(
            ["git", "clone", "--depth", "1", url_with_token, str(local_path)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while cloning the repo: {e}")
        raise e


def git_pull(folder=None, prefer_local=True):
    """Pulls the latest changes from a GitHub repository."""
    """
    Pulls the latest changes from a GitHub repository using token-based authentication.
    If there are conflicts, prefer local changes over remote ones based on the prefer_local flag.

    Args:
        folder (Path, optional): Path object representing the folder to pull. If not specified, the current directory is used.
        prefer_local (bool, optional): If true, resolve merge conflicts by preferring local changes; otherwise, prefer remote changes.
    """
    if folder is None:
        folder = (
            Path.cwd()
        )  # Use the current working directory if no folder is provided
    elif not folder.is_dir():
        raise ValueError(f"{folder} does not exist or is not a directory.")

    github_token = os.getenv("GITHUB_TOKEN")
    if github_token is None:
        raise ValueError("GITHUB_TOKEN is not set in the environment variables.")

    repo_url = os.getenv("GITHUB_RESULTS_REPO")
    if repo_url is None:
        raise ValueError("GITHUB_RESULTS_REPO is not set in the environment variables.")

    if not repo_url.startswith("https://"):
        raise ValueError("The repository URL must start with 'https://'")

    repo_url_with_token = repo_url.replace(
        "https://", f"https://{github_token}:x-oauth-basic@"
    )

    original_cwd = Path.cwd()  # Save the original working directory
    try:
        os.chdir(folder)  # Change to the target directory
        print(f"Changed directory to: {folder}")

        print("Starting to pull latest changes from remote repository...")
        # Fetch the changes from the remote repository
        subprocess.run(["git", "fetch", repo_url_with_token], check=True)

        if prefer_local:
            print("Merging changes with strategy favoring local changes...")
            subprocess.run(["git", "merge", "-Xours", "FETCH_HEAD"], check=True)
        else:
            print("Merging changes with strategy favoring remote changes...")
            subprocess.run(["git", "merge", "-Xtheirs", "FETCH_HEAD"], check=True)

        print("Operation completed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while pulling from GitHub: {str(e)}")
        raise  # Rethrow the exception to handle it at a higher level of your application
    finally:
        os.chdir(original_cwd)


def git_push(folder=None, commit_message="Update files"):
    """
    Pushes changes to a GitHub repository using token-based authentication, resolving conflicts by favoring local changes.

    Args:
    folder (Path, optional): Path object representing the folder to push. If not provided, the current directory is used.
    commit_message (str, optional): The commit message. Defaults to "Update files".
    """
    # Check for necessary environment variables
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token is None:
        raise ValueError("GITHUB_TOKEN is not set in the environment variables.")

    repo_url = os.getenv("GITHUB_RESULTS_REPO")
    if repo_url is None:
        raise ValueError("GITHUB_RESULTS_REPO is not set in the environment variables.")

    # Ensure the repository URL is correctly formatted
    if not repo_url.startswith("https://"):
        raise ValueError("The repository URL must start with 'https://'")

    repo_url_with_token = repo_url.replace(
        "https://", f"https://{github_token}:x-oauth-basic@"
    )

    # Validate and set the working directory
    if folder is None:
        folder = Path.cwd()
    elif not folder.is_dir():
        raise ValueError(f"{folder} does not exist or is not a directory.")

    original_cwd = Path.cwd()  # Store the original working directory

    try:
        os.chdir(folder)
        print(f"Changed directory to: {folder}")

        # Stage any changes and commit them
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(
            ["git", "commit", "-m", commit_message], check=False
        )  # It's okay if nothing to commit

        print("Pulling changes from remote repository...")
        subprocess.run(["git", "fetch", repo_url_with_token], check=True)

        print("Merging changes with strategy favoring local changes...")
        # Using -Xours keeps local changes in case of conflict
        subprocess.run(["git", "merge", "-Xours", "FETCH_HEAD"], check=True)

        print("Pushing local changes to the remote repository...")
        subprocess.run(["git", "push", repo_url_with_token, "main"], check=True)

        print("Operation completed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while pushing to GitHub: {str(e)}")
        raise
    finally:
        os.chdir(original_cwd)  # Always return to the original directory


# -------------------------
# ENVIRONMENT SETUP FUNCTIONS
# -------------------------


def load_secrets(file_path):
    """Loads secrets from a file into environment variables."""
    print(f"Loading secrets: {file_path}")
    with open(file_path, "r") as file:
        for line in file:
            # Clean up the line
            cleaned_line = line.strip()

            # Ignore empty lines and comments
            if cleaned_line == "" or cleaned_line.startswith("#"):
                continue

            # Check if line contains 'export ' from shell script format and remove it
            if cleaned_line.startswith("export "):
                cleaned_line = cleaned_line.replace(
                    "export ", "", 1
                )  # remove the first occurrence of 'export '

            # Split the line into key and value
            if "=" in cleaned_line:
                key, value = cleaned_line.split("=", 1)
                print(f"Loading {key}")
                os.environ[key] = value
            else:
                print(f"Warning: Ignoring line, missing '=': {cleaned_line}")


# -------------------------
# MAIN FUNCTION (if needed)
# -------------------------


def main():
    """Main function that could be used to execute the script."""
    args = parse_arguments()
    validate_directory(Path(args.data_dir))

    # Other function calls based on script logic...


if __name__ == "__main__":
    main()
