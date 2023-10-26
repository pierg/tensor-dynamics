import argparse
import os
import subprocess
from urllib.parse import urlparse, urlunparse
from pathlib import Path
import sys


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
        os.chdir(
            original_cwd
        )  


def git_push(folder=None, commit_message="Update files"):
    """Pushes changes to a GitHub repository."""
    """
    Pushes changes to a GitHub repository using the token-based authentication.

    Args:
    folder (Path, optional): Path object representing the folder to push. If not specified, the current directory is used.
    commit_message (str, optional): The commit message. Defaults to "Update files".
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

        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        local_changes = result.stdout.strip() != ""

        if local_changes:
            print("Adding local changes...")
            subprocess.run(["git", "add", "."], check=True)

            print(f"Committing with message: {commit_message}...")
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
        else:
            print("No local changes to commit.")

        print("Pushing to the remote repository...")
        push_command = ["git", "push", repo_url_with_token, "main"]
        try:
            subprocess.run(push_command, check=True)  # Attempt to push
        except subprocess.CalledProcessError:
            # If push fails, pull with strategy to prefer local changes
            print("Push failed. Pulling latest changes from remote repository...")
            subprocess.run(["git", "fetch", repo_url_with_token], check=True)
            print("Merging changes with strategy favoring local changes...")
            subprocess.run(["git", "merge", "-Xours", "FETCH_HEAD"], check=True)
            print("Pushing merged changes to the remote repository...")
            subprocess.run(push_command, check=True)  # Attempt to push again

        print("Operation completed successfully.")

    except subprocess.CalledProcessError as e:
        # Here, you might want to add logic to handle merge conflicts by pulling and preferring local changes.
        print(f"An error occurred while pushing to GitHub: {str(e)}")
        raise  # Rethrow the exception to handle it at a higher level of your application

    finally:
        os.chdir(
            original_cwd
        )  # Ensure that you always return to the original directory




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
