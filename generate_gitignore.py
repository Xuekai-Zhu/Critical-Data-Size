import os

def get_file_size(file_path):
    """Get the size of a file in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

def generate_gitignore():
    # Create a set to store large files
    large_files = set()

    # Walk through current directory and all its subdirectories
    for dirpath, dirnames, filenames in os.walk('.'):
        # Check the size of each file and add it to the large_files set if its size is greater than 50 MB
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_size = get_file_size(file_path)
            if file_size > 50:
                # Append the relative file path to the set
                relative_path = os.path.relpath(file_path)
                large_files.add(relative_path)

    # Load the existing .gitignore file
    existing_files = set()
    try:
        with open('.gitignore', 'r') as gitignore_file:
            for line in gitignore_file:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith("#"):  # ignore comments and empty lines
                    existing_files.add(stripped_line)
    except FileNotFoundError:
        pass  # if the file doesn't exist yet, we'll create it below

    # Append the new large files to the .gitignore file, if they're not already there
    with open('.gitignore', 'a') as gitignore_file:
        for file in large_files:
            if file not in existing_files:
                gitignore_file.write(file + '\n')

if __name__ == "__main__":
    generate_gitignore()
