import os

def get_file_size(file_path):
    """Get the size of a file in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

def generate_gitignore():
    # Create a list to store large files
    large_files = []

    # Walk through current directory and all its subdirectories
    for dirpath, dirnames, filenames in os.walk('.'):
        # Check the size of each file and add it to the large_files list if its size is greater than 50 MB
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_size = get_file_size(file_path)
            if file_size > 50:
                # Append the relative file path to the list
                relative_path = os.path.relpath(file_path)
                large_files.append(relative_path)

    # Append the large files to the .gitignore file
    with open('.gitignore', 'a') as gitignore_file:
        gitignore_file.write("# Gitignore for large files\n")
        for file in large_files:
            gitignore_file.write(file + '\n')

if __name__ == "__main__":
    generate_gitignore()
