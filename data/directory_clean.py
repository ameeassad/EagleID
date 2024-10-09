import os

def rename_screenshot_files(root_dir):
    # Walk through all directories and files
    for subdir, _, files in os.walk(root_dir):
        # Filter out files that begin with "Screenshot"
        screenshot_files = [file for file in files if file.startswith("Screenshot")]

        # If there are no "Screenshot" files, continue to the next directory
        if not screenshot_files:
            continue

        # Sort the files alphabetically
        files_sorted = sorted(files)
        
        # Get the folder name
        parent_folder_name = os.path.basename(subdir)

        # Rename files that begin with "Screenshot"
        for idx, file in enumerate(files_sorted):
            if file.startswith("Screenshot"):
                file_path = os.path.join(subdir, file)
                new_file_name = f"{parent_folder_name}_{idx}.png"
                new_file_path = os.path.join(subdir, new_file_name)
                
                try:
                    os.rename(file_path, new_file_path)
                    print(f"Renamed {file_path} to {new_file_path}")
                except Exception as e:
                    print(f"Failed to rename {file_path}: {e}")


def delete_ds_store_files(root_dir):
    # Walk through all directories and files
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == '.DS_Store':
                file_path = os.path.join(subdir, file)
                
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

# Example usage:
delete_ds_store_files('../datasets/raptor_individuals_new')
# rename_screenshot_files('../datasets/raptor_individuals_new')
