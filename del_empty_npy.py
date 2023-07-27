import os
import numpy as np

def is_empty_npy(file_path):
    try:
        arr = np.load(file_path)
        return arr.size == 0
    except (IOError, ValueError):
        # In case of any error while loading the file
        return False

def delete_empty_npy_files(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                if is_empty_npy(file_path):
                    os.remove(file_path)
                    print(f"Deleted empty file: {file_path}")

if __name__ == "__main__":
    folder_path = "./tless"  # Replace this with the path to your folder
    delete_empty_npy_files(folder_path)
