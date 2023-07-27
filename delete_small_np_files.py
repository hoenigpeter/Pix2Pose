import os

def delete_small_files(folder_path, max_size_bytes):
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                file_size = os.path.getsize(file_path)
                if file_size <= max_size_bytes:
                    print(f"Found small file: {file_path} ({file_size} bytes)")
                    confirm = input("Do you want to delete this file? (yes/no): ").strip().lower()
                    if confirm == "yes":
                        os.remove(file_path)
                        print(f"Deleted {file_path}")
                    else:
                        print("File was not deleted.")
            except OSError as e:
                print(f"Error while processing {file_path}: {e}")

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    folder_name = input("Enter the name of the folder in the current directory to scan for small files: ")

    folder_path = os.path.join(current_directory, folder_name)

    if os.path.exists(folder_path):
        max_file_size_bytes = 134
        delete_small_files(folder_path, max_file_size_bytes)
    else:
        print(f"The folder '{folder_name}' does not exist in the current directory.")