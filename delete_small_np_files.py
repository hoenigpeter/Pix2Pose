import os
import argparse

def delete_small_files(folder_path, max_size_bytes):
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                file_size = os.path.getsize(file_path)
                if file_size <= max_size_bytes:
                    print(f"Found small file: {file_path} ({file_size} bytes)")
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
            except OSError as e:
                print(f"Error while processing {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete small files in a folder.")
    parser.add_argument("folder_name", type=str, help="Name of the folder to scan for small files")
    parser.add_argument("--max_size_bytes", type=int, default=134, help="Maximum size of the small files in bytes")
    args = parser.parse_args()

    current_directory = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_directory, args.folder_name)

    if os.path.exists(folder_path):
        delete_small_files(folder_path, args.max_size_bytes)
    else:
        print(f"The folder '{args.folder_name}' does not exist in the current directory.")
