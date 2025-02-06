# import torch
# ### MY addition
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if device.type == "cpu":
#     print("CUDA is not available, using CPU for training.")
# else:
#     # Print available devices
#     num_devices = torch.cuda.device_count()
#     print(f"CUDA is available. Number of devices: {num_devices}")

#     # Try connecting to the specific device
#     try:
#         torch.cuda.set_device(1)  # SET GPU INDEX HERE:
#         current_device = torch.cuda.current_device()
#         device_name = torch.cuda.get_device_name(current_device)
#         print(f"Using GPU device {current_device}: {device_name}")
#     except Exception as e:
#         device = torch.device("cpu")

import os

def count_lines(file_path):
    """Returns the number of lines in a file."""
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

def check_files(original_dir='/dsi/scratch/home/dsi/orrbavly/corona_data/original_files/', embeddings_dir='/dsi/scratch/home/dsi/orrbavly/corona_data/embeddings/'):
    mismatched_files = []
    counter = 0
    for file_name in os.listdir(original_dir):
        original_path = os.path.join(original_dir, file_name)
        base_name, ext = os.path.splitext(file_name)
        embedding_file_name = f"{base_name}_embedded{ext}"
        embedding_path = os.path.join(embeddings_dir, embedding_file_name)
        
        if os.path.isfile(original_path) and os.path.isfile(embedding_path):
            print(f"Working on file: {file_name}")
            counter += 1
            original_lines = count_lines(original_path)
            embedding_lines = count_lines(embedding_path)
            
            if original_lines != embedding_lines:
                mismatched_files.append(file_name)
    print(f"Counter: {counter}")
    for file in mismatched_files:
        print(file)
    print(f"Total mismatched files: {len(mismatched_files)}")

import os
import shutil
import sys

def move_files(file_list, source_dir, destination_dir):
      # Ensure source directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    moved_count = 0
    mismatched_files = []

    # Read the file list and move files
    with open(file_list, "r", encoding="utf-8") as f:
        for line in f:
            basename = line.strip()  # Trim spaces and newlines

            if not basename:
                continue  # Skip empty lines

            # Extract file name and extension
            base_name, ext = os.path.splitext(basename)

            # Append "_embedded" before the extension
            embedded_file_name = f"{base_name}_embedded{ext}"
            print (embedded_file_name)
            src_file = os.path.join(source_dir, basename)
            dest_file = os.path.join(destination_dir, embedded_file_name)

            if os.path.exists(src_file):
                shutil.move(src_file, dest_file)
                print(f"Moved: {src_file} -> {dest_file}")
                moved_count += 1
            else:
                print(f"Warning: File '{src_file}' not found.")
                mismatched_files.append(basename)

    print(f"\nFile moving process completed. Total files moved: {moved_count}")
    
    if mismatched_files:
        print("\nFiles not found:")
        for file in mismatched_files:
            print(file)



if __name__ == "__main__":
    # check_files()
    move_files("/home/dsi/orrbavly/GNN_project/testing_scripts/scripts/list_to_move.txt",  "/dsi/scratch/home/dsi/orrbavly/corona_data/embeddings",
                "/dsi/scratch/home/dsi/orrbavly/corona_data/watchlist_embedding")
