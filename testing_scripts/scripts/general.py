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

def rename_corona():
    import os
    import pandas as pd
    import re

    # Define paths
    metadata_file = "/home/dsi/orrbavly/GNN_project/data/corona_metadata.csv"  # Path to metadata CSV
    embeddings_dir = "/dsi/scratch/home/dsi/orrbavly/corona_data/embeddings/"  # Directory containing embedding files
    # Load metadata
    df = pd.read_csv(metadata_file, dtype={"longer_sample_name": str, "Biological Sex": str})  # Read as str to avoid NaN issues

    # Map Biological Sex values
    sex_map = {"Female": "F", "Male": "M"}

    # Create sets to track files
    expected_files = set()
    sex_non_list = []  # List to store files missing sex value
    corrected_files = []  # List to track files that needed correction

    # Regular expression to find existing sex suffixes (_M, _F, _noSex)
    sex_pattern = re.compile(r"_(M|F|noSex)\.csv$")

    # Rename files based on metadata
    for _, row in df.iterrows():
        sample_name = row["longer_sample_name"]

        # Ensure sample_name exists and correct its extension
        if pd.isna(sample_name):
            print("Skipping row with missing sample name.")
            continue

        # Match filenames in embeddings folder (metadata names use .tsv, but the actual files have _embedded.csv)
        sample_name_embedded = sample_name.replace(".tsv", ".csv")  

        # Get sex from metadata, handling missing values
        sex = str(row["Biological Sex"]).strip() if pd.notna(row["Biological Sex"]) else None
        correct_sex = sex_map.get(sex, "noSex")  # Map to "F"/"M" or "noSex" if missing

        # Check if the file exists
        old_path = os.path.join(embeddings_dir, sample_name_embedded)
        if not os.path.exists(old_path):
            print(f"File not found: {old_path}")
            continue

        # Remove _embedded from filename and check if it already has _M, _F, or _noSex
        clean_name = sample_name.replace(".tsv", "")  # Remove .tsv extension
        current_sex_match = sex_pattern.search(clean_name)

        if current_sex_match:
            current_sex = current_sex_match.group(1)  # Extract existing _M/_F/_noSex
            if current_sex != correct_sex:
                print(f"❌ Incorrect sex in filename: {clean_name} → Changing to _{correct_sex}.csv")
                corrected_files.append(clean_name)
                clean_name = re.sub(sex_pattern, f"_{correct_sex}.csv", clean_name)  # Replace wrong suffix
            else:
                print(f"✅ Correct sex in filename: {clean_name}")
        else:
            # If no sex suffix exists, add the correct one
            print(f"⚠️ Missing sex suffix: {clean_name} → Adding _{correct_sex}.csv")
            clean_name += f"_{correct_sex}.csv"

        # Full new path
        new_path = os.path.join(embeddings_dir, clean_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"🔄 Renamed: {old_path} → {new_path}")

        # Track expected files
        expected_files.add(clean_name)

    # Check for unexpected files in the embeddings directory
    existing_files = set(f for f in os.listdir(embeddings_dir) if f.endswith(".csv"))
    unexpected_files = existing_files - expected_files  # Files not in metadata

    # Print unexpected files
    if unexpected_files:
        print("\n⚠️ Unexpected files found in embeddings directory (not in metadata):")
        for file in unexpected_files:
            print(f" - {file}")
    else:
        print("\n✅ No unexpected files found in embeddings directory.")

    # Print files that were corrected
    if corrected_files:
        print("\n🔄 Files that had incorrect sex and were renamed:")
        for file in corrected_files:
            print(f" - {file}")
    else:
        print("\n✅ No files needed correction for sex.")

    print("Processing complete.")



def faissg():
    import faiss
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set GPU 0 as visible

    res = faiss.StandardGpuResources()
    print("FAISS is now using GPU!")

def check():
    import os
    import pandas as pd
    import re

    # Define paths
    metadata_file = "/home/dsi/orrbavly/GNN_project/data/corona_metadata.csv"  # Path to metadata CSV
    embeddings_dir = "/dsi/scratch/home/dsi/orrbavly/corona_data/embeddings/"  # Directory containing embedding files

    # Load metadata
    df = pd.read_csv(metadata_file, dtype={"longer_sample_name": str, "Biological Sex": str})  # Read as str to avoid NaN issues

    # Regex to remove sex suffix (_M, _F, _noSex) from filenames
    sex_pattern = re.compile(r"_(M|F|noSex)\.csv$")

    # Initialize counters
    male_count = 0
    female_count = 0
    missing_count = 0

    # Iterate over embedding files
    for filename in os.listdir(embeddings_dir):
        if filename.endswith(".csv"):  # Process only CSV files
            # Remove sex suffix and change extension to .tsv
            clean_name = re.sub(sex_pattern, ".tsv", filename)

            # Check occurrences in metadata
            matched_rows = df[df["longer_sample_name"] == clean_name]

            if not matched_rows.empty:
                sex_values = matched_rows["Biological Sex"].dropna().unique()

                if "Male" in sex_values:
                    male_count += len(matched_rows)
                elif "Female" in sex_values:
                    female_count += len(matched_rows)
                else:
                    missing_count += len(matched_rows)
            else:
                missing_count += 1  # If not found in metadata, count as missing

    # Print results
    print(f"Metadata Occurrences:")
    print(f"Male count: {male_count}")
    print(f"Female count: {female_count}")
    print(f" Missing (NA) count: {missing_count}")

    print("\nProcessing complete.")



def seq_to_emb():
    import os
    import csv

    # Directories
    tcr_dir = "/dsi/scratch/home/dsi/orrbavly/corona_data/original_files/"
    embedding_dir = "/dsi/scratch/home/dsi/orrbavly/corona_data/embeddings/"
 
    # Get list of TCR sequence files
    tcr_files = [f for f in os.listdir(tcr_dir) if f.endswith(".csv")]

    # Counters and lists for statistics
    processed_count = 0
    skipped_files = []  # List to store skipped file names

    for tcr_file in tcr_files:
        # Construct possible embedding file names
        base_name = tcr_file.replace("_TCRB.csv", "")
        embedding_files = [
            f"{base_name}_TCRB_M.csv",
            f"{base_name}_TCRB_F.csv"
        ]

        # Find the corresponding embedding file
        matching_embedding_file = None
        for emb_file in embedding_files:
            if emb_file in os.listdir(embedding_dir):
                matching_embedding_file = emb_file
                break

        if not matching_embedding_file:
            print(f"Embedding file not found for {tcr_file}, skipping immediately.")
            skipped_files.append(tcr_file)
            continue  # Skip this file immediately

        # File paths
        tcr_path = os.path.join(tcr_dir, tcr_file)
        emb_path = os.path.join(embedding_dir, matching_embedding_file)
        temp_output_path = emb_path + ".tmp"  # Temporary file for safe overwriting

        # Check if the first column is already "Sequences"
        with open(emb_path, "r") as emb_f:
            first_line = emb_f.readline().strip().split(",")  # Read first row
            if first_line and first_line[0] == "Sequences":
                print(f"Skipping {matching_embedding_file}, 'Sequences' column already exists.")
                skipped_files.append(matching_embedding_file)
                continue  # Skip this file immediately

        # Check if TCR file is empty
        if os.stat(tcr_path).st_size == 0:
            print(f"Warning: {tcr_file} is empty, skipping.")
            skipped_files.append(tcr_file)
            continue  # Skip to the next file

        # Open both files using csv.reader and csv.writer
        try:
            with open(tcr_path, "r") as tcr_f, open(emb_path, "r") as emb_f, open(temp_output_path, "w", newline='') as out_f:
                tcr_reader = csv.reader(tcr_f)
                emb_reader = csv.reader(emb_f)
                writer = csv.writer(out_f)

                # Read and discard the first row in the embedding file (column numbers)
                embedding_header = next(emb_reader, None)  # Ignore first row (0,1,2,...,767)
                if embedding_header is None:
                    print(f"Skipping {matching_embedding_file}, embedding file is empty.")
                    skipped_files.append(matching_embedding_file)
                    continue  # Skip this file

                # Read the first row of TCR file
                tcr_header = next(tcr_reader, None)
                if tcr_header is None:
                    print(f"Skipping {matching_embedding_file}, TCR file is empty.")
                    skipped_files.append(matching_embedding_file)
                    continue  # Skip this file

                # Manually create the correct header
                embedding_dim = len(embedding_header)
                new_header = ["Sequences"] + [str(i) for i in range(embedding_dim)]
                writer.writerow(new_header)

                # Debugging: Print what is being processed
                print(f"Processing {matching_embedding_file} (embedding file has {embedding_dim} columns)")

                # Process rows one by one
                row_count_tcr = 0
                row_count_emb = 0
                wrote_data = False  # Track if we wrote anything

                for tcr_row, emb_row in zip(tcr_reader, emb_reader):
                    if not tcr_row or not emb_row:
                        continue  # Skip empty rows

                    new_row = [tcr_row[0]] + emb_row  # Add Sequences column

                    # Ensure the output row has exactly 769 columns
                    if len(new_row) != 769:
                        print(f"Column count mismatch in {matching_embedding_file} (Expected 769, got {len(new_row)}), skipping update.")
                        skipped_files.append(matching_embedding_file)
                        os.remove(temp_output_path)  # Immediately delete temp file
                        break  # Skip to the next file

                    writer.writerow(new_row)
                    row_count_tcr += 1
                    row_count_emb += 1
                    wrote_data = True  # We wrote at least one valid row

                # If we didn't write any data, delete the temp file
                if not wrote_data:
                    print(f"No data written for {matching_embedding_file}, deleting temp file.")
                    os.remove(temp_output_path)
                    continue  # Skip this file

                # Check row count
                if row_count_tcr != row_count_emb:
                    print(f"Row mismatch for {tcr_file} ({row_count_tcr} vs {row_count_emb}), skipping update.")
                    skipped_files.append(tcr_file)
                    os.remove(temp_output_path)  # Immediately delete temp file
                    continue  # Skip this file

            # Safely replace the old file
            os.remove(emb_path)  # Delete original file
            os.rename(temp_output_path, emb_path)  # Move new file in place

            print(f"Updated {matching_embedding_file} successfully.")
            processed_count += 1

        except OSError as e:
            print(f"Error processing {matching_embedding_file}: {e}")
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)  # Immediately delete temp file
            skipped_files.append(matching_embedding_file)
            continue  # Skip this file

    # Print summary
    print("\nProcessing completed.")
    print(f"Successfully updated: {processed_count}")
    print(f"Skipped due to issues: {len(skipped_files)}")

    # Print skipped file names immediately
    if skipped_files:
        print("\nSkipped Files:")
        for file in skipped_files:
            print(file)


if __name__ == "__main__":
    # check_files()
    # move_files("/home/dsi/orrbavly/GNN_project/testing_scripts/scripts/list_to_move.txt",  "/dsi/scratch/home/dsi/orrbavly/corona_data/embeddings",
    #             "/dsi/scratch/home/dsi/orrbavly/corona_data/watchlist_embedding")
    seq_to_emb()

