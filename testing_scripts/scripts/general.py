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
    """
    This function adds the correct Sequences column (from original_files directory) to corona embedding files. 
    """
    import os
    import csv
    import re

    # Directories
    original_dir = "/dsi/scratch/home/dsi/orrbavly/corona_data/original_files/"
    embedding_dir = "/dsi/scratch/home/dsi/orrbavly/corona_data/embeddings/"
    output_dir = "/dsi/scratch/home/dsi/orrbavly/corona_data/embeddings_new/"   
    temp_dir = "/dsi/sbm/OrrBavly/corona_data/temp_embeddings/"
    report_path = "/dsi/sbm/OrrBavly/corona_data/merge_report.txt"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Track status
    processed_files = []
    skipped_files = []

    # Logging helper
    def log(message):
        print(message)
        with open(report_path, "a") as log_file:
            log_file.write(message + "\n")

    def merge_large_csvs(original_path, temp_emb_path, output_path, emb_file):
        try:
            if os.stat(original_path).st_size == 0:
                log(f"⚠️ Skipping empty original file: {original_path}")
                skipped_files.append(original_path)
                return

            with open(original_path, 'r') as f_seq, open(temp_emb_path, 'r') as f_emb, open(output_path, 'w', newline='') as f_out:
                reader_seq = csv.reader(f_seq)
                reader_emb = csv.reader(f_emb)
                writer = csv.writer(f_out)

                seq_header = next(reader_seq)
                emb_header = next(reader_emb)

                if "Sequences" in emb_header:
                    log(f"⚠️ Skipping {emb_file} — already contains 'Sequences'.")
                    skipped_files.append(emb_file)
                    return

                writer.writerow(["Sequences"] + emb_header)

                row_count = 0
                for seq_row, emb_row in zip(reader_seq, reader_emb):
                    if not seq_row or not emb_row:
                        continue

                    new_row = [seq_row[0].strip()] + emb_row
                    if len(new_row) != 769:
                        log(f"❌ Column mismatch in {emb_file}: got {len(new_row)}, expected 769. Deleting output.")
                        os.remove(output_path)
                        skipped_files.append(emb_file)
                        return

                    writer.writerow(new_row)
                    row_count += 1

            written_lines = sum(1 for _ in open(output_path)) - 1
            if written_lines != row_count:
                log(f"❌ Row mismatch in {output_path}: wrote {written_lines}, expected {row_count}. Deleting output.")
                os.remove(output_path)
                skipped_files.append(emb_file)
                return

            log(f"✅ Merged: {output_path}")
            processed_files.append(emb_file)

        except Exception as e:
            log(f"❌ Error processing {emb_file}: {e}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    log(f"🗑️ Deleted incomplete file: {output_path}")
                except Exception as del_e:
                    log(f"⚠️ Could not delete output file: {del_e}")
            skipped_files.append(emb_file)

    # Clear previous log
    open(report_path, "w").close()

    # Main loop
    for emb_file in os.listdir(embedding_dir):
        if emb_file.endswith(".csv") and ("_F.csv" in emb_file or "_M.csv" in emb_file):
            match = re.match(r"^(\d+_TCRB)_([FM])\.csv$", emb_file)
            if not match:
                log(f"⚠️ Invalid file name: {emb_file}")
                skipped_files.append(emb_file)
                continue

            base_id, gender = match.groups()
            embedding_path = os.path.join(embedding_dir, emb_file)
            original_path = os.path.join(original_dir, f"{base_id}.csv")
            output_file = f"{base_id}_{gender}_seqs.csv"
            output_path = os.path.join(output_dir, output_file)
            temp_emb_path = os.path.join(temp_dir, emb_file)

            if os.path.exists(output_path):
                log(f"🔁 Output already exists: {output_path}")
                continue

            if not os.path.exists(original_path):
                log(f"❌ Missing original: {original_path}")
                skipped_files.append(emb_file)
                continue

            try:
                shutil.copy2(embedding_path, temp_emb_path)
                os.remove(embedding_path)
            except Exception as copy_err:
                log(f"❌ Could not prepare temp file for {emb_file}: {copy_err}")
                skipped_files.append(emb_file)
                continue

            merge_large_csvs(original_path, temp_emb_path, output_path, emb_file)

            if os.path.exists(temp_emb_path):
                os.remove(temp_emb_path)

    # Final summary
    log("\n✅ Processing complete.")
    log(f"Files successfully processed: {len(processed_files)}")
    log(f"Files skipped or errored: {len(skipped_files)}")

    if processed_files:
        log("\n📦 Processed files:")
        for f in processed_files:
            log(f"  - {f}")

    if skipped_files:
        log("\n🔎 Skipped files:")
        for f in skipped_files:
            log(f"  - {f}")


if __name__ == "__main__":
    # check_files()
    # move_files("/home/dsi/orrbavly/GNN_project/testing_scripts/scripts/list_to_move.txt",  "/dsi/scratch/home/dsi/orrbavly/corona_data/embeddings",
    #             "/dsi/scratch/home/dsi/orrbavly/corona_data/watchlist_embedding")
    seq_to_emb()

