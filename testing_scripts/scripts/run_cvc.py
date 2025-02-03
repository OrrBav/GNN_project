import os, sys
import pandas as pd
import time
import torch
CSV_FOLDER = "/dsi/sbm/OrrBavly/kidney_data/downsamples_clonotype_21355/"
# /dsi/sbm/or/for_sol/downsampled/TRA/
OUTPUT_FOLDER =  "/dsi/sbm/OrrBavly/kidney_data/downsamples_clonotype_21355/embeddings/"
FILE_STRING_EXT = "_TRA_mig_cdr3_clones_all.txt"
VALID_COLUMN_VALS = ['T1', 'T2', 'T3']
HIGH_INDICATOR = 'T3'
META_ROW = 'T'
GPU_DEVICE = 0


def run_embedding(filtered_df, output_path):
    # Create embeddings
    embed_wrap = EmbeddingWrapper(TRANSFORMER_TO_USE, DEVICE, filtered_df, batch_size=256, method="mean", layers=[-1], pbar=True, max_len=120)
    tcr_embeddings_df = pd.DataFrame(embed_wrap.embeddings)

    # Extract the 'Sequences' column from tcrb_data
    sequences = filtered_df['Sequences']
    # Insert the 'Sequences' column into tcrb_embeddings_df at the first position
    tcr_embeddings_df.insert(0, 'Sequences', sequences)
    save_csv(tcr_embeddings_df, output_path)

def save_csv(tcr_embeddings_df, output_path, OUTPUT_FORMAT='csv'):
    # output embeddings
    if OUTPUT_FORMAT == "csv":
        tcr_embeddings_df.to_csv(output_path, index=False)
    elif OUTPUT_FORMAT == "pickle":
        import pickle
        with open(output_path, 'wb') as handle:
            pickle.dump(tcr_embeddings_df, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_kidney():
    input_path = CSV_FOLDER
    files = os.listdir(input_path)
    output_folder = OUTPUT_FOLDER
    counter = 1
    for file in files:
        file_path = os.path.join(input_path, f"{file}")
        if os.path.isfile(file_path):    
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            if 'aminoAcid' in df.columns:
                sequences_df = df[['aminoAcid']].rename(columns={'aminoAcid': 'Sequences'})
                # Check if the output file already exists
                output_path = f"{output_folder}/{file}"
                if os.path.exists(output_path):
                    print(f"Skipping {file}, file already exists.")
                    continue  # Skip the sample if the file already exists
                
                print(f"working on: {file.strip('.csv')}\tnumber {counter}")
                run_embedding(sequences_df, output_path)
                counter +=1
            else:
                print(f"The input file {file} does not contain an 'aminoAcid' column.")


# Function to filter meta_df based on file name and 'N' column
def file_is_valid(file, meta_df, valid_n_values, invalid_files):
    # Remove the suffix from the file name to match with 'filename' in meta_df
    filename = file.replace(f'{FILE_STRING_EXT}', '')
    
    # Check if the file is in the invalid_files list
    if filename in invalid_files:
        print(f"{filename} is in invalid files list")
        return False

    # Filter the meta_df to find the row matching the file
    meta_row = meta_df[meta_df['filename'].str.contains(filename, na=False)]

    # Check if the 'N' column has valid values and return True if valid
    if not meta_row.empty and meta_row[META_ROW].iloc[0] in valid_n_values:
        return True
    
    print(f"{filename} has an invalid value in metadata column")
    return False

def load_colon():
    # the sequences in the csv file need to be in a column called Sequences
    input_path = CSV_FOLDER
    files = os.listdir(input_path)
    output_folder = OUTPUT_FOLDER
    meta_df = pd.read_csv("/home/dsi/orrbavly/GNN_project/data/colon_meta.csv")
    valid_n_values = VALID_COLUMN_VALS
    # patint files with invalid annotations (marked red in Excel file)
    invalid_files = [
    'pool1_S24', 'pool2_S22', 'pool3_S22', 'pool3_S3', 'pool4_S4', 
    'pool5_S20', 'pool6_S12', 'pool7_S22', 'pool9_S8'
    ]
    # files that appear in metadata files but not in final downsamples files, probebly because did not undergo mixcr. should also be ignored?
    pool_files_not_in_metadata = [
    'pool8_S5', 'pool8_S7', 'pool8_S3', 'pool3_S23', 'pool4_S17', 'pool3_S19',
    'pool8_S24', 'pool8_S14', 'pool2_S18', 'pool1_S6', 'pool8_S4', 'pool7_S20', 'pool7_S1'
    ]

    counter = 1
    low_counter = 0
    high_counter = 0

    print(f"~~~~~~~Starting Work~~~~~~~\noutput files will be saved in: {OUTPUT_FOLDER}")
    for file in files:
        # Construct full file path
        # FILES can be either TRA or TRB. change dir string accordingly.
        file_path = os.path.join(input_path, f"{file}") # should add {FILE_STRING_EXT}?
        # if file.endswith('.txt') and file_is_valid(file, meta_df, valid_n_values, invalid_files):

        # Check if the file is a CSV
        if file_is_valid(file, meta_df, valid_n_values, invalid_files):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path, delimiter="\t")
            # Renaming the 'CDR3.aa' column to 'sequences'
            df.rename(columns={'CDR3.aa': 'Sequences'}, inplace=True)
            # Run the embedding function with the DataFrame and file name
            filename = file.replace(f"{FILE_STRING_EXT}", "")
            risk = 'high' if meta_df.loc[meta_df['filename'].str.contains(filename, na=False), f'{META_ROW}'].iloc[0] == HIGH_INDICATOR else 'low'
            output_path = f"{output_folder}/{filename}_{risk}.csv"
            
            # Check if the output file already exists
            if os.path.exists(output_path):
                print(f"Skipping {filename}, file already exists.")
                continue  # Skip the sample if the file already exists
            
            print(f"working on: {filename}\tnumber {counter}")
            run_embedding(df, output_path)
            counter +=1

            # if risk == 'low':
            #     low_counter += 1
            # elif risk == 'high':
            #     high_counter += 1
        else:
            print(f"skipping invalid file: {file}")
    # print(f"There are {low_counter} low labled files, \nand there are {high_counter} high labled files")


def rename_files(): 
    FILE_STRING_EXT = ".csv"  # Extension of your files
    input_folder = "/dsi/sbm/OrrBavly/colon_data/embeddings/T_TRA/"
    meta_df = pd.read_csv("/home/dsi/orrbavly/GNN_project/data/colon_meta.csv")
    # Iterate over the files in the directory
    for file in os.listdir(input_folder):
        # Process only CSV files
        if file.endswith(FILE_STRING_EXT):
            # Extract samplename and current risk from the file name
            filename, current_risk = file.replace(FILE_STRING_EXT, "").rsplit("_", 1)

            # Build the pattern to match in the meta_df (e.g., `pool5_S21_TRB_mig_cdr3_clones_all`)
            meta_pattern = f"{filename}_TRB_mig_cdr3_clones_all"

            # Recalculate the risk based on the metadata
            try:
                calculated_risk = 'high' if meta_df.loc[
                    meta_df['filename'] == meta_pattern, f'{META_ROW}'
                ].iloc[0] == HIGH_INDICATOR else 'low'
            except IndexError:
                print(f"Warning: No match found for {filename} in metadata. Skipping.")
                continue

            # If the calculated risk differs from the current risk, rename the file
            if calculated_risk != current_risk:
                new_file_name = f"{filename}_{calculated_risk}{FILE_STRING_EXT}"
                old_file_path = os.path.join(input_folder, file)
                new_file_path = os.path.join(input_folder, new_file_name)

                # Rename the file to the correct risk
                os.rename(old_file_path, new_file_path)
                print(f"Renamed {file} to {new_file_name}")
    
def load_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        print("CUDA is not available, using CPU for training.")
    else:
        # Print available devices
        num_devices = torch.cuda.device_count()
        print(f"CUDA is available. Number of devices: {num_devices}")

        # Try connecting to the specific device
        try:
            torch.cuda.set_device(GPU_DEVICE)  # SET GPU INDEX HERE:
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"Using GPU device {current_device}: {device_name}")
        except Exception as e:
            print(f"Failed to connect to GPU: {e}")
            device = torch.device("cpu")

    print(f"Using device: {device}")


if __name__ == '__main__':
    start = time.time()
    load_gpu()
    # Change working directory
    project_root = '/home/dsi/orrbavly/GNN_project/CVC'
    os.chdir(project_root)

    # Add the project root to sys.path
    sys.path.append(project_root)

    SRC_DIR = "cvc"
    assert os.path.isdir(SRC_DIR), f"Cannot find src dir: {SRC_DIR}"
    sys.path.append(SRC_DIR)

    from cvc import model_utils

    from lab_notebooks.utils import SC_TRANSFORMER, TRANSFORMER, DEVICE
    MODEL_DIR = os.path.join(SRC_DIR, "models")
    sys.path.append(MODEL_DIR)

    FILT_EDIT_DIST = True
    from lab_notebooks.utils import HOME_DIR_GCP
    HOME_DIR_GCP
    # CVC - TRANSFORMER
    # scCVC - SC_TRANSFORMER
    TRANSFORMER_TO_USE = SC_TRANSFORMER
    from cvc.embbeding_wrapper import EmbeddingWrapper

    print("Succesully loaded the model\nstarting files...")
    load_kidney()
    end = time.time()
    print(f"Time running script: {(end - start) / 60}")