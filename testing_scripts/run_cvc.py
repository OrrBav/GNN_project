import os, sys
import pandas as pd
CSV_FOLDER = "/dsi/sbm/or/for_sol/downsampled/TRB/"
OUTPUT_FOLDER =  "/dsi/sbm/OrrBavly/colon_data/embeddings"


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


# Function to filter meta_df based on file name and 'N' column
def file_is_valid(file, meta_df, valid_n_values, invalid_files):
    # Remove the suffix from the file name to match with 'filename' in meta_df
    filename = file.replace('_TRB_mig_cdr3_clones_all.txt', '')
    
    # Check if the file is in the invalid_files list
    if filename in invalid_files:
        return False

    # Filter the meta_df to find the row matching the file
    meta_row = meta_df[meta_df['filename'].str.contains(filename, na=False)]

    # Check if the 'N' column has valid values and return True if valid
    if not meta_row.empty and meta_row['N'].iloc[0] in valid_n_values:
        return True
    
    return False

def load():
    # the sequences in the csv file need to be in a column called Sequences
    input_path = CSV_FOLDER
    files = os.listdir(input_path)
    output_folder = OUTPUT_FOLDER
    meta_df = pd.read_csv("/home/dsi/orrbavly/GNN_project/data/colon_meta.csv")
    valid_n_values = ['0', '1', '2', '1a', '1b']
    # patint files with invalid annotations
    invalid_files = [
    'pool1_S24', 'pool2_S22', 'pool3_S22', 'pool3_S3', 'pool4_S4', 
    'pool5_S20', 'pool6_S12', 'pool7_S22', 'pool9_S8'
    ]

    pool_files = [
    'pool8_S5', 'pool8_S7', 'pool8_S3', 'pool3_S23', 'pool4_S17', 'pool3_S19',
    'pool8_S24', 'pool8_S14', 'pool2_S18', 'pool1_S6', 'pool8_S4', 'pool7_S20', 'pool7_S1'
    ]

    counter = 1
    for file in pool_files:
        # Construct full file path
        file_path = os.path.join(input_path, f"{file}_TRB_mig_cdr3_clones_all.txt")
        #         if file.endswith('.txt') and file_is_valid(file, meta_df, valid_n_values, invalid_files):

        # Check if the file is a CSV
        if file_is_valid(file, meta_df, valid_n_values, invalid_files):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path, delimiter="\t")
            # Renaming the 'CDR3.aa' column to 'sequences'
            df.rename(columns={'CDR3.aa': 'Sequences'}, inplace=True)
            # Run the embedding function with the DataFrame and file name
            filename = file.replace("_TRB_mig_cdr3_clones_all.txt", "")
            risk = 'low' if meta_df.loc[meta_df['filename'].str.contains(filename, na=False), 'N'].iloc[0] == '0' else 'high'
            output_path = f"{output_folder}/{filename}_{risk}.csv"
            
            # Check if the output file already exists
            if os.path.exists(output_path):
                print(f"Skipping {filename}, file already exists.")
                continue  # Skip the sample if the file already exists
            
            print(f"working on: {filename}\tnumber {counter}")
            run_embedding(df, output_path)
            counter +=1
        else:
            print(f"skipping invalid file: {file}")


if __name__ == '__main__':
   # Change working directory
    project_root = '/home/dsi/orrbavly/GNN_project/CVC'
    os.chdir(project_root)

    # Add the project root to sys.path
    sys.path.append(project_root)
    import os, sys
    import pandas as pd
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
    TRANSFORMER_TO_USE = SC_TRANSFORMER
    from cvc.embbeding_wrapper import EmbeddingWrapper

    print("Succesully loaded the model\nstarting files...")
    load()