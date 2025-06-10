# #!/bin/bash

# # Define input and output directories
# INPUT_DIR="/dsi/scratch/home/dsi/orrbavly/corona_data/original_files"
# OUTPUT_DIR="/dsi/scratch/home/dsi/orrbavly/corona_data/embeddings"

# # Ensure the output directory exists
# # mkdir -p "$OUTPUT_DIR"

# # Loop through all .tsv files in the input directory
# for file in "$INPUT_DIR"/*.csv; do
#     # Extract the base file name without extension (e.g., "file")
#     base_name=$(basename "$file" .csv)

#     # Define the output file path with .csv extension
#     output_file="$OUTPUT_DIR/${base_name}_embedded.csv"

#     # Check if the output file already exists
#     if [[ -f "$output_file" ]]; then
#         echo "Skipping $base_name (already processed)."
#         continue
#     fi

#     echo "working on $base_name"
#     # Run the Docker command for the current file
#     docker run --gpus all -it \
#         -v "$INPUT_DIR:/app/data" \
#         -v "$OUTPUT_DIR:/app/output" \
#         cvc_env_saved \
#         --input_path "/app/data/${base_name}.csv" \
#         --output_path "/app/output/$(basename "$output_file")"
# done

# echo "All files processed successfully and saved as CSV."
#!/bin/bash

# Define input and output directories
INPUT_DIR="/dsi/sbm/OrrBavly/colon_data/new_mixcr/TRB/downsamples_209378/original_data/"
OUTPUT_DIR="/dsi/sbm/OrrBavly/colon_data/new_mixcr/TRB/downsamples_209378/embeddings/"
# MISSING_FILES="/home/dsi/orrbavly/GNN_project/testing_scripts/scripts/missing_files.txt"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Loop through each CSV file in INPUT_DIR
for input_file in "$INPUT_DIR"/*.csv; do
    file_name=$(basename "$input_file")
    base_name=$(basename "$file_name" .csv)
    output_file="$OUTPUT_DIR/${base_name}_embedded.csv"

    # Skip if the output already exists
    if [[ -f "$output_file" ]]; then
        echo "Skipping $file_name (already processed)."
        continue
    fi

    echo "Processing $file_name"

    # Run the Docker command for the current file
    docker run --rm --gpus all -it \
        -v "$INPUT_DIR:/app/data" \
        -v "$OUTPUT_DIR:/app/output" \
        cvc_env \
        --input_path "/app/data/${file_name}" \
        --model_type CVC \
        --output_path "/app/output/${base_name}_embedded.csv"

done

echo "Processing of all input files completed successfully."