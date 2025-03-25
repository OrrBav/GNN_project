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
INPUT_DIR="/dsi/scratch/home/dsi/orrbavly/corona_data/original_files"
OUTPUT_DIR="/dsi/scratch/home/dsi/orrbavly/corona_data/embeddings"
MISSING_FILES="/home/dsi/orrbavly/GNN_project/testing_scripts/scripts/missing_files.txt"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Check if missing_files.txt exists
if [[ ! -f "$MISSING_FILES" ]]; then
    echo "Error: File '$MISSING_FILES' not found!"
    exit 1
fi

# Loop through each file in missing_files.txt
while IFS= read -r file_name || [[ -n "$file_name" ]]; do
    # Trim any whitespace or extra characters
    file_name=$(echo "$file_name" | xargs)

    # Define input and output file paths
    input_file="$INPUT_DIR/$file_name"
    base_name=$(basename "$file_name" .csv)
    output_file="$OUTPUT_DIR/${base_name}_embedded.csv"

    # Check if the input file exists
    if [[ ! -f "$input_file" ]]; then
        echo "Warning: Input file '$input_file' not found. Skipping."
        continue
    fi

    # Check if the output file already exists
    if [[ -f "$output_file" ]]; then
        echo "Skipping $file_name (already processed)."
        continue
    fi

    echo "Processing $file_name"

    # Run the Docker command for the current file
    docker run --gpus all -it \
        -v "$INPUT_DIR:/app/data" \
        -v "$OUTPUT_DIR:/app/output" \
        cvc_env_saved \
        --input_path "/app/data/${file_name}" \
        --output_path "/app/output/$(basename "$output_file")"

done < "$MISSING_FILES"

echo "Processing of missing files completed successfully."
