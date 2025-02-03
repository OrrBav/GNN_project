#!/bin/bash

# Define input and output directories
INPUT_DIR="/dsi/scratch/home/dsi/orrbavly/corona_data/original_files"
OUTPUT_DIR="/dsi/scratch/home/dsi/orrbavly/corona_data/embeddings"

# Ensure the output directory exists
# mkdir -p "$OUTPUT_DIR"

# Loop through all .tsv files in the input directory
for file in "$INPUT_DIR"/*.csv; do
    # Extract the base file name without extension (e.g., "file")
    base_name=$(basename "$file" .csv)

    # Define the output file path with .csv extension
    output_file="$OUTPUT_DIR/${base_name}_embedded.csv"

    # Check if the output file already exists
    if [[ -f "$output_file" ]]; then
        echo "Skipping $base_name (already processed)."
        continue
    fi

    echo "working on $base_name"
    # Run the Docker command for the current file
    docker run --gpus all -it \
        -v "$INPUT_DIR:/app/data" \
        -v "$OUTPUT_DIR:/app/output" \
        cvc_env_saved \
        --input_path "/app/data/${base_name}.csv" \
        --output_path "/app/output/$(basename "$output_file")"
done

echo "All files processed successfully and saved as CSV."
