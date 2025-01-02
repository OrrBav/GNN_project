import kagglehub

# Download latest version
path = kagglehub.dataset_download("arnaud58/landscape-pictures")

print("Path to dataset files:", path)