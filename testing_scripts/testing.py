import torch

# Get the name of the GPU device
if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name(0))
    print("Number of GPUs available:", torch.cuda.device_count())
else:
    print("No GPU available.")
