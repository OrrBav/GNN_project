import torch
### MY addition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cpu":
    print("CUDA is not available, using CPU for training.")
else:
    # Print available devices
    num_devices = torch.cuda.device_count()
    print(f"CUDA is available. Number of devices: {num_devices}")

    # Try connecting to the specific device
    try:
        torch.cuda.set_device(1)  # SET GPU INDEX HERE:
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Using GPU device {current_device}: {device_name}")
    except Exception as e:
        device = torch.device("cpu")