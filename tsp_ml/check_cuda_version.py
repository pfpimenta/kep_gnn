import torch

is_cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {is_cuda_available}")

if is_cuda_available:
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
