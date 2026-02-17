import torch

def check_gpu():
    """Check GPU availability and display information."""
    print("GPU Availability Check")
    print("=" * 50)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # Number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        # GPU details
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"\nGPU {i}: {gpu_name}")
            print(f"Total Memory: {gpu_memory:.2f} GB")
        
        # Current GPU
        current_gpu = torch.cuda.current_device()
        print(f"\nCurrent GPU: {current_gpu}")
    else:
        print("No GPU detected. Using CPU only.")
    
    print("=" * 50)

if __name__ == "__main__":
    check_gpu()