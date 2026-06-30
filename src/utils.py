import os
import torch
import random
import numpy as np
import gc

def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy` and `torch`.

    Args:
        seed (int): The seed to set.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # num_cuda_devices = 1 # number of gpus to use
        # torch.cuda.set_device(0)  # Set specific GPU device to use
        # torch.set_float32_matmul_precision('high') # Mixed precicion setting to use TensorFloat32 (TF32) mode.
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        gc.collect()
        torch.mps.empty_cache()
    else:
        device = torch.device("cpu")

    # Dataloader variables
    pin_memory = (device.type == "cuda")  # Speeds up transfering dataset from CPU to GPU
    # num_workers = X

    print(f"Using device: {device}")

    return device, pin_memory#, num_cuda_devices, num_workers