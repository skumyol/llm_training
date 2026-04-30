#!/usr/bin/env python3
"""Check GPU availability and torch configuration before training."""

import sys
import subprocess

print("=" * 60)
print("GPU AVAILABILITY CHECK")
print("=" * 60)

# Check 1: nvidia-smi
print("\n[1] nvidia-smi output:")
print("-" * 40)
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if result.returncode == 0:
    print(result.stdout)
else:
    print("ERROR: nvidia-smi not found or failed")
    print("stderr:", result.stderr)

# Check 2: PyTorch CUDA
print("\n[2] PyTorch CUDA status:")
print("-" * 40)
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"    Compute capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
            
            # Current memory usage
            print(f"    Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"    Reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    else:
        print("  WARNING: CUDA not available - training will use CPU (very slow!)")
        
except ImportError:
    print("ERROR: PyTorch not installed")
    sys.exit(1)

# Check 3: transformers/accelerate
print("\n[3] Transformers/Accelerate status:")
print("-" * 40)
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError:
    print("Transformers not installed")

try:
    import accelerate
    print(f"Accelerate version: {accelerate.__version__}")
    from accelerate import Accelerator
    acc = Accelerator()
    print(f"  Device: {acc.device}")
except ImportError:
    print("Accelerate not installed")

# Check 4: Test tensor creation
print("\n[4] GPU test (creating test tensor):")
print("-" * 40)
if torch.cuda.is_available():
    try:
        test_tensor = torch.randn(1000, 1000).cuda()
        result = torch.matmul(test_tensor, test_tensor)
        print(f"  Test tensor created on GPU: {test_tensor.device}")
        print(f"  Test matmul successful: shape {result.shape}")
        del test_tensor, result
        torch.cuda.empty_cache()
        print("  GPU is working!")
    except Exception as e:
        print(f"  ERROR: GPU test failed - {e}")
else:
    print("  Skipping (no CUDA)")

print("\n" + "=" * 60)
if torch.cuda.is_available():
    print("RESULT: GPU is available and working")
    print("Training should use: device='cuda'")
else:
    print("RESULT: NO GPU - training will be extremely slow on CPU")
    print("Consider: Check CUDA installation, drivers, or use CPU config")
print("=" * 60)
