"""
Simple script to test PyTorch installation and capabilities
"""

print("Testing PyTorch installation...\n")

# Test 1: Can we import PyTorch?
try:
    import torch
    print("✓ PyTorch is installed!")
    print(f"  Version: {torch.__version__}")
except ImportError:
    print("✗ PyTorch is NOT installed")
    print("  Install with: pip install torch")
    exit(1)

print()

# Test 2: Check for GPU/CUDA
print("GPU/CUDA Status:")
if torch.cuda.is_available():
    print(f"✓ CUDA is available!")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("✗ No CUDA GPU detected - will use CPU")
    print("  (This is fine for toy examples!)")

print()

# Test 3: Check for MPS (Apple Silicon)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("✓ MPS (Apple Silicon GPU) is available!")
else:
    print("✗ MPS not available (not on Apple Silicon)")

print()

# Test 4: Basic computation test
print("Basic computation test:")
try:
    # CPU test
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = x @ y  # Matrix multiplication
    print(f"✓ CPU computation works!")
    print(f"  Created tensors, did matrix multiply")
    
    # GPU test (if available)
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        y_gpu = y.cuda()
        z_gpu = x_gpu @ y_gpu
        print(f"✓ GPU computation works!")
    
except Exception as e:
    print(f"✗ Computation failed: {e}")

print()

# Test 5: Recommended device
print("Recommended device for training:")
if torch.cuda.is_available():
    device = "cuda"
    print(f"  Use: device = 'cuda'")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    print(f"  Use: device = 'mps'")
else:
    device = "cpu"
    print(f"  Use: device = 'cpu'")

print()

# Test 6: Quick neural network test
print("Neural network test:")
try:
    import torch.nn as nn
    
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    x = torch.randn(32, 10)  # Batch of 32
    output = model(x)
    
    print(f"✓ Neural network works!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
except Exception as e:
    print(f"✗ Neural network failed: {e}")

print()
print("=" * 50)
print("Summary:")
print(f"PyTorch {torch.__version__} is ready to use!")
print(f"Recommended device: {device}")
if device == "cpu":
    print("Note: Training will be slower on CPU, but fine for toy examples")
print("=" * 50)