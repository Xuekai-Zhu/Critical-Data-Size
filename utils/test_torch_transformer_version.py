# Python code to check the installed versions of transformers and torch libraries
try:
    import transformers
    transformers_version = transformers.__version__
except ImportError:
    transformers_version = "transformers library is not installed"

try:
    import torch
    torch_version = torch.__version__
except ImportError:
    torch_version = "torch library is not installed"

print("Transformers version:", transformers_version)
print("Torch version:", torch_version)
