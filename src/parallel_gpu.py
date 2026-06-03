import torch
# If you use the Intel Extension for PyTorch (IPEX)
import intel_extension_for_pytorch as ipex

print(f"Is Arc GPU available? {torch.xpu.is_available()}")
print(f"Device Name: {torch.xpu.get_device_name(0)}")