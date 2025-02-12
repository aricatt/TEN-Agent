import torch

print('CUDA是否可用:', torch.cuda.is_available())
print('PyTorch版本:', torch.__version__)
print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else 'NA')
if torch.cuda.is_available():
    print('当前CUDA设备:', torch.cuda.get_device_name(0))
