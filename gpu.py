import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    # 显示可用的GPU设备
    device = torch.device("cuda")
    print("可用的GPU设备：", torch.cuda.get_device_name(0))
else:
    print("没有可用的GPU设备，将使用CPU。")
    device = torch.device("cpu")