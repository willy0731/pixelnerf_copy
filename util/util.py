import cv2
import numpy as np
import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import functools
import math
import warnings

def get_cuda(gpu_id):
    """
    Get a torch.device for GPU gpu_id. If GPU not available,
    returns CPU device.
    """
    return (
        torch.device("cuda:%d" % gpu_id)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

# torch的該套件速度太慢, 因此這邊自己宣告
# 在指定維度重複輸入數據以擴展data的大小形狀
def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])

# 架構圖中把各個ResBlock取Mean
def combine_interleaved(t, inner_dims=(1,), agg_type="average"):
    if len(inner_dims) == 1 and inner_dims[0] == 1: # 就不用壓縮了
        return t
    t = t.reshape(-1, *inner_dims, *t.shape[1:])
    if agg_type == "average":
        t = torch.mean(t, dim=1)
    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t

# 不同的normalizer層, ConvEncoder會用到group 可以在不同batch_size下穩定normalize
def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer

# 計算各邊緣要添加的padding數
def calc_same_pad_conv2d(t_shape, kernel_size=3, stride=1):
    in_height, in_width = t_shape[-2:] # image的H,W
    # 計算輸出尺寸的H,W (除以stride並取天花板(ceil))
    out_height = math.ceil(in_height / stride) 
    out_width = math.ceil(in_width / stride)
    # 若輸出尺寸<輸入, 則padding值=0
    pad_along_height = max((out_height - 1) * stride + kernel_size - in_height, 0)
    pad_along_width = max((out_width - 1) * stride + kernel_size - in_width, 0)
    # top&bottom, left&right 分別是pad_along_height,pad_along_width除以2的商和餘
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return pad_left, pad_right, pad_top, pad_bottom

# 取得該block的第一層convolution layer, 呼叫"cal_same_pad_conv2d"來計算padding數後作padding
def same_pad_conv2d(t, padding_type="reflect", kernel_size=3, stride=1, layer=None):
    """ 
    Perform SAME padding on tensor, given kernel size/stride of conv operator
    assumes kernel/stride are equal in all dimensions.
    Use before conv called.
    Dilation not supported.
    :param t image tensor input (B, C, H, W)
    :param padding_type padding type constant | reflect | replicate | circular
    constant is 0-pad.
    :param kernel_size kernel size of conv
    :param stride stride of conv
    :param layer optionally, pass conv layer to automatically get kernel_size and stride
    (overrides these)
    """ 
    # 將相應的padding添加到input image "t"上, 並指定padding的類型(reflect/replicate/constant)
    if layer is not None:
        if isinstance(layer, nn.Sequential): # 確定layer是否為Sequential
            layer = next(layer.children()) # 取得第一個子層(第一個convolution layer)
        kernel_size = layer.kernel_size[0] # 取得該層的kernel_size
        stride = layer.stride[0] # 取得該層的stride值
    return F.pad( # 作Padding
        t, calc_same_pad_conv2d(t.shape, kernel_size, stride), mode=padding_type
    )

# 取得deconvolution layer, 呼叫"cal_same_pad_conv2d"來計算unpadding數後作unpadding
def same_unpad_deconv2d(t, kernel_size=3, stride=1, layer=None):
    """
    Perform SAME unpad on tensor, given kernel/stride of deconv operator.
    Use after deconv called.
    Dilation not supported.
    """
    if layer is not None:
        if isinstance(layer, nn.Sequential):
            layer = next(layer.children())
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
    h_scaled = (t.shape[-2] - 1) * stride # 縮放H&W
    w_scaled = (t.shape[-1] - 1) * stride
    pad_left, pad_right, pad_top, pad_bottom = calc_same_pad_conv2d(
        (h_scaled, w_scaled), kernel_size, stride
    )
    # =0表示不需要裁減, 直接設成-10000確保後續操作不影響輸出
    if pad_right == 0:
        pad_right = -10000
    if pad_bottom == 0:
        pad_bottom = -10000
    return t[..., pad_top:-pad_bottom, pad_left:-pad_right]

# 把image轉成pytorch張量
def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        # 將pixel 從[0,255]normalize到[0,127]
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    return transforms.Compose(ops)

# 把mask轉成pytorch張量
def get_mask_to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
    )