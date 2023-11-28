import cv2
import numpy as np
import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import functools
import math
import warnings

def batched_index_select_nd(all_images, image_ord): # 根據image_ord找出對應的輸入圖片
    """
    param all_images [SB,NV,3,H,W]
    param image_ord [SB,1]
    return [SB,1]
    """
    X = image_ord[(...,) + (None,) * (len(all_images.shape) - 2)] # [SB,1,1,1,1]將image_ord擴展成跟all_images一樣的dimension (5維)
    X = X.expand(-1,-1, *all_images.shape[2:]) # 擴展成[NV,1,3,H,W]
    return all_images.gather(1, X) # [NV,1,3,H,W], 根據索引找出對應的圖片

def bbox_sample(bboxes, N_pixels): # 在bounding bboxes中sample N_pixels個pixels並找到對應的輸入圖片索引
    # N_pixels default=5000
    # 一次訓練N_pixels (args.ray_batch_size為50000), 為每個pixel對應不同張輸入圖片
    image_id = torch.randint(0, bboxes.shape[0], (N_pixels,)) # [N_pixels]
    pix_bboxes = bboxes[image_id] # [N_pixels,4] 取出相應輸入圖片的bbox
    x = (torch.rand(N_pixels) * (pix_bboxes[:,2] + 1 - pix_bboxes[:,0]) + pix_bboxes[:,0]).long()
    y = (torch.rand(N_pixels) * (pix_bboxes[:,3] + 1 - pix_bboxes[:,1]) + pix_bboxes[:,1]).long()
    pix = torch.stack((image_id, y, x), dim=-1)
    return pix # +1是為了不要超出邊界, x,y就是在bounding bboxes中 Sample一些pixels的座標(2D) 並結合他們對應的索引

def unproj_map(width, height, focal, c=None, device="cpu"): # 將圖片中的每個pixel映射到相機坐標系中
    if c is None: # 取圖像中心點
        c = [width*0.5, height*0.5]
    else: # 壓縮多餘的維度, EX: [1,1,2] => [2]
        c = c.squeeze()
    if isinstance(focal, float):
        focal = [focal,focal]
    elif len(focal.shape) == 0: # SRN default
        focal = focal[None].expand(2) # 也是把[f]擴展成[f,f]
    elif len(focal.shape) == 1:
        focal = focal.expand(2)
    Y, X = torch.meshgrid(torch.arange(height, dtype=torch.float32) - float(c[1]),
                          torch.arange(width, dtype=torch.float32) - float(c[0]),
                          indexing='xy') # 建立座標網格
    X = X.to(device=device) / float(focal[0]) # [H,W,3]
    Y = Y.to(device=device) / float(focal[1]) # 將pixel映射到相機坐標系
    Z = torch.ones_like(X)
    unproj = torch.stack((X,-Y,-Z), dim=-1) # [H,W,3]
    # 相機坐標系中,通常X軸向前, Y軸向左, Z軸向上
    # 因此Y軸要翻轉, 因為Y軸翻轉Z軸才要跟著乘上負號 (右手定則)
    unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1) # 分母為[H,W,1], normalize unproj
    return unproj

def gen_rays(poses, width, height, focal, z_near, z_far, c=None, ndc=False):
    num_images = poses.shape[0] # NV
    
    device = poses.device
    # 將圖片中的pixel映射到相機坐標系中
    cam_unproj_map = (unproj_map(width, height, focal.squeeze(), c=c, device=device) # [H,W,3]
                      .unsqueeze(0) # [1,H,W,3]
                      .repeat(num_images, 1, 1, 1)) # [N_images,H,W,3]
    
    rays_o = poses[:, None, None, :3, 3].expand(-1,height,width,-1) # 取得各張圖的相機位置 複製Pixel個
    # 把 "相機的右,上,前向量" 乘上 "把各pixel映射到相機坐標系上" = "各pixel的方向向量"
    rays_d = torch.matmul(poses[:, None, None, :3, :3], cam_unproj_map.unsqueeze(-1))[:,:,:,:,0]
    # print(poses[:, None, None, :3, :3].shape) [N_images,1,1,3,3]
    # print(cam_unproj_map.unsqueeze(-1).shape) [N_images,H,W,3,1]
    # print(torch.matmul(poses[:, None, None, :3, :3], cam_unproj_map.unsqueeze(-1)).shape) [N_images,H,W,3,1]
    # print(rays_d.shape) [N_images,H,W,3]
    if ndc: # SRN預設是沒有
        if not (z_near == 0 and z_far == 1):
            warnings.warn("dataset z near and z_far not compatible with NDC, setting them to 0, 1 NOW")
    
        z_near, z_far = 0.0, 1.0
        rays_o, rays_d = ndc_rays(width, height, focal, 1.0, rays_o, rays_d)
    
    cam_nears = torch.tensor(z_near, device=device).view(1,1,1,1).expand(num_images, height, width, -1)
    cam_fars = torch.tensor(z_far, device=device).view(1,1,1,1).expand(num_images, height, width, -1)
    return torch.cat((rays_o, rays_d, cam_nears, cam_fars), dim=-1)

def ndc_rays(H, W, focal, near, rays_o, rays_d): # 將rays_o移動到近平面
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d

def get_cuda(gpu_id): # 使用GPU加速運算
    """
    Get a torch.device for GPU gpu_id. If GPU not available,
    returns CPU device.
    """
    return (
        torch.device("cuda:%d" % gpu_id)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    
def repeat_interleave(input, repeats, dim=0): # 在指定維度重複輸入數據以擴展data的大小形狀
    """ torch的該套件速度太慢, 因此這邊自己宣告
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])

########################################

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
