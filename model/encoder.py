"""
Implements image encoders
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import util
from model.custom_encoder import ConvEncoder
import torch.autograd.profiler as profiler
from torchvision.models import ResNet34_Weights

# PixelNeRF 主要是用這個
class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        # 應該是Ablation 畢竟pixelnerf是用Resnet作Encoder
        self.use_custom_resnet = backbone == "custom" # 若不是用resnet則用客製化的custom
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = util.get_norm_layer(norm_type)

        if self.use_custom_resnet:
            print("WARNING: Custom encoder is experimental only")
            print("Using simple convolutional encoder")
            self.model = ConvEncoder(3, norm_layer=norm_layer)
            self.latent_size = self.model.dims[-1]
        else: # 使用Resnet
            print("Using torchvision", backbone, "encoder")
            # self.model = getattr(torchvision.models, backbone)(
            #     pretrained=pretrained, norm_layer=norm_layer
            # )
            self.model = getattr(torchvision.models, backbone)(
                weights=ResNet34_Weights.IMAGENET1K_V1, norm_layer=norm_layer
            )
            # Following 2 lines need to be uncommented for older configs
            self.model.fc = nn.Sequential() # 去掉最後的fully connected層
            self.model.avgpool = nn.Sequential() # 去掉最後的pooling層
            # num_layers=4, 因此latent_size=256
            self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers # 4 
        self.index_interp = index_interp # "bilinear", Interpolation to use for indexing
        self.index_padding = index_padding # "border", adding mode to use for indexing
        self.upsample_interp = upsample_interp # "bilinear", Interpolation to use for upscaling latent code
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )
        # self.latent (B, L, H, W)

    # 抓要丟進MLP訓練用的feature latent 用bilinear interpolation
    def index(self, uv, cam_z=None, image_size=(), z_bounds=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y) 圖片的pixel二維座標
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """
        with profiler.record_function("encoder_index"):
            # 我理解成padding, 處理batch size不匹配的情況, 將uv張量複製多次
            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1: # 正方形的image
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0
            # (B, N, 1, 2) # B=Batch_size, N=image編號, 2為pixel的平面座標
            uv = uv.unsqueeze(2)  
            samples = F.grid_sample( # 從self.latent中提取和pixel對齊的feature latent
                self.latent, # 輸入的feature map [B,C,H,W], C=channels
                uv, # 圖像平面表格 也就是網格grid [B,H,W,2], 2=xy座標
                align_corners=True, # 是否對其四個角,否的話則對齊圖像中心
                mode=self.index_interp, # bilinear
                padding_mode=self.index_padding, # border
            )
            # 只保留每個pixel點的一個值 形狀從[B,C,H,W] => [B,C,N]
            return samples[:, :, :, 0]

    # 
    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        if self.feature_scale != 1.0: # image需要縮放
            x = F.interpolate( # 執行縮放
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)

        if self.use_custom_resnet: # 直接丟進ConvEncoder
            self.latent = self.model(x)
        else: # Resnet
            x = self.model.conv1(x)
            x = self.model.bn1(x) # batch normalize layer
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1: # default=4
                if self.use_first_pool: # default=True
                    x = self.model.maxpool(x)
                x = self.model.layer1(x) # 3次residual (resnet中的basic block)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x) # 4次
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x) # 6次
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x) # 3次
                latents.append(x)

            self.latents = latents
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:] # [H,W] 以latents[0]為基準
            for i in range(len(latents)): # 使每個latents具有相同維度, 
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )
            self.latent = torch.cat(latents, dim=1) # 將所有latent合在一起
        self.latent_scaling[0] = self.latent.shape[-1] # W
        self.latent_scaling[1] = self.latent.shape[-2] # H
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0 # 縮放
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            num_layers=conf.get_int("num_layers", 4),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            feature_scale=conf.get_float("feature_scale", 1.0),
            use_first_pool=conf.get_bool("use_first_pool", True),
        )

# 不想看 八成只是拿來作Ablation的
class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            latent_size=conf.get_int("latent_size", 128),
        )
