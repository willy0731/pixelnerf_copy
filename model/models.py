import os
import torch
from .encoder import ImageEncoder
from .positional_encoding import PositionalEncoding
from .model_util import make_encoder, make_mlp
import torch.autograd.profiler as profiler
import util
import warnings

class PixelNeRFNet(torch.nn.Module):
    def __init__(self, conf, stop_encoder_grad=False):
        super().__init__()
        self.encoder = make_encoder(conf["encoder"]) 
        self.use_encoder = conf.get_bool("use_encoder", True) # True
        self.use_xyz = conf.get_bool("use_xyz", False) # False
        assert self.use_encoder or self.use_xyz # 確保只少一為True
        self.normalize_z = conf.get_bool("normalize_z", True) # True, No default
        self.stop_encoder_grad = stop_encoder_grad # False(?)不懂, No default
        self.use_PE = conf.get_bool("use_PE", False)  # True, 對location作PE
        self.use_PE_viewdirs = conf.get_bool("use_PE_viewdirs") # False, 對viewdirs作PE
        self.use_viewdirs = conf.get_bool("use_viewdirs", False) # True
        
        d_latent = self.encoder.latent_size if self.use_encoder else 0 # 512
        d_in = 3 if self.use_xyz else 1
        if self.use_viewdirs and self.use_PE_viewdirs: # add viewdirs
            d_in += 3
        if self.use_PE and d_in > 0: # Positional Encoding
            self.PE = PositionalEncoding.from_conf(conf["PE"], d_in=d_in)
        if self.use_viewdirs and not self.use_PE_viewdirs: # add viewdirs
            d_in += 3
        d_out = 4 # RGB,sigma
         
        self.latent_size = self.encoder.latent_size
        self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_latent, d_out=d_out)
        self.mlp_fine = make_mlp(conf["mlp_fine"], d_in, d_latent, d_out=d_out, allow_empty=True)
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False) # camera poses
        self.register_buffer("image_shape", torch.empty(2), persistent=False) # H&W
        self.register_buffer("focal", torch.empty(1, 2), persistent=False) # focal
        self.register_buffer("c", torch.empty(1, 2), persistent=False) # image center
        self.d_in = d_in
        self.d_out = d_out
        self.d_latent = d_latent
        self.num_objects = 0
        self.num_views_per_object = 1
    
    def encode(self, images, poses, focal, z_bounds=None, c=None):
        """
        param images [SB,NV,3,H,W]
        param poses [SB,NV,4,4]
        param focal [1]
        param z_bounds "None"
        param c [1,2]
        """
        self.number_objects = images.size(0) # 1, SB
        if len(images.shape) == 5:
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(1)
            self.number_views_per_object = images.size(1) # 1 NV
            images = images.reshape(-1, *images.shape[2:]) # [SB*NV,3,H,W]
            poses = poses.reshape(-1,4,4) # [SB*NV,4,4]
        else:
            self.number_views_per_object = 1
        
        self.encoder(images) # 把輸入圖像丟進CNN encoder
        rot = poses[:,:3,:3].transpose(1,2) # 2,3維度互換 = 三維矩陣轉置
        trans = -torch.bmm(rot, poses[:,:3,3:]) 
            # [SB*NV,3,1], 將相機的座標從世界坐標系轉到相機坐標系
        self.poses = torch.cat((rot, trans), dim=-1) # [SB*NV,3,4]

        self.image_shape[0] = images.shape[-1] # H
        self.image_shape[1] = images.shape[-2] # W
        
        if len(focal.shape) == 0:
            focal = focal[None, None].repeat((1,2))
        elif len(focal.shape) == 1: # SRN default
            focal = focal.unsqueeze(-1).repeat((1,2))
        else:
            focal = focal.clone()
        self.focal = focal.float()
        self.focal[...,1] *= -1.0 
            # 圖像座標中y方向通常向下, 但3D世界座標通常向上, 因此乘上-1
        
        if c is None:
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            c = c[None,None].repeat((1,2))
        elif len(c.shape) == 1:
            c = c.unsqueeze(-1).repeat((1,2))
        self.c = c
    
    def forward(self, xyz, coarse=True, viewdirs=None, far=False):
        """
        param xyz [SB, ray_batch_size, 8] (SRN [1, 50000, 8])
        return [SB, ray_batch_size, 4] (SRN [1, 50000, 4])
        """
        with profiler.record_function("model_inference"):
            SB, ray_batch_size, _ = xyz.shape
            NS = self.num_views_per_object
            # Transform query points into the camera spaces of the input views
            # 讓xyz在第0個維度重複NS次
            xyz = util.repeat_interleave(xyz, NS)  # (SB*NV, B, 3)
            # 將旋轉矩陣和xyz的尾巴添加一個新維度後相乘
            # 也就是 [SB*NV,1,3,3] * [SB*NS,B,3,1] = [SB*NV,B,3,1] 再去除最後一個維度因為[...,0]
            # 得到 [SB*NV,B,3] 也就是相機坐標系下的座標 
            # 也就是把xyz座標從世界坐標系轉成相機坐標系
            xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
            xyz = xyz_rot + self.poses[:, None, :3, 3]

            if self.d_in > 0:
                if self.use_xyz:
                    if self.normalize_z:
                        z_feaure = xyz_rot.reshape(-1,3)
            從這邊開始寫
                    