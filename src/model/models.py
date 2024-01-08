import torch
from .encoder import ImageEncoder
from .positional_encoding import PositionalEncoding
from .model_util import make_encoder, make_mlp
import torch.autograd.profiler as profiler
import util
import os
import warnings

class PixelNeRFNet(torch.nn.Module):
    def __init__(self, conf, stop_encoder_grad=False):
        super().__init__()
        self.encoder = make_encoder(conf["encoder"]) 
        self.use_encoder = conf.get_bool("use_encoder", True) # True
        self.use_xyz = conf.get_bool("use_xyz", False) # False
        assert self.use_encoder or self.use_xyz # 確保至少一個為True
        self.normalize_z = conf.get_bool("normalize_z", True) # True, No default
        self.stop_encoder_grad = stop_encoder_grad # False(?)不懂, No default
        self.use_PE = conf.get_bool("use_PE", False)  # True, 對location作PE
        self.use_PE_viewdirs = conf.get_bool("use_PE_viewdirs", True) # False, 對viewdirs作PE
        self.use_viewdirs = conf.get_bool("use_viewdirs", False) # True
        d_latent = self.encoder.latent_size if self.use_encoder else 0 # 512
        d_in = 3 if self.use_xyz else 1
        if self.use_viewdirs and self.use_PE_viewdirs: # add viewdirs
            d_in += 3
        if self.use_PE and d_in > 0: # Positional Encoding
            self.PE = PositionalEncoding.from_conf(conf["PE"], d_in=d_in)
            d_in = self.PE.d_out
        if self.use_viewdirs and not self.use_PE_viewdirs: # viewdir不作PE
            d_in += 3
        d_out = 4 # RGB,sigma
        self.latent_size = self.encoder.latent_size
        self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_latent, d_out=d_out)
        self.mlp_fine = make_mlp(conf["mlp_fine"], d_in, d_latent, d_out=d_out, allow_empty=True)
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False) # camera poses
        self.register_buffer("image_shape", torch.empty(2), persistent=False) # H&W
        self.d_in = d_in
        self.d_out = d_out
        self.d_latent = d_latent # image feature的dimension
        self.register_buffer("focal", torch.empty(1, 2), persistent=False) # focal
        self.register_buffer("c", torch.empty(1, 2), persistent=False) # image center

        self.num_objs = 0
        self.num_views_per_obj = 1
    
    def encode(self, images, poses, focal, z_bounds=None, c=None):
        """
        param images [SB,NV,3,H,W]
        param poses [SB,NV,4,4]
        param focal [1]
        param z_bounds "None"
        param c [1,2]
        """
        self.num_objs = images.size(0) # 1, SB
        if len(images.shape) == 5:
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(1)
            self.num_views_per_obj = images.size(1) # 1, NV
            images = images.reshape(-1, *images.shape[2:]) # [SB*NV,3,H,W]
            poses = poses.reshape(-1, 4, 4) # [SB*NV,4,4]
        else:
            self.num_views_per_obj = 1
        
        self.encoder(images) # 把輸入圖像丟進CNN encoder

        rot = poses[:, :3, :3].transpose(1, 2) # 2,3維度互換 = 三維矩陣轉置
        trans = -torch.bmm(rot, poses[:, :3, 3:]) 
          # [SB*NV,3,1], 將相機的座標從世界坐標系轉到相機坐標系
        self.poses = torch.cat((rot, trans), dim=-1) # [SB*NV,3,4]
        self.image_shape[0] = images.shape[-1] # H
        self.image_shape[1] = images.shape[-2] # W
        
        if len(focal.shape) == 0:
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1: # SRN default
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone()
        self.focal = focal.float()
        self.focal[...,1] *= -1.0 
            # 圖像座標中y方向通常向下, 但3D世界座標通常向上, 因此乘上-1
        
        if c is None:
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c
    def forward(self, xyz, coarse=True, viewdirs=None, far=False):
        """
        param xyz [SB, B, 8] (SRN [1, 128*64, 8])
        return [SB, B, 4] (SRN [1, 128*64, 4])
        """
        with profiler.record_function("model_inference"):
            SB, ray_batch_size, _ = xyz.shape
            NV = self.num_views_per_obj # 物體數量
            # Transform query points into the camera spaces of the input views
            # 讓xyz在第0個維度重複NV次
            xyz = util.repeat_interleave(xyz, NV)  # (SB*NV, B, 3)
            # 將旋轉矩陣和xyz的尾巴添加一個新維度後相乘
            # 也就是 [SB*NV,1,3,3] * [SB*NV,B,3,1] = [SB*NV,B,3,1] 再去除最後一個維度因為[...,0]
            # 得到 [SB*NV,B,3] 也就是相機坐標系下的座標 
            # 也就是把xyz座標從世界坐標系轉成相機坐標系
            xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
            xyz = xyz_rot + self.poses[:, None, :3, 3]

            if self.d_in > 0:
                if self.use_xyz:
                    if self.normalize_z: # default
                        z_feature = xyz_rot.reshape(-1,3)
                    else:
                        z_feature = xyz.reshape(-1,3)
                else: # 只取z分量 實現pixelnerf應該用不太多 我猜是Ablation
                    if self.normalize_z:
                        z_feature = -xyz_rot[..., 2].reshape(-1, 1)  # (SB*B, 1)
                    else:
                        z_feature = -xyz[..., 2].reshape(-1, 1)  # (SB*B, 1)

                if self.use_PE and not self.use_PE_viewdirs: # default, 僅對座標PE
                    z_feature = self.PE(z_feature)
                if self.use_viewdirs:
                    assert viewdirs is not None
                    viewdirs = viewdirs.reshape(SB, ray_batch_size, 3, 1)
                    viewdirs = util.repeat_interleave(viewdirs, NV)
                    viewdirs = torch.matmul(self.poses[:, None, :3, :3], viewdirs) # [SB*NV,B,3,1], viewdir也要轉成相機坐標系
                    viewdirs = viewdirs.reshape(-1, 3) # [SB*NV*B, 3]
                    z_feature = torch.cat((z_feature, viewdirs), dim=1) # [SB*NV*B, 3+36+3]
                # 這邊是把viewdir作positional encoding
                # 但pixelnerf預設是不用
                if self.use_PE and self.use_PE_viewdirs:
                    # Positional encoding (with viewdirs)
                    z_feature = self.code(z_feature)
                mlp_input = z_feature
            if self.use_encoder: # 抓取latent feature, 要先轉換成二維座標並進行focal&c的校正
                uv = -xyz[:, :, :2] / xyz[:, :, 2:] # [SB, B, 2] (1,128*64,2)
                uv *= util.repeat_interleave(self.focal.unsqueeze(1), NV if self.focal.shape[0] > 1 else 1)
                uv += util.repeat_interleave(self.c.unsqueeze(1), NV if self.c.shape[0] > 1 else 1) # 
                latent = self.encoder.index(uv, None, self.image_shape) # [SB*NV, latent_size, B], (1,512,128*64)
                if self.stop_encoder_grad: 
                    latent = latent.detach() # 停止gradient
                latent = latent.transpose(1, 2).reshape(-1, self.latent_size) # [SB*NV*B, latent_size], (128*64,512)
                if self.d_in == 0:
                    mlp_input = latent
                else: # default
                    mlp_input = torch.cat((latent, z_feature), dim=-1) # [SB*NV*B, 512+3+36+3], (128*64, 554)
                combine_index = None
                dim_size = None
                if coarse or self.mlp_fine is None: # default
                    mlp_output = self.mlp_coarse(mlp_input, combine_inner_dims=(self.num_views_per_obj, ray_batch_size),
                                                 combine_index=combine_index, dim_size=dim_size) # dim_size default=None
                else:
                    mlp_output = self.mlp_fine(mlp_input, combine_inner_dims=(self.num_views_per_obj, ray_batch_size),
                                                 combine_index=combine_index, dim_size=dim_size) # dim_size default=None
                mlp_output = mlp_output.reshape(-1, ray_batch_size, self.d_out) # [1, 128*64, 4]確保形狀正確而已 (本來就正確)
                rgb = mlp_output[..., :3]
                sigma = mlp_output[..., 3:4]

                output_list = [torch.sigmoid(rgb), torch.relu(sigma)]
                output = torch.cat(output_list, dim=-1)
                output = output.reshape(SB, ray_batch_size, -1) # [1, 128*64, 4]
                return output

    def load_weights(self, args, opt_init=False, strict=True, device=None):
        if opt_init and not args.resume:
            return
        ckpt_name = ("pixel_nerf_init" if opt_init or not args.resume else "pixel_nerf_latest")
        model_path = "%s/%s/%s" % (args.checkpoints_path, args.name, ckpt_name)
        if device is None:
            device = self.poses.device
        if os.path.exists(model_path):
            print("Load", model_path)
            self.load_state_dict(torch.load(model_path, map_location=device), strict=strict)
        elif not opt_init:
            warnings.warn((
                    "WARNING: {} does not exist, not loaded!! Model will be re-initialized.\n"
                    + "If you are trying to load a pretrained model, STOP since it's not in the right place. "
                    + "If training, unless you are startin a new experiment, please remember to pass --resume."
                ).format(model_path))
        return self

    def save_weights(self, args, opt_init=False):
        from shutil import copyfile
        ckpt_name = "pixel_nerf_init" if opt_init else "pixel_nerf_latest"
        backup_name = "pixel_nerf_init_backup" if opt_init else "pixel_nerf_backup"
        ckpt_path = os.path.join(args.checkpoints_path, args.name, ckpt_name)
        ckpt_backup_path = os.path.join(args.checkpoints_path, args.name, backup_name)
        if os.path.exists(ckpt_path):
            copyfile(ckpt_path, ckpt_backup_path)
        torch.save(self.state_dict(), ckpt_path)
        return self
