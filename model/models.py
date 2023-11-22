"""
Main model implementation
"""
import torch
from .positional_encoding import PositionalEncoding
from .model_util import make_encoder, make_mlp
import torch.autograd.profiler as profiler
from util import repeat_interleave
import os
import os.path as osp
import warnings


class PixelNeRFNet(torch.nn.Module):
    def __init__(self, conf, stop_encoder_grad=False): # Finished
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        # 看起來會導向default.conf中的encoder下 
        # 但裡面沒有type 所以make_encoder會是呼叫SpatialEncoder
        self.encoder = make_encoder(conf["encoder"]) 
        # 使用SpatialEncoder來生成Image features
        self.use_encoder = conf.get_bool("use_encoder", True)
        # 使用xyz座標
        self.use_xyz = conf.get_bool("use_xyz", False) 

        assert self.use_encoder or self.use_xyz  # 確保至少一個為True

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will
        # be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = conf.get_bool("normalize_z", True)
        #不去訓練ConvNet
        self.stop_encoder_grad = (stop_encoder_grad)
        # PE 
        self.use_code = conf.get_bool("use_code", False)  
        # Enable view directions
        self.use_viewdirs = conf.get_bool("use_viewdirs", False)
        # 預設為False, 將viewdir作PE
        self.use_code_viewdirs = conf.get_bool("use_code_viewdirs",True)

        d_latent = self.encoder.latent_size if self.use_encoder else 0
        d_in = 3 if self.use_xyz else 1

        if self.use_viewdirs and self.use_code_viewdirs: # input_dims +3
            # Apply positional encoding to viewdirs
            # 這邊應該有問題 若viewdir作PE d_in應該不只+3
            d_in += 3
        if self.use_code and d_in > 0: # 在這邊作Positional Encoding, input_dims也會變
            # Positional encoding for x,y,z OR view z
            self.code = PositionalEncoding.from_conf(conf["code"], d_in=d_in)
            d_in = self.code.d_out
        if self.use_viewdirs and not self.use_code_viewdirs: # viewdir不作PE
            # Don't apply positional encoding to viewdirs (concat after encoded)
            d_in += 3

        d_out = 4 # RGB & sigma

        self.latent_size = self.encoder.latent_size
        self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_latent, d_out=d_out)
        self.mlp_fine = make_mlp(
            conf["mlp_fine"], d_in, d_latent, d_out=d_out, allow_empty=True
        )
        # Note: this is world -> camera, and bottom row is omitted
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False) # camera poses
        self.register_buffer("image_shape", torch.empty(2), persistent=False) # H&W

        self.d_in = d_in
        self.d_out = d_out
        self.d_latent = d_latent # image feature的dimension
        self.register_buffer("focal", torch.empty(1, 2), persistent=False) # 焦距
        self.register_buffer("c", torch.empty(1, 2), persistent=False) # Principal point

        self.num_objs = 0
        self.num_views_per_obj = 1

    def encode(self, images, poses, focal, z_bounds=None, c=None): # Finished
        """
        :param images (NS, 3, H, W)
        NS is number of input (aka source or reference) views
        :param poses (NS, 4, 4)
        :param focal focal length () or (2) or (NS) or (NS, 2)  #[fx, fy]
        :param z_bounds ignored argument (used in the past)
        :param c principal point None or () or (2) or (NS) or (NS, 2) [cx, cy],
        default is center of image
        """
        self.num_objs = images.size(0) # image數
        if len(images.shape) == 5: # 若=5則有不只一張的視角
            assert len(poses.shape) == 4 # 確認poses shape正確
            assert poses.size(1) == images.size(1) # 確保 poses & image 數量(NS)一致
            self.num_views_per_obj = images.size(1) # num_views_per_obj = 每個物體的視角數
            images = images.reshape(-1, *images.shape[2:]) # [NS,3,H,W]
            poses = poses.reshape(-1, 4, 4) # [B,4,4]
        else:
            self.num_views_per_obj = 1

        self.encoder(images) # 丟進Encoder

        # B = batch_size
        # Extract旋轉&平移合併成新的self.poses
        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3) # 1,2對調
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        self.poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)

        # Extract H,W
        self.image_shape[0] = images.shape[-1] # H 
        self.image_shape[1] = images.shape[-2] # W

        # Handle various focal length/principal point formats
        if len(focal.shape) == 0: # 若為純量則將其擴展複製
            # Scalar: fx = fy = value for all views
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1: #  向量->每個view的focal不同, 擴展為 (N,2) N=views數
            # Vector f: fx = fy = f_i *for view i*
            # Length should match NS (or 1 for broadcast)
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone() # 複製確保不會動到原數據
        self.focal = focal.float()
        """
        圖像座標中y方向通常向下, 但3D世界座標通常向上
        因此乘上-1
        """
        self.focal[..., 1] *= -1.0 

        #　c為圖像中心點 類似上面focal的處理方式
        if c is None: # 以image中心點為中心點
            # Default principal point is center of image
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            # Scalar: cx = cy = value for all views
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            # Vector c: cx = cy = c_i *for view i*
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c

        if self.use_global_encoder: # 若有用ImageEncoder 這邊Encoder
            self.global_encoder(images)

    def forward(self, xyz, coarse=True, viewdirs=None, far=False): # Finished
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3) # 座標
        SB is batch of objects 
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """ # poses [NS,4,4]
        with profiler.record_function("model_inference"):
            SB, B, _ = xyz.shape
            NS = self.num_views_per_obj # 物體數量

            # Transform query points into the camera spaces of the input views
            # 讓xyz在第0個維度重複NS次
            # 
            xyz = repeat_interleave(xyz, NS)  # (SB*NS, B, 3)
            # 將旋轉矩陣和xyz的尾巴添加一個新維度後相乘
            # 也就是 [NS,1,3,3] * [SB*NS,B,3,1] = [SB*NS,B,3,1] 再去除最後一個維度因為[...,0]
            # 得到 [SB*NS,B,3] 也就是相機坐標系下的座標 
            # 也就是把xyz座標從世界坐標系轉成相機坐標系
            xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
            # [SB*NS,B,3] + [SB*NS,1,3] = [SB*NS,B,3] 這邊是加上平移向量
            xyz = xyz_rot + self.poses[:, None, :3, 3] 


            # 有個大問題 不知為捨後面z_feature shape變[SB*B]

            if self.d_in > 0:
                # * Encode the xyz coordinates
                if self.use_xyz:
                    if self.normalize_z: # 不加平移向量
                        z_feature = xyz_rot.reshape(-1, 3)  # (SB*B, 3)
                    else: # 加平移向量
                        z_feature = xyz.reshape(-1, 3)  # (SB*B, 3)
                else: # 只取z分量 實現pixelnerf應該用不太多 我猜是Ablation
                    if self.normalize_z:
                        z_feature = -xyz_rot[..., 2].reshape(-1, 1)  # (SB*B, 1)
                    else:
                        z_feature = -xyz[..., 2].reshape(-1, 1)  # (SB*B, 1)

                if self.use_code and not self.use_code_viewdirs: # 僅對座標作PE
                    # Positional encoding (no viewdirs)
                    z_feature = self.code(z_feature)

                if self.use_viewdirs: # 把viewdir跟作完PE的座標合併
                    # * Encode the view directions
                    assert viewdirs is not None
                    # Viewdirs to input view space
                    viewdirs = viewdirs.reshape(SB, B, 3, 1)
                    viewdirs = repeat_interleave(viewdirs, NS)  # (SB*NS, B, 3, 1)
                    viewdirs = torch.matmul( # viewdir也要轉成相機坐標系
                        self.poses[:, None, :3, :3], viewdirs
                    )  # (SB*NS, B, 3, 1)
                    viewdirs = viewdirs.reshape(-1, 3)  # (SB*B, 3)
                    z_feature = torch.cat(
                        (z_feature, viewdirs), dim=1
                    )  # (SB*B, 4 or 6)

                # 這邊是把viewdir作positional encoding
                # 但pixelnerf預設是不用
                if self.use_code and self.use_code_viewdirs:
                    # Positional encoding (with viewdirs)
                    z_feature = self.code(z_feature)

                # 最後處理完的z_feature丟進mlp
                mlp_input = z_feature

            if self.use_encoder:
                # Grab encoder's latent code.
                uv = -xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
                uv *= repeat_interleave(
                    self.focal.unsqueeze(1), NS if self.focal.shape[0] > 1 else 1
                )
                uv += repeat_interleave(
                    self.c.unsqueeze(1), NS if self.c.shape[0] > 1 else 1
                )  # (SB*NS, B, 2)
                latent = self.encoder.index(
                    uv, None, self.image_shape
                )  # (SB * NS, latent, B)

                if self.stop_encoder_grad:
                    latent = latent.detach()
                latent = latent.transpose(1, 2).reshape(
                    -1, self.latent_size
                )  # (SB * NS * B, latent)

                if self.d_in == 0:
                    # z_feature not needed
                    mlp_input = latent
                else:
                    mlp_input = torch.cat((latent, z_feature), dim=-1)


            combine_index = None
            dim_size = None

            # Run main NeRF network 執行Coarse/Fine network
            if coarse or self.mlp_fine is None:
                mlp_output = self.mlp_coarse(
                    mlp_input,
                    combine_inner_dims=(self.num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )
            else:
                mlp_output = self.mlp_fine(
                    mlp_input,
                    combine_inner_dims=(self.num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )

            # Interpret the output 
            mlp_output = mlp_output.reshape(-1, B, self.d_out) # [-1,B,4]

            rgb = mlp_output[..., :3]
            sigma = mlp_output[..., 3:4]

            # Sigmoid RGB, RELU sigma 這邊是整理一下rgb和sigma
            output_list = [torch.sigmoid(rgb), torch.relu(sigma)]
            output = torch.cat(output_list, dim=-1)
            output = output.reshape(SB, B, -1) # [SB,B,4]
        return output

    def load_weights(self, args, opt_init=False, strict=True, device=None): # Finished
        """
        Helper for loading weights according to argparse arguments.
        Your can put a checkpoint at checkpoints/<exp>/pixel_nerf_init to use as initialization.
        :param opt_init if true, loads from init checkpoint instead of usual even when resuming
        """
        # TODO: make backups
        if opt_init and not args.resume:
            return
        ckpt_name = (
            "pixel_nerf_init" if opt_init or not args.resume else "pixel_nerf_latest"
        )
        model_path = "%s/%s/%s" % (args.checkpoints_path, args.name, ckpt_name)

        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print("Load", model_path)
            self.load_state_dict( # 將 weights 載到load_state_dict中
                torch.load(model_path, map_location=device), strict=strict
            )
        elif not opt_init:
            warnings.warn(
                (
                    "WARNING: {} does not exist, not loaded!! Model will be re-initialized.\n"
                    + "If you are trying to load a pretrained model, STOP since it's not in the right place. "
                    + "If training, unless you are startin a new experiment, please remember to pass --resume."
                ).format(model_path)
            )
        return self

    def save_weights(self, args, opt_init=False): # Finished
        """
        Helper for saving weights according to argparse arguments
        :param opt_init if true, saves from init checkpoint instead of usual
        """
        from shutil import copyfile

        ckpt_name = "pixel_nerf_init" if opt_init else "pixel_nerf_latest"
        backup_name = "pixel_nerf_init_backup" if opt_init else "pixel_nerf_backup"

        ckpt_path = osp.join(args.checkpoints_path, args.name, ckpt_name)
        ckpt_backup_path = osp.join(args.checkpoints_path, args.name, backup_name)

        if osp.exists(ckpt_path):
            copyfile(ckpt_path, ckpt_backup_path)
        torch.save(self.state_dict(), ckpt_path)
        return self
