import torch
import torch.autograd.profiler as profiler
from dotmap import DotMap

# multi-GPU render
class _RenderWrapper(torch.nn.Module):
    def __init__(self, net, renderer, simple_output):
        super().__init__()
        self.net = net
        self.renderer = renderer
        self.simple_output = simple_output

    def forward(self, rays, want_weights=False):
        if rays.shape[0] == 0:
            return (torch.zeros(0, 3, device=rays.device),torch.zeros(0, device=rays.device),)
        outputs = self.renderer(self.net, rays, want_weights=want_weights and not self.simple_output)
        if self.simple_output:
            if self.renderer.using_fine:
                rgb = outputs.fine.rgb
                depth = outputs.fine.depth
            else:
                rgb = outputs.coarse.rgb
                depth = outputs.coarse.depth
            return rgb, depth
        else: # default This
            # Make DotMap to dict to support DataParallel
            return outputs.toDict()

# Volume Rendering
class NeRFRenderer(torch.nn.Module):
    """
    NeRF differentiable renderer
    :param n_coarse number of coarse (binned uniform) samples
    :param n_fine number of fine (importance) samples
    :param n_fine_depth number of expected depth samples
    :param noise_std noise to add to sigma. We do not use it
    :param depth_std noise for depth samples
    :param eval_batch_size ray batch size for evaluation
    :param white_bkgd if true, background color is white; else black
    :param lindisp if to use samples linear in disparity instead of distance
    :param sched ray sampling schedule. list containing 3 lists of equal length.
    sched[0] is list of iteration numbers,
    sched[1] is list of coarse sample numbers,
    sched[2] is list of fine sample numbers
    """

    def __init__(
        self, n_coarse=128, n_fine=0, n_fine_depth=0,
        noise_std=0.0, depth_std=0.01, eval_batch_size=100000, 
        white_bkgd=False, lindisp=False, sched=None # ray sampling schedule for coarse and fine rays
        ):  
        super().__init__()
        self.n_coarse = n_coarse # 64
        self.n_fine = n_fine # 32
        self.n_fine_depth = n_fine_depth # 16
        self.noise_std = noise_std # 0.0
        self.depth_std = depth_std # 0.01
        self.eval_batch_size = eval_batch_size # 100000
        self.white_bkgd = white_bkgd # 1.0
        self.lindisp = lindisp # False
        if lindisp:
            print("Using linear displacement rays")
        self.using_fine = n_fine > 0
        self.sched = sched
        if sched is not None and len(sched) == 0:
            self.sched = None
        self.register_buffer("iter_idx", torch.tensor(0, dtype=torch.long), persistent=True)
        self.register_buffer("last_sched", torch.tensor(0, dtype=torch.long), persistent=True)

    def sample_coarse(self, rays): # 計算Coarse network的Sample points
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays ray [ray_batch_size, 8]
        :return [ray_batch_size, 8] 
        """
        device = rays.device
        near, far = rays[:, -2:-1], rays[:, -1:]  # [ray_batch_size, 1]

        step = 1.0 / self.n_coarse # 1/64
        B = rays.shape[0] # ray_batch_size, 50000
        # 1有沒有-step好像沒差 (nerfpl就沒有)
        z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)  # [Kc], default=64
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # [ray_batch_size] 擴展B倍 
        z_steps += torch.rand_like(z_steps) * step # 添加一個隨機數增加浮動性 (optional)
        if not self.lindisp:  # SRN default, Use linear sampling in depth space
            return near * (1 - z_steps) + far * z_steps  # [ray_batch_size, Kc] (128,64)
        else:  # Use linear sampling in disparity space
            return 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

        # Use linear sampling in depth space
        return near * (1 - z_steps) + far * z_steps  # (B, Kc)

    def sample_fine(self, rays, weights): # 計算Fine network的Sample points
        """
        Weighted stratified (importance) sample
        :param rays ray [ray_batch_size, 8]
        :param weights [ray_batch_size, Kc]
        :return [ray_batch_size, Kf-Kfd]
        """
        device = rays.device
        ray_batch_size = rays.shape[0]

        weights = weights.detach() + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # [ray_batch_size, Kc], normalize
        cdf = torch.cumsum(pdf, -1)  # (B, Kc) # 累積分布函數
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # [ray_batch_size, Kc+1] 讓起始值=0
        u = torch.rand(ray_batch_size, self.n_fine - self.n_fine_depth, dtype=torch.float32, device=device) # [ray_batch_size, Kf-Kfd] 
        inds = torch.searchsorted(cdf, u, right=True).float() - 1.0  # [ray_batch_size, Kf-Kfd]
        # 類似binary search, 找到會回傳右邊那一個索引 (-1就是左邊那個)
        inds = torch.clamp_min(inds, 0.0) # 索引值不可為負數
        z_steps = (inds + torch.rand_like(inds)) / self.n_coarse 
            # [ray_batch_size, Kf-Kfd], 增加浮動性, 除以Kc是為了規範在[0,1]之間

        near, far = rays[:, -2:-1], rays[:, -1:]  # [ray_batch_size, 1]
        if not self.lindisp:  # default, Use linear sampling in depth space
            z_samp = near * (1 - z_steps) + far * z_steps  # [ray_batch_size, Kf-Kfd]
        else:  # Use linear sampling in disparity space
            z_samp = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # [ray_batch_size, Kf-Kfd]
        return z_samp

    def sample_fine_depth(self, rays, depth): # 根據每個點的深度值計算該點最有值得sample的點並sample 16個
        """
        Sample around specified depth
        :param rays ray [ray_batch_size, 8]
        :param depth [ray_batch_size]
        :return [ray_batch_size, Kfd)
        """
        z_samp = depth.unsqueeze(1).repeat((1, self.n_fine_depth)) # [ray_batch_size, Kfd]
        z_samp += torch.randn_like(z_samp) * self.depth_std # 加雜訊
        # Clamp does not support tensor bounds
        z_samp = torch.max(torch.min(z_samp, rays[:, -1:]), rays[:, -2:-1]) # 不可溢出bounds
        return z_samp

    # 計算該條rays打在pixel上的RGB 也就是作真正的Volume Rendering
    def composite(self, model, rays, z_sample, coarse=True, sb=0):
        """ Kc = coarse sample數
        param model should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
        param rays [ray_batch_size, 8]
        param z_sample [ray_batch_size, Kc]
        param coarse True
        param sb super-batch dimension; 0 = disable
        return weights [ray_batch_size, Kc]
        return rgb [ray_batch_size, 3]
        return depth [ray_batch_size]
        """
        with profiler.record_function("renderer_composite"):
            ray_batch_size, K = z_sample.shape # 128, 64
            # 計算delta
            deltas = z_sample[:, 1:] - z_sample[:, :-1]  # [ray_batch_size, Kc-1]
            delta_inf = rays[:, -1:] - z_sample[:, -1:] # delta_inf為最後一格delta, 限於far
            deltas = torch.cat([deltas, delta_inf], -1)  # [ray_batch_size, Kc]

            # 計算sample points的座標
            points = rays[:, None, :3] + z_sample.unsqueeze(2) * rays[:, None, 3:6] # [ray_batch_size, Kc, 3] x=o+td 
            points = points.reshape(-1, 3)  # (ray_batch_size*Kc, 3)
            use_viewdirs = hasattr(model, "use_viewdirs") and model.use_viewdirs # True
            val_all = []
            if sb > 0: # default, 但我設SB=1 所以不變
                points = points.reshape(sb, -1, 3)  # [SB, ray_batch_size*Kc/SB, 3]
                eval_batch_size = (self.eval_batch_size - 1) // sb + 1 # 把eval_batch_size均分給SB
                eval_batch_dim = 1
            else:
                eval_batch_size = self.eval_batch_size # 100000
                eval_batch_dim = 0

            split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)
                # SRN(1), len(split_points) = ray_batch_size * Kc / (SB * eval_batch_size))
            
            if use_viewdirs: # default
                dim1 = K # 64, Kc
                viewdirs = rays[:, None, 3:6].expand(-1, dim1, -1)  # [ray_batch_size, Kc, 3]
                if sb > 0: # SB>0, default
                    viewdirs = viewdirs.reshape(sb, -1, 3)  # [SB, ray_batch_size*Kc/SB, 3]
                else:
                    viewdirs = viewdirs.reshape(-1, 3)  # [ray_batch_size*Kc/SB, 3]
                split_viewdirs = torch.split(viewdirs, eval_batch_size, dim=eval_batch_dim)
                    # SRN(32), len(split_viewdirs) = ray_batch_size * Kc / (SB * eval_batch_size)

                # 把pts,viewdirs丟進MLP中作預測rgb,sigma
                for pnts, dirs in zip(split_points, split_viewdirs): 
                    val_all.append(model(pnts, coarse=coarse, viewdirs=dirs))
                    # val_all -> [SB, 128*Kc, 4] * 1
            else:
                for pnts in split_points:
                    val_all.append(model(pnts, coarse=coarse))
            points = None
            viewdirs = None
            # [ray_batch_size * K, 4] or (SB, ray_batch_size, 4] -> if multi-object
            out = torch.cat(val_all, dim=eval_batch_dim) # [SB, ray_batch_size, 4]
            out = out.reshape(ray_batch_size, K, -1)  # [ray_batch_size, Kc, 4] 

            rgbs = out[..., :3]  # [ray_batch_size, Kc, 3] 
            sigmas = out[..., 3]  # [ray_batch_size, Kc] 
            if self.training and self.noise_std > 0.0: # 將sigma也加上noise, default 不跑
                sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std 
            # 計算alpha
            alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))
            deltas = None
            sigmas = None
                # [ray_batch_size, Kc], # 計算alpha
                # concat完的shape=[B, K+1], cumprod的部分是在計算T_i
                # 累積的乘積 也就是(1-alpha_1), (1-alpha_1)(1-alpha_2), 
                # (1-alpha_1)(1-alpha_2)(1-alpha_3)...以此類推
                # 最後去掉最後一列 -> [B, K]
            alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  
              # [ray_batch_size, Kc+1] = [1, 1-a1, 1-a2, ..., 1-an]
            T = torch.cumprod(alphas_shifted, -1)  # [ray_batch_size, Kc+1]
            weights = alphas * T[:, :-1]  # [ray_batch_size, Kc]

            
            
            alphas = None
            alphas_shifted = None

            rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # [ray_batch_size, 3])
            depth_final = torch.sum(weights * z_sample, -1)  # [ray_batch_size]
            # 每個pixels把各點的weights乘上各點加總起來就是各pixels的weights
            if self.white_bkgd: # default
                # White background, weights總和低表示本來是透明 +1-0就會變白色
                pix_alpha = weights.sum(dim=1)  # [ray_batch_size], pixel alpha
                rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)  # [ray_batch_size, 3])
            return (weights, rgb_final, depth_final)

    def forward(self, model, rays, want_weights=False): # 將資料整理好開始呼叫composite作render
        """
        param rays [SB,ray_batch_size,8] (1,128,8)
        :model nerf model, should return (SB, B, (r, g, b, sigma))
        :param rays ray spec [origins (3), directions (3), near (1), far (1)] (SB, B, 8)
        :param want_weights "True", returns compositing weights (SB, B, K)
        :return render dict
        """
        with profiler.record_function("renderer_forward"):
            if self.sched is not None and self.last_sched.item() > 0: # 預設不會跑
                self.n_coarse = self.sched[1][self.last_sched.item() - 1] # Sample_coarse
                self.n_fine = self.sched[2][self.last_sched.item() - 1] # Sample_fine

            assert len(rays.shape) == 3
            superbatch_size = rays.shape[0] # SB(1)
            rays = rays.reshape(-1, 8)  # (SB*ray_batch_size, 8)

            z_coarse = self.sample_coarse(rays)  # [ray_batch_size, Kc] (128, 64)
            coarse_composite = self.composite(model, rays, z_coarse, coarse=True, sb=superbatch_size)
                # 計算Coarse network的Volume Rendering
            outputs = DotMap(coarse=self._format_outputs(coarse_composite, superbatch_size, want_weights=want_weights))
                # print((outputs.coarse.rgb).shape) [1,128,3]

            if self.using_fine:
                all_samps = [z_coarse]

                if self.n_fine - self.n_fine_depth > 0: # 32-16=16
                    # print(coarse_composite[0].shape) = [128,64] 各個點的weights
                    all_samps.append(self.sample_fine(rays, coarse_composite[0].detach()))  # (B, Kf - Kfd)
                if self.n_fine_depth > 0: # 16
                    # print(coarse_composite[2].shape) = [128] 各個點的depth
                    all_samps.append(self.sample_fine_depth(rays, coarse_composite[2]))  # (B, Kfd)
                z_combine = torch.cat(all_samps, dim=-1)  # (ray_batch_size, Kc + Kf), [128,96]
                z_combine_sorted, argsort = torch.sort(z_combine, dim=-1) #　依照距離排序points
                fine_composite = self.composite( model, rays, z_combine_sorted, coarse=False, sb=superbatch_size)
                    # 計算Fine network的Volume Rendering
                outputs.fine = self._format_outputs(fine_composite, superbatch_size, want_weights=want_weights)

            return outputs

    def _format_outputs( self, rendered_outputs, superbatch_size, want_weights=False): # 產生output "ret_dict" 儲存rgb, sigma, (weights)
        weights, rgb, depth = rendered_outputs
        if superbatch_size > 0: # default = 1
            rgb = rgb.reshape(superbatch_size, -1, 3) # [SB,ray_batch_size,3]
            depth = depth.reshape(superbatch_size, -1) # [SB,ray_batch_size]
            weights = weights.reshape(superbatch_size, -1, weights.shape[-1]) # [SB,ray_batch_size,K]
        ret_dict = DotMap(rgb=rgb, depth=depth)
        if want_weights: # default, 把每個point的權重存起來
            ret_dict.weights = weights
        return ret_dict

    def sched_step(self, steps=1): # 依據現在的steps來更新sample數
        """
        Called each training iteration to update sample numbers
        according to schedule
        """
        if self.sched is None: # default
            return
        self.iter_idx += steps
        while (self.last_sched.item() < len(self.sched[0])
                and self.iter_idx.item() >= self.sched[0][self.last_sched.item()]):
            self.n_coarse = self.sched[1][self.last_sched.item()]
            self.n_fine = self.sched[2][self.last_sched.item()]
            print("INFO: NeRF sampling resolution changed on schedule ==> c",self.n_coarse,"f",self.n_fine)
            self.last_sched += 1

    @classmethod
    def from_conf(cls, conf, white_bkgd=False, lindisp=False, eval_batch_size=100000):
        return cls(
            conf.get_int("n_coarse", 128),
            conf.get_int("n_fine", 0),
            n_fine_depth=conf.get_int("n_fine_depth", 0),
            noise_std=conf.get_float("noise_std", 0.0),
            depth_std=conf.get_float("depth_std", 0.01),
            white_bkgd=conf.get_float("white_bkgd", white_bkgd),
            lindisp=lindisp,
            eval_batch_size=conf.get_int("eval_batch_size", eval_batch_size),
            sched=conf.get_list("sched", None),
        )
 
    def bind_parallel(self, net, gpus=None, simple_output=False): # 多個GPU一起跑
        """
        Returns a wrapper module compatible with DataParallel.
        Specifically, it renders rays with this renderer
        but always using the given network instance.
        Specify a list of GPU ids in 'gpus' to apply DataParallel automatically.
        :param net A PixelNeRF network
        :param gpus list of GPU ids to parallize to. If length is 1,
        does not parallelize
        :param simple_output only returns rendered (rgb, depth) instead of the 
        full render output map. Saves data tranfer cost.
        :return torch module
        """
        wrapped = _RenderWrapper(net, self, simple_output=simple_output)
        if gpus is not None and len(gpus) > 1:
            print("Using multi-GPU", gpus)
            wrapped = torch.nn.DataParallel(wrapped, gpus, dim=1)
        return wrapped
