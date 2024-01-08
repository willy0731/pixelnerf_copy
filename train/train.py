import sys
import os
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
import warnings
import trainlib
from model import make_model, loss
from render import NeRFRenderer
from data import get_split_dataset
import util
import numpy as np
import torch.nn.functional as F
import torch
from dotmap import DotMap


def extra_args(parser):
    parser.add_argument("--batch_size", "-B", type=int, default=4, help="Object batch size(SB)")
    parser.add_argument("--nviews", "-V", type=str, default="1", help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')")
    parser.add_argument("--freeze_enc", action="store_true", default=None, help="Freeze encoder weights and only train MLP")
    parser.add_argument(
        "--no_bbox_step",
        type=int,
        default=100000,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument("--fixed_test", action="store_true", default=None, help="Freeze encoder weights and only train MLP")
    return parser

class PixelNeRFTrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(model, train_dataset, val_dataset, args, conf["train"], device=device)
        self.renderer_state_path = "%s/%s/_renderer" % (self.args.checkpoints_path, self.args.name) # checkpoints/exp_name

        # Setting Loss function
        self.lambda_coarse = conf.get_float("loss.lambda_coarse") # 1.0
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0) # 1.0
        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        if args.resume: # 載入logs以恢復訓練
            if os.path.exists(self.renderer_state_path):
                renderer.load_state_dict(torch.load(self.renderer_state_path, map_location=device))

        self.z_near = train_dataset.z_near
        self.z_far= train_dataset.z_far
        self.use_bbox = args.no_bbox_step > 0 # 若非0則為True
    def post_batch(self, epoch, batch): # 預設是沒用
        renderer.sched_step(args.batch_size) # Batch_size = 1

    def extra_save_state(self):
        torch.save(renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train=True, global_step=0):
        if "images" not in data:
            return {}
        all_images = data["images"].to(device=device) # [SB,NV,3,H,W]

        SB, NV, _, H, W = all_images.shape
        all_poses = data["poses"].to(device=device) #[SB,NV,4,4]
        all_bboxes = data.get("bbox") # [SB,NV,4]
        all_focals = data["focal"] # [SB]
        all_c = data.get("c") # [SB, 2], image center

        if self.use_bbox and global_step >= args.no_bbox_step:
            self.use_bbox = False
            print(">>> Stopped using bbox sampling @ iter", global_step)
        if not is_train or not self.use_bbox: # 非traning時或不用bbox則 bbox取消
            all_bboxes = None

        all_rgb_gt = []
        all_rays = []
        curr_nviews = nviews[torch.randint(0, len(nviews), ()).item()] # 要拿來訓練的Image數量
        if curr_nviews == 1: # 隨機選一張
            image_ord = torch.randint(0, NV, (SB, 1)) # image的索引
        else: # 選指定張數 (目前用不到))
            image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)
        for obj_idx in range(SB): 
            if all_bboxes is not None:
                bboxes = all_bboxes[obj_idx] # [NV,4]
            images = all_images[obj_idx] # [NV,3,H,W]
            poses = all_poses[obj_idx] # [NV,4,4]
            focal = all_focals[obj_idx] # [NV,1]
            c = None
            if "c" in data:
                c = data["c"][obj_idx] # [2]
            if curr_nviews > 1: # 如果要不只取一張照片 則取多個索引
                image_ord[obj_idx] = torch.from_numpy(np.random.choice(NV, curr_nviews, replace=False)) 
            images_0to1 = images * 0.5 + 0.5 # [N_images,3,H,W], normalize images
            cam_rays = util.gen_rays(poses, W, H, focal, self.z_near, self.z_far, c=c) # [NV,H,W,8]
            rgb_gt_all = images_0to1
            rgb_gt_all = rgb_gt_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3) # [N_images,H,W,3] -> [N_images*H*W,3]

            if all_bboxes is not None: # default
                pix = util.bbox_sample(bboxes, args.ray_batch_size)
                # pix[...,0]是圖片索引 所以共有H*W個pixels, pix[...,1]為Y座標有W個pixel
                pix_inds = pix[..., 0] * H * W + pix[..., 1] * W + pix[..., 2] # [ray_batch_size] 所有輸入圖片中的pixels索引
            else:
                pix_inds = torch.randint(0, NV * H * W, (args.ray_batch_size,))
            rgb_gt = rgb_gt_all[pix_inds] # 映射對應的pixels索引到相應的rgb_gt
            rays = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(device=device) 
                # 從rays找出pix_inds(要預測的pixels)對應的rays
            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, ray_batch_size, 3) 
        all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)

        image_ord = image_ord.to(device)
        src_images = util.batched_index_select_nd(all_images, image_ord) # [SB,NV(1),3,H,W]
        src_poses = util.batched_index_select_nd(all_poses, image_ord) # [SB,NV(1),4,4]
        all_bboxes = all_poses = all_images = None # 歸零
        model.encode(src_images, src_poses, all_focals.to(device=device), 
                     c=all_c.to(device=device) if all_c is not None else None)
        render_dict = DotMap(render_par(all_rays, want_weights=True))
        coarse = render_dict.coarse
        fine = render_dict.fine
        # print(render_dict.coarse.rgb.shape) = [1,128,3]
        using_fine = len(fine) > 0
        loss_dict = {}
        rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
        loss_dict["rc"] = rgb_loss.item() * self.lambda_coarse
        if using_fine:
            fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
            rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
            loss_dict["rf"] = fine_loss.item() * self.lambda_fine

        loss = rgb_loss
        if is_train:
            loss.backward()
        loss_dict["t"] = loss.item() # coarse_loss + fine_loss

        return loss_dict

    def train_step(self, data, global_step):
        return self.calc_losses(data, is_train=True, global_step=global_step)

    def eval_step(self, data, global_step):
        renderer.eval()
        losses = self.calc_losses(data, is_train=False, global_step=global_step)
        renderer.train()
        return losses

    def vis_step(self, data, global_step, idx=None):
        if "images" not in data:
            return {}
        if idx is None: # default This
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx
        images = data["images"][batch_idx].to(device=device)  # (NV, 3, H, W)
        poses = data["poses"][batch_idx].to(device=device)  # (NV, 4, 4)
        focal = data["focal"][batch_idx : batch_idx + 1]  # (1)
        c = data.get("c")
        if c is not None:
            c = c[batch_idx : batch_idx + 1]
        NV, _, H, W = images.shape # NV=251 cars_val有251張
        cam_rays = util.gen_rays(poses, W, H, focal, self.z_near, self.z_far, c=c) # [251,H,W,8]
        images_0to1 = images * 0.5 + 0.5
        curr_nviews = nviews[torch.randint(0, len(nviews), (1,)).item()] # 隨機生成一個index
        views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False)) # 根據該index生成不重複的隨機views列表
        view_dest = np.random.randint(0, NV - curr_nviews) # 隨機生成目標view的index 數字會>views_src
        for vs in range(curr_nviews): # 避免source和test views同一張
            view_dest += view_dest >= views_src[vs]
        views_src = torch.from_numpy(views_src)

        renderer.eval()
        source_views = images_0to1[views_src].permute(0, 2, 3, 1).cpu().numpy().reshape(-1, H, W, 3) # [1,H,W,3]
        gt = images_0to1[view_dest].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
        with torch.no_grad():
            test_rays = cam_rays[view_dest] # [H,W,8]
            test_images = images[views_src]
            model.encode(test_images.unsqueeze(0),poses[views_src].unsqueeze(0), focal.to(device=device), 
                         c=c.to(device=device) if c is not None else None)
            test_rays = test_rays.reshape(1, H * W, -1) # [1, H*W, 8]
            render_dict = DotMap(render_par(test_rays, want_weights=True))
            coarse = render_dict.coarse
            fine = render_dict.fine

            using_fine = len(fine) > 0

            alpha_coarse_np = coarse.weights[0].sum(dim=-1).cpu().numpy().reshape(H, W)
            rgb_coarse_np = coarse.rgb[0].cpu().numpy().reshape(H, W, 3)
            depth_coarse_np = coarse.depth[0].cpu().numpy().reshape(H, W)
            if using_fine:
                alpha_fine_np = fine.weights[0].sum(dim=1).cpu().numpy().reshape(H, W)
                depth_fine_np = fine.depth[0].cpu().numpy().reshape(H, W)
                rgb_fine_np = fine.rgb[0].cpu().numpy().reshape(H, W, 3)
        print("c rgb min {} max {}".format(rgb_coarse_np.min(), rgb_coarse_np.max()))
        print("c alpha min {}, max {}".format(alpha_coarse_np.min(), alpha_coarse_np.max()))
        alpha_coarse_cmap = util.cmap(alpha_coarse_np) / 255
        depth_coarse_cmap = util.cmap(depth_coarse_np) / 255
        vis_list = [*source_views, gt, depth_coarse_cmap, rgb_coarse_np, alpha_coarse_cmap]
        vis_coarse = np.hstack(vis_list) # [128,128,3] * 5 = [128,640,3]
        vis = vis_coarse

        if using_fine:
            print("f rgb min {} max {}".format(rgb_fine_np.min(), rgb_fine_np.max()))
            print("f alpha min {}, max {}".format(alpha_fine_np.min(), alpha_fine_np.max()))
            depth_fine_cmap = util.cmap(depth_fine_np) / 255
            alpha_fine_cmap = util.cmap(alpha_fine_np) / 255
            vis_list = [*source_views, gt, depth_fine_cmap, rgb_fine_np, alpha_fine_cmap]
            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))
            rgb_psnr = rgb_fine_np
        else:
            rgb_psnr = rgb_coarse_np

        psnr = util.psnr(rgb_psnr, gt)
        vals = {"psnr": psnr}
        print("psnr", psnr)

        renderer.train()
        return vis, vals
    # vis=[source views, gt, depth/rgb/alpha map](coarse&fine), vals=psnr

if __name__ == '__main__':
    args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=8)
    device = util.get_cuda(args.gpu_id[0])

    train_dataset, val_dataset, _ = get_split_dataset(args.dataset_format, args.datadir)
    
    model = make_model(conf["model"]).to(device=device)
    model.stop_encoder_grad = args.freeze_enc # default = None
    if args.freeze_enc: # 不去訓練Encoder 只訓練MLP
        print("Encoder frozen")
        model.encoder.eval()

    renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=train_dataset.lindisp).to(device=device)

    # Parallize 使得render可以拆分在不同GPU上並行操作 並指定要使用的view數量
    render_par = renderer.bind_parallel(model, args.gpu_id).eval()

    # Parallize 使得render可以拆分在不同GPU上並行操作 並指定要使用的view數量
    nviews = list(map(int, args.nviews.split())) # [1]

    trainer = PixelNeRFTrainer()
    trainer.start()
