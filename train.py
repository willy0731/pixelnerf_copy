import os,sys
import util
from data import get_split_dataset
from model import make_model, loss
from render import NeRFRenderer
import numpy as np
import torch.nn.functional as F 
import torch
from dotmap import DotMap
import os.path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import tqdm
import imageio

class PixelNeRFTrainer:
    def __init__(self):
        self.conf = conf["train"]
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = val_dataset
        self.render_state_path = "%s/%s/_renderer" % (self.args.checkpoints_path, self.args.name)
        
        self.train_dataset_loader = DataLoader(
            train_dataset,
            batch_size = args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        self.val_dataset_loader = DataLoader(
            val_dataset,
            batch_size=min(args.batch_size, 16),
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        self.num_total_batches = len(self.train_dataset) # 3619728 for llff
        self.exp_ame = args.name
        self.save_interval = self.conf.get_int("save_interval")
        self.print_interval = self.conf.get_int("print_interval")
        self.vis_interval = self.conf.get_int("vis_interval")
        self.eval_interval = self.conf.get_int("eval_interval")
        self.num_epoch_repeats = conf.get_int("num_epoch_repeats", 1)
        self.num_epochs = args.epochs
        self.accu_grad = self.conf.get_int("accu_grad", 1)
        self.summary_path = os.path.join(args.logs_path, args.name)
        self.writer = SummaryWriter(self.summary_path)
        self.fixed_test = hasattr(args, "fixed_test") and args.fixed_test
        os.makedirs(self.summary_path, exist_ok=True) # logs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.gamma != 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optmizer = self.optimizer, gamma = args.gamma
            )
        else:
            self.lr_scheduler = None
        # Load weights
        # self.managed_weight_saving = hasattr(model, "load_weights")
        # if self.managed_weight_saving:
        #     model.load_weights(self.args)
        # 還沒打完
        self.start_iter_id = 0
        self.z_near = self.train_dataset.z_near
        self.z_far= self.train_dataset.z_far
        self.use_bbox = args.no_bbox_step > 0 # 若非0則為True
        
    def post_batch(self, epoch, batch):
        """
        Ran after each batch
        """
        pass

    def extra_save_state(self):
        """
        Ran at each save step for saving extra state
        """
        pass

    def calc_losses(self, data, is_train=True, global_step=0):
        if "images" not in data:
            return {}
        all_images = data["images"].to(device=device) # [SB,NV,3,H,W]
        SB, NV, _, H, W = all_images.shape
        all_poses = data["poses"].to(device=device) #[SB,NV,4,4]
        all_bboxes = data.get("bbox") # [SB,NV,4]
        all_focal = data["focal"] # [SB]
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
            focal = all_focal[obj_idx] # [NV,1]
            c = None
            if "c" in data:
                c = data["c"][obj_idx] # [2]
            if curr_nviews > 1: # 如果要不只取一張照片 則取多個索引
                image_ord[obj_idx] = torch.from_numpy(np.random.choice(NV, curr_nviews, replace=False)) 
            rays = util.gen_rays(poses, W, H, focal, self.z_near, self.z_far, c=c) # [NV,H,W,8]
            rgb_gt_all = images * 0.5 + 0.5 # [N_images,3,H,W], normalize images
            rgb_gt_all = rgb_gt_all.permute(0,2,3,1).contiguous().reshape(-1,3) # [N_images,H,W,3] -> [N_images*H*W,3]
            
            if all_bboxes is not None:
                pix = util.bbox_sample(bboxes, args.ray_batch_size)
                # pix[...,0]是圖片索引 所以共有H*W個pixels, pix[...,1]為Y座標有W個pixel
                pix_inds = pix[...,0]*H*W + pix[...,1]*W + pix[...,2] # [ray_batch_size] 所有輸入圖片中的pixels索引
            else:
                pix_inds = torch.randint(0, NV*H*W, (args.ray_batch_size,))
            
            rgb_gt = rgb_gt_all[pix_inds] # 映射對應的pixels索引到相應的rgb_gt
            rays = rays.view(-1, rays.shape[-1])[pix_inds].to(device=device) 
                # 從rays找出pix_inds(要預測的pixels)對應的rays
            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)
        
        all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, ray_batch_size, 3) 
        all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)

        image_ord = image_ord.to(device)
        src_images = util.batched_index_select_nd(all_images, image_ord) # [SB,NV(1),3,H,W]
        src_poses = util.batched_index_select_nd(all_poses, image_ord) # [SB,NV(1),4,4]
        all_bboxes = all_poses = all_images = None # 歸零
        model.encode(src_images, src_poses, all_focal.to(device=device), 
                     c=all_c.to(device=device) if all_c is not None else None)


    def train_step(self, data, global_step):
        return self.calc_losses(data, is_train=True, global_step=global_step)

    def eval_step(self, data, global_step):
        renderer.eval()
        losses = self.calc_losses(data, is_train=False, global_step=global_step)
        renderer.train()
        return losses

    def vis_step(self, data, global_step):
        """
        Visualization step
        """
        return None, None
    
    def start(self):
        def fmt_loss_str(losses):
            return "loss " + (" ".join(k + ":" + str(losses[k]) for k in losses))
        
        def data_loop(dl):
            while True:
                for x in iter(dl):
                    yield x
        
        test_data_iter = data_loop(self.val_dataset_loader)
        step_id = self.start_iter_id
        progress = tqdm.tqdm(bar_format="[{bar}] {n}/{total} [{elapsed}<{remaining}]")
        for epoch in range(self.num_epoch_repeats):
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], global_step=step_id)
            
            batch = 0
            for _ in range(self.num_epoch_repeats):
                for data in self.train_dataset_loader:
                    losses = self.train_step(data, global_step=step_id)
                    loss_str = fmt_loss_str(losses)
                    if batch % self.print_interval == 0:
                        print("E",epoch, "B",batch,loss_str, "lr",self.optimizer.param_groups[0]["lr"])
                    
                    if batch % self.eval_interval == 0:
                        test_data = next(test_data_iter)
                        self.model.eval()
                        with torch.no_grad():
                            test_losses = self.eval_step(test_data, global_step=step_id)
                        self.model.train()
                        test_loss_str = fmt_loss_str(test_losses)
                        self.writer.add_scalars("train", losses, global_step=step_id)
                        self.writer.add_scalars("test", test_losses, global_step=step_id)
                        print("*** Eval:","E",epoch, "B",batch,test_loss_str, "lr",self.optim.param_groups[0]["lr"])
                    
                    if batch % self.save_interval == 0 and (epoch > 0 or batch > 0):
                        print("Saving...")
                        # 還沒寫完
                    
                    if batch % self.vis_interval == 0:
                        print("Generating Visualization...")
                        if self.fixed_test:
                            test_data = next(iter(sel.test_data_loader))
                        else:
                            test_data = next(test_data_iter)
                        self.model.eval()
                        with torch.no_grad():
                            vis, vis_vals = self.vis_step(test_data, global_step=step_id)
                        if vis_vals is not None:
                            self.writer.add_scalars("vis", vis_vals, global_step=step_id)
                        self.model.train()
                        if vis is not None:
                            vis_u8 = (vis * 255).astype(np.uint8)
                            imageio.imwrite(os.path.join(self.visual_path, "{:04}_{:04}_vis.png".format(epoch, batch)), vis_u8)
                        
                    if(batch == self.num_total_batches-1 or batch % self.accu_grad == self.accu_grad-1):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    self.post_batch(epoch, batch)                    
                    step_id += 1
                    batch += 1
                    progress.update(1)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

if __name__ == '__main__':
    args, conf = util.args.parse_args()
    device = util.get_cuda(args.gpu_id[0])

    train_dataset, val_dataset, _ = get_split_dataset(args.dataset_format, args.datadir)

    model = make_model(conf["model"]).to(device=device)
    model.encoder.eval()
    renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=False,).to(device=device)
    nviews = list(map(int, args.nviews.split())) # [1]
    
    trainer = PixelNeRFTrainer()
    trainer.start()       