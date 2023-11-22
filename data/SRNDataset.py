import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from util import get_image_to_tensor_balanced, get_mask_to_tensor


class SRNDataset(torch.utils.data.Dataset):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """

    def __init__(
        self, path, stage="train", image_size=(128, 128), world_scale=1.0,
    ):
        """
        :param stage train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        super().__init__()
        self.base_path = "data/" + path + "_" + stage # data/srn_cars_train|val|test
        self.dataset_name = os.path.basename(path) # srn_cars
        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        
        self.stage = stage # train|val|test

        assert os.path.exists(self.base_path) # 確定路徑存在

        is_chair = "chair" in self.dataset_name # 若訓練dataset為chair則=True
        if is_chair and stage == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp # 路徑在chairs_2.0_train中

        self.intrins = sorted( # 把intrins.txt按照順序排列
            # "*" 代表不管中間資料夾名稱
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = image_size # default=(128, 128)
        self.world_scale = world_scale # default=1.0
        self._coord_trans = torch.diag( # 把y,z軸加上負號
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        if is_chair: # chair 的near&far
            self.z_near = 1.25
            self.z_far = 2.75
        else: # 非chair的near&far
            self.z_near = 0.8
            self.z_far = 1.8
        self.lindisp = False

    def __len__(self): # 可以知道總共訓練了幾個資料夾(類別)
        return len(self.intrins)

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))

        assert len(rgb_paths) == len(pose_paths)

        with open(intrin_path, "r") as intrinfile:
            lines = intrinfile.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            height, width = map(int, lines[-1].split())

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        count=0
        for rgb_path, pose_path in zip(rgb_paths, pose_paths):
            if count == 3:
                break
            count += 1
            img = imageio.imread(rgb_path)[..., :3] # [3,256,256]
            img_tensor = self.image_to_tensor(img) # [3,128,128]
            
            # mask中的值只有0 or 255 決定該Pixel為前景or背景
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255 #[128,128,1]
            mask_tensor = self.mask_to_tensor(mask) # [1,128,128]

            pose = torch.from_numpy(
                np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            )
            pose = pose @ self._coord_trans # 將y,z乘-1

            # 計算遮罩在水平&垂直非0的邊界 定義物體在image中的位置
            rows = np.any(mask, axis=1) 
            cols = np.any(mask, axis=0)
            # mask在水平&垂直中非0的索引
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0: # 表示image全部都是0 根本沒有物體在image中
                raise RuntimeError(
                    "ERROR: Bad image at", rgb_path, "please investigate!"
                )
            # 取物體的上下界
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            # 前景物體的外接矩形框
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

        if all_imgs.shape[-2:] != self.image_size: # H,W 尺寸不合則進行縮放
            scale = self.image_size[0] / all_imgs.shape[-2] # 分母分子都是H
            focal *= scale
            cx *= scale
            cy *= scale
            all_bboxes *= scale

            # 根據image_size進行縮放
            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        if self.world_scale != 1.0: # 通常預設為1 依特殊情形縮放
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        focal = torch.tensor(focal, dtype=torch.float32)

        result = {
            "path": dir_path,
            "img_id": index,
            "focal": focal,
            "c": torch.tensor([cx, cy], dtype=torch.float32),
            "images": all_imgs,
            "masks": all_masks,
            "bbox": all_bboxes,
            "poses": all_poses,
        }
        return result
