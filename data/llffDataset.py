import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np

class llffDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root_dir, 
                 split='train', 
                 img_wh=(504,378), 
                 spheric_poses=False,
                 val_num=1
                 ):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.val_num = max(1, val_num) # at least 1
        self.define_transform()
        self.read_meta()
        self.white_back = False

    def read_meta(self):
        poses_path = os.path.join(self.root_dir, 'poses_bounds.npy')
        poses_bounds = np.load(poses_path) # (images數, 17)
        # 把所有images的路徑儲存並排列
        self.image_path = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))

        if self.split in ['train', 'val']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1,3,5) # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:] # (N_images, 2) -> near&far

        # 