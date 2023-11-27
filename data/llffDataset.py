import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from torchvision import transforms as T
from PIL import Image
from kornia import create_meshgrid

def get_ray_dirs(H,W,focal): # 計算相機到各個Pixel的觀察方向
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    directions = torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # [H,W,3]
    return directions

def get_rays(directions, c2w): # 生成每張圖相機的原點和之於每個pixel的viewdirs
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in 相機坐標系
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d
    
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

def normalize(v): # 正規化
    return v/np.linalg.norm(v)

def average_poses(poses): # 計算各個相機的平均poses矩陣
    center = poses[..., 3].mean(0) # (3) pose[...,3]是camera的座標
    z = normalize(poses[...,2].mean(0)) # (3) 對z分量normalize
    y_ = poses[..., 1].mean(0) # (3)
    x = normalize(np.cross(y_, z)) # (3)
    y = np.cross(z, x) # (3)
    
    poses_avg = np.stack([x,y,z,center], 1) # (3,4)
    return poses_avg

def center_poses(poses): # 把poses矩陣從世界坐標系轉換成以平均camera poses為中心的新坐標系
    poses_avg = average_poses(poses)
    poses_avg_homo = np.eye(4) # [4,4] # 就是c2w, camera to world 轉換矩陣
    poses_avg_homo[:3] = poses_avg
    last_row = np.tile(np.array([0,0,0,1]), (len(poses),1,1)) # [N_images, 1, 4]
    poses_homo = np.concatenate([poses, last_row], 1) # [N_images,4,4]
    # 把poses矩陣從世界坐標系轉換成以平均camera poses為中心的新坐標系
    poses_centered = np.linalg.inv(poses_avg_homo) @ poses_homo # [N_images,4,4]
    poses_centered = poses_centered[:, :3] # [N_images,3,4]
    
    return poses_centered, np.linalg.inv(poses_avg_homo) # 轉換後的poses和w2c

def create_spiral_poses(radii, focus_depth, n_poses=120): # 螺旋化poses
    """
    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, depth that spiral poseslook at
        n_poses: int, number of poses to create along the path
        
    Outputs:
        poses_spiral: (n_poses, 3 , 4) the poses in the spiral path
    """
    
    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii
        z = normalize(center - np.array([0, 0, -focus_depth]))
        y_ = np.array([0,1,0]) # (3)
        x = normalize(np.cross(y_,z)) # (3)
        y = np.cross(z,x) # (3)
        poses_spiral += [np.stack([x,y,z,center], 1)] # [3,4]
    
    return np.stack(poses_spiral, 0) # [n_poses,3,4]
        
def create_spheric_poses(radius, n_poses=120):
    """
    Inputs:
        radius: the (negative) height and the radius of the circle
        
    Output:
        spheric_poses: (n_poses,3,4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([ # 平移
            [1,0,0,0],
            [0,1,0,-0.9*t],
            [0,0,1,t],
            [0,0,0,1]
        ])
        rot_phi = lambda phi : np.array([ # 在x軸旋轉phi
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1]
        ])
        rot_theta = lambda th : np.array([ # 在y軸旋轉th
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1]
        ])
        
        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w[:3]
    
    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/5, radius)] # 36 degree view downwards
    
    return np.stack(spheric_poses,0)
    

class llffDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 path, 
                 stage='train', 
                 image_size=(504,378), 
                 spheric_poses=False,
                 val_num=1
                 ):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.root_dir = path
        self.split = stage
        self.img_wh = image_size
        self.spheric_poses = spheric_poses
        self.val_num = max(1, val_num) # at least 1
        self.define_transforms()
        self.read_meta()
        self.white_back = False

    def read_meta(self):
        poses_path = os.path.join(self.root_dir, 'poses_bounds.npy')
        poses_bounds = np.load(poses_path) # (images數, 17)
        # 把所有images的路徑儲存並排列
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))

        if self.split in ['train', 'val']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1,3,5) # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:] # (N_images, 2) -> near&far

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1] 
        assert H*self.img_wh[0] == W*self.img_wh[1], \
            f'You must set @img_wh to have the same aspect ratio as ({W}, {H}) !'
        
        self.focal *= self.img_wh[0]/W # 這邊調整focal
        
        # Step 2: correct poses, "下右後" -> "右上後"
        # 第一列乘上負號和第二列對調
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
                # (N_images, 3, 4) exclude H, W, focal
        
        self.poses, self.pose_avg = center_poses(poses)
        
        distance_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(distance_from_center) # 找最接近center image的當val_image
        
        # Step 3: correct scale -> nearest depth is little more than 1.0
        near_original = self.bounds.min()
        scale_factor = near_original*0.75 # 0.75 is the default parameter
                                          # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor # 各camera座標也要跟著縮放
        self.directions = get_ray_dirs(self.img_wh[1], self.img_wh[0], self.focal)
            # [H,W,3]
        if self.split == 'train':
            self.all_rays = []
            self.all_rgbs = []
            for i , image_path in enumerate(self.image_paths):
                if i == val_idx:
                    continue
                c2w = torch.FloatTensor(self.poses[i])
                
                img = Image.open(image_path)
                assert img.size[1]*self.img_wh[0] == img.size[0]*self.img_wh[1], \
                    f'''{image_path} has different aspect ratio than img_wh, 
                        please check your data!'''
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # [3,H,W]
                img = img.view(3, -1).permute(1, 0) # [H,W, 3],各pixel的RGB
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both [H*W, 3]
                if not self.spheric_poses:
                    near, far = 0, 1
                    rays_o, rays_d = ndc_rays(self.img_wh[1], self.img_wh[0],
                                                  self.focal, 1.0, rays_o, rays_d)
                                     # near plane is always at 1.0
                                     # near and far in NDC are always 0 and 1
                                     # See https://github.com/bmild/nerf/issues/34
                else:
                    near = self.bounds.min()
                    far = min(8 * near, self.bounds.max()) # focus on central object only

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             near*torch.ones_like(rays_o[:, :1]),
                                             far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)
                                 
            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
            
        elif self.split == 'val':
            print('val image is', self.image_paths[val_idx])
            self.c2w_val = self.poses[val_idx]
            self.image_path_val = self.image_paths[val_idx]
            
        else : # for testing
            if self.split.endswith('train'): # test on training set
                self.poses_test = self.poses
            elif not self.spheric_poses:
                focus_depth = 3.5
                radii = np.percentile(np.abs(self.poses[...,3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)
            else:
                radius = 1.1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius)
    def define_transforms(self):
        self.transform = T.ToTensor()
     
    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return self.val_num
        return len(self.poses_test)
    
    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}
        
        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.c2w_val)
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])
                
            rays_o, rays_d = get_rays(self.directions, c2w)
            
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
            else:
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1) # [H*W,8], 含ray_o, rays_d, near, far
            
            sample = {'rays': rays,
                      'c2w': c2w}
            
            if self.split == 'val': # 如果是val則要有GT的rgb
                img = Image.open(self.image_path_val).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3)
                sample['rgbs'] = img
                
        return sample