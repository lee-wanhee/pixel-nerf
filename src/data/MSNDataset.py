import os
import math
import random

import torch
from torch.utils import data
import numpy as np
import yaml
import imageio
from PIL import Image
import cv2
import sys
sys.path.append('/data2/wanhee/pixel-nerf/src/')
from util import get_image_to_tensor_balanced, get_mask_to_tensor
import torchvision.transforms.functional as TF

def get_world_to_camera_matrix(camera_pos, vertical=None):
    # print(camera_pos)
    # We assume that the camera is pointed at the origin
    camera_z = -camera_pos / torch.norm(camera_pos, dim=-1, keepdim=True)
    if vertical is None:
        vertical = torch.tensor((0., 0., 1.))
    else:
        vertical = torch.tensor(vertical)
    vertical = vertical.to(camera_pos)
    camera_x = torch.cross(camera_z, vertical.expand_as(camera_z), dim=-1)
    camera_x = camera_x / torch.norm(camera_x, dim=-1, keepdim=True)
    camera_y = torch.cross(camera_z, camera_x, dim=-1)

    camera_matrix = torch.stack((camera_x, camera_y, camera_z), -2)
    translation = -torch.einsum('...ij,...j->...i', camera_matrix, camera_pos)
    camera_matrix = torch.cat((camera_matrix, translation.unsqueeze(-1)), -1)
    return camera_matrix


class Clevr3dDataset(data.Dataset):
    def __init__(self, path, mode, max_n=6, max_views=None, points_per_item=2048, do_frustum_culling=False,
                 shapenet=False, max_len=None, importance_cutoff=0.5):
        self.path = path
        self.mode = mode
        self.max_n = max_n
        self.points_per_item = points_per_item
        self.do_frustum_culling = do_frustum_culling
        self.shapenet = shapenet
        self.max_len = max_len
        self.importance_cutoff = importance_cutoff

        self.max_num_entities = 5 if shapenet else 11

        if shapenet:
            self.start_idx, self.end_idx = {'train': (0, 80000),
                                            'val': (80000, 80500),
                                            'test': (90000, 100000)}[mode]
        else:
            self.start_idx, self.end_idx = {'train': (0, 70000),
                                            'val': (70000, 70500),
                                            'test': (85000, 100000)}[mode]

        self.metadata = np.load(os.path.join(path, 'metadata.npz'))
        self.metadata = {k: v for k, v in self.metadata.items()}

        num_objs = (self.metadata['shape'][self.start_idx:self.end_idx] > 0).sum(1)
        num_available_views = self.metadata['camera_pos'].shape[1]
        if max_views is None:
            self.num_views = num_available_views
        else:
            assert(max_views <= num_available_views)
            self.num_views = max_views

        self.idxs = np.arange(self.start_idx, self.end_idx)[num_objs <= max_n]

        print(f'Initialized CLEVR {mode} set, {len(self.idxs)} examples')
        print(self.idxs)

        # added for pixelnerf
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.z_near = 1
        self.z_far = 15
        self.lindisp = False

    def __len__(self):
        # if self.max_len is not None:
        #     return self.max_len
        # return len(self.idxs) * self.num_views
        return len(self.idxs)

    def __getitem__(self, idx, noisy=True):
        scene_idx = idx % len(self.idxs)
        view_idx = idx // len(self.idxs)

        scene_idx = self.idxs[scene_idx]

        imgs = [np.asarray(imageio.imread(
            os.path.join(self.path, 'images', f'img_{scene_idx}_{v}.png')))
            for v in range(self.num_views)]
        # depths = [np.asarray(imageio.imread(
        #     os.path.join(self.path, 'depths', f'depths_{scene_idx}_{v}.png')))
        #     for v in range(self.num_views)]

        imgs = [img[..., :3].astype(np.float32) / 255 for img in imgs]
        imgs = [self.image_to_tensor(img) for img in imgs] # [None, ...]
        imgs_list = []
        for img in imgs:
            img = TF.center_crop(img, (240, 240))
            img = TF.resize(img, (128, 128))
            imgs_list.append(img)
        all_imgs = torch.stack(imgs_list)
        # Convert 16 bit integer depths to floating point numbers.
        # 0.025 is the normalization factor used while drawing the depthmaps.
        # depths = [d.astype(np.float32) / (65536 * 0.025) for d in depths]

        # input_img = np.transpose(imgs[view_idx], (2, 0, 1))
        #
        metadata = {k: v[scene_idx] for (k, v) in self.metadata.items()}

        # input_camera_pos = metadata['camera_pos'][view_idx]

        # all_rays = []
        all_camera_pos = metadata['camera_pos'][:self.num_views]

        all_poses = []
        all_mags = []
        for i in range(self.num_views):
            pose_tensor = get_world_to_camera_matrix(torch.tensor(all_camera_pos[i]))
            # print(all_camera_pos[i])
            # print(pose_tensor)
            real_pose = torch.zeros([4, 4])
            real_pose[:3, :] = pose_tensor
            real_pose[3, 3] = 1.
            real_pose_inverse = real_pose.inverse()

            pose = torch.zeros([4, 4])
            pose[:3, :3] = - real_pose_inverse[:3, :3]
            pose[:3, 3] = real_pose_inverse[:3, 3]
            pose[3, 3] = 1.

            # pose = torch.zeros([4, 4])
            # pose[:3, :3] = pose_tensor[:3, :3]
            # pose[:3, 3] = real_pose_inverse[:3, 3]
            # pose[3, 3] = 1.

            # print(pose)

            # pose_tensor = torch.tensor(pose_np)
            all_mags.append(real_pose_inverse[:3, 3].norm(p=2))
            all_poses.append(pose) # [None, ...]

        # fixme: hacky way to get the camera transformation right. fix it
        new_poses = [all_poses[k].clone() for k in [0, 2, 1]]
        new_poses[1][:3, 3] = new_poses[1][:3, 3] / new_poses[1][:3, 3].norm(p=2) * all_mags[1]
        new_poses[2][:3, 3] = new_poses[2][:3, 3] / new_poses[2][:3, 3].norm(p=2) * all_mags[2]
        all_poses = new_poses

        all_poses = torch.stack(all_poses)
        # for i in range(self.num_views):
        #     cur_rays = get_camera_rays(all_camera_pos[i], noisy=False)
        #     all_rays.append(cur_rays)
        # all_rays = np.stack(all_rays, 0)

        # if self.shapenet:
        #     # For the shapenet dataset, the depth images represent the z-coordinate in camera space.
        #     # Here, we convert this into Euclidian depths.
        #     new_depths = []
        #     for i in range(self.num_views):
        #         new_depth = zs_to_depths(depths[i], all_rays[i], all_camera_pos[i])
        #         new_depths.append(new_depth)
        #     depths = np.stack(new_depths, 0)

        # example = dict(metadata)
        # if self.shapenet:
        #     # We're not loading the path to the model files into PyTorch, since those are strings.
        #     del example['shape_file']

        # example['view_idxs'] = view_idx
        # example['camera_pos'] = input_camera_pos.astype(np.float32)
        # example['inputs'] = input_img
        # example['input_rays'] = all_rays[view_idx].astype(np.float32)
        # if self.mode != 'train':
        #     example['input_depths'] = depths[view_idx]
        #
        # example['input_points'] = depths_to_world_coords(depths[view_idx],
        #                                                  example['input_rays'],
        #                                                  example['camera_pos'])
        #
        # all_values = np.reshape(np.stack(imgs, 0), (self.num_views * 240 * 320, 3))
        # all_depths = np.reshape(np.stack(depths, 0), (self.num_views * 240 * 320,))
        # all_rays = np.reshape(all_rays, (self.num_views * 240 * 320, 3))
        # all_camera_pos = np.tile(np.expand_dims(all_camera_pos, 1), (1, 240 * 320, 1))
        # all_camera_pos = np.reshape(all_camera_pos, (self.num_views * 240 * 320, 3))
        #
        # num_points = all_rays.shape[0]

        # If we have fewer points than we want, sample with replacement
        # replace = num_points < self.points_per_item
        # sampled_idxs = np.random.choice(np.arange(num_points),
        #                                 size=(self.points_per_item,),
        #                                 replace=replace)
        #
        # rays = all_rays[sampled_idxs]
        # camera_pos = all_camera_pos[sampled_idxs]
        # values = all_values[sampled_idxs]
        # depths = all_depths[sampled_idxs]
        #
        # surface_points_base = depths_to_world_coords(depths, rays, camera_pos)
        #
        # empty_points, empty_points_weights, empty_t1 = importance_sample_empty_points(
        #     surface_points_base, depths, camera_pos, cutoff=self.importance_cutoff)


        # if noisy:
        #     depth_noise = 0.07 if noisy else None
        #     surface_points = depths_to_world_coords(depths, rays, camera_pos, depth_noise=depth_noise)
        # else:
        #     surface_points = surface_points_base
        #
        # if self.do_frustum_culling:
        #     # Cull those points which lie outside the input view
        #     visible = frustum_cull(surface_points, input_camera_pos, rays)
        #
        #     surface_points = surface_points[visible]
        #     empty_points = empty_points[visible]
        #     values = values[visible]
        #     depths = depths[visible]
        #     rays = rays[visible]
        #     camera_pos = camera_pos[visible]
        #
        # example['surface_points'] = surface_points
        # example['empty_points'] = empty_points
        #
        # example['empty_points_weights'] = empty_points_weights
        # example['query_camera_pos'] = camera_pos.astype(np.float32)
        # example['values'] = values
        # example['rays'] = rays
        # example['depths'] = depths
        #
        # if self.mode != 'train':
        #     mask_idx = imageio.imread(os.path.join(self.path, 'masks', f'masks_{scene_idx}_{view_idx}.png'))
        #     mask = np.zeros((240, 320, self.max_num_entities), dtype=np.uint8)
        #     np.put_along_axis(mask, np.expand_dims(mask_idx, -1), 1, axis=2)
        #     mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
        #     example['masks'] = mask

        # focal = torch.tensor([1.09375 * 128, 1.45833 * 128], dtype=torch.float32)
        focal = torch.tensor([1.45833 * 128, 1.45833 * 128], dtype=torch.float32) # before

        # print(all_imgs.shape)
        # print(all_poses.shape)
        # torch.Size([3, 3, 240, 320])
        # torch.Size([3, 4, 4])

        result = {
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
        }
        # "path": [None],
        # "img_id": [None],

        return result

if __name__=='__main__':
    dset = Clevr3dDataset(path='/ccn2/u/honglinc/datasets/multishapenet', mode='train', max_n=6, max_views=None, points_per_item=2048, do_frustum_culling=False, shapenet=True, max_len=None, importance_cutoff=0.5)
    data = dset[0]
    for k, v in data.items():
        try:
            print(f'{k}, {v.shape}')
        except:
            print(f'{k}, {v}')
