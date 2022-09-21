import os

import torchvision.transforms.functional as TF

from .base_dataset import BaseDataset
from PIL import Image
import torch
import glob
import numpy as np
import random
import h5py
# from model.morf.evaluate_mesh import load_gt_mesh_from_hdf
import pdb

# added for pixelnerf
import warnings
warnings.filterwarnings("ignore")

class MultiscenesDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.set_defaults(input_nc=3, output_nc=3)
        # parser.add_argument('--n_scenes', type=int, default=1000, help='dataset length is #scenes')
        # parser.add_argument('--n_img_each_scene', type=int, default=10, help='for each scene, how many images to load in a batch')
        # parser.add_argument('--no_shuffle', action='store_true')
        # parser.add_argument('--mask_size', type=int, default=128)
        # parser.add_argument('--frame5', action='store_true')
        # parser.add_argument('--skip', type=int, default=0)
        # parser.add_argument('--dataset_nearest_interp', action='store_true')
        # parser.add_argument('--dataset_combine_masks', action='store_true')
        # parser.add_argument('--color_jitter', action='store_true')
        # parser.add_argument('--use_eisen_seg', action='store_true')
        # parser.add_argument('--small_dataset', action='store_true')
        return parser

    def __init__(self, opt, **kwargs):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        # added for pixelnerf
        self.z_near = 1
        self.z_far = 15
        self.lindisp = False
        opt.dataroot = opt.datadir
        opt.num_slots = 4
        opt.val_n_scenes = 600

        print('TDW dataset initialized')

        BaseDataset.__init__(self, opt)

        self.n_scenes = opt.n_scenes if opt.isTrain else opt.val_n_scenes
        self.n_img_each_scene = opt.n_img_each_scene
        self.use_eisen_seg = opt.use_eisen_seg
        self.min_num_masks = self.opt.num_slots - 1 if not self.opt.dataset_combine_masks else 4
        self.dataroot = self.opt.dataroot
        self.skip = self.opt.skip
        self.frame5 = self.opt.frame5

        if self.opt.color_jitter:
            self.color_transform = transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)

        # todo: debugging room_chair_train only, set the flag to False for TDW dataset
        self.disable_load_mask = False

    def _transform(self, img):
        if self.opt.dataset_nearest_interp:
            img = TF.resize(img, (self.opt.load_size, self.opt.load_size), Image.NEAREST)
        else:
            img = TF.resize(img, (self.opt.load_size, self.opt.load_size))

        if self.opt.color_jitter:

            if self.reset_color_jitter:
                self.reset_color_jitter = False
                self.prev_brightness = float(torch.empty(1).uniform_(0.6, 1.4))
                self.prev_contrast = float(torch.empty(1).uniform_(0.6, 1.4))
                self.prev_saturation = float(torch.empty(1).uniform_(0.6, 1.4))
                self.prev_hue = float(torch.empty(1).uniform_(-0.4, 0.4))

            img = TF.adjust_brightness(img, self.prev_brightness)
            img = TF.adjust_contrast(img, self.prev_contrast)
            img = TF.adjust_saturation(img, self.prev_saturation)
            img = TF.adjust_hue(img, self.prev_hue)

        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.5] * img.shape[0], [0.5] * img.shape[0])  # [0,1] -> [-1,1]
        return img

    def _transform_mask(self, img):
        img = TF.resize(img, (self.opt.mask_size, self.opt.mask_size), Image.NEAREST)
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.5] * img.shape[0], [0.5] * img.shape[0])  # [0,1] -> [-1,1]
        return img

    @staticmethod
    def _object_id_hash(objects, val=256, dtype=torch.long):
        ''' Hash RGB values into unique integer indices '''
        objects = torch.tensor(np.array(objects)).permute(2, 0, 1) # [3, H, W]

        C = objects.shape[0]
        objects = objects.to(dtype)
        out = torch.zeros_like(objects[0:1, ...])
        for c in range(C):
            scale = val ** (C - 1 - c)
            out += scale * objects[c:c + 1, ...]

        _, out = torch.unique(out, return_inverse=True)
        out -= out.min()

        out = Image.fromarray(out.squeeze(0).numpy().astype(np.uint8))
        return out

    def _get_filenames(self, idx):

        # print("self.opt.small_dataset", self.opt.small_dataset)
        if not self.opt.small_dataset:
            prefix = 'sc{:06d}'.format(idx)
        else:
            prefix = 'sc{:04d}'.format(idx)
        if self.opt.frame5:
            prefix += '_frame5' if not 'bridge' in self.dataroot else '_frame9'
        if 'room_chair' in self.dataroot:
            filenames = [os.path.join(self.dataroot, '{:05d}_'.format(idx*4+i)+prefix + '_az0%d.png' % i) for i in range(self.n_img_each_scene)]
        else:
            filenames = [os.path.join(self.dataroot, prefix+'_img%d.png' % i) for i in range(self.n_img_each_scene)]
        return filenames

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing, here it is scene_idx
        """
        scene_idx = index + self.skip
        scene_filenames = self._get_filenames(scene_idx)

        if self.opt.isTrain and not self.opt.no_shuffle:
            filenames = random.sample(scene_filenames, self.n_img_each_scene)
        else:
            filenames = scene_filenames[:self.n_img_each_scene]
        rets = []
        self.reset_color_jitter = True

        # hdf_path = scene_filenames[0].split('_frame5')[0]+'.hdf5'
        # with h5py.File(hdf_path, "r") as f:
        #     positions = f['frames']['0005']['objects']['positions'][:]


        for path in filenames:
            img = Image.open(path).convert('RGB')
            img_data = self._transform(img)
            pose_path = path.replace('.png', '_RT.txt')
            try:
                pose = np.loadtxt(pose_path)
            except FileNotFoundError:
                print('filenotfound error: {}'.format(pose_path))
                assert False
            pose = torch.tensor(pose, dtype=torch.float32)

            # pose = pose.inverse()
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )

            pose = self._coord_trans_world @ torch.tensor(pose, dtype=torch.float32) @ self._coord_trans_cam


            if self.opt.fixed_locality:
                azi_rot = np.eye(3)  # not used; placeholder
            else:
                azi_path = pose_path.replace('_RT.txt', '_azi_rot.txt')
                azi_rot = np.loadtxt(azi_path)
            azi_rot = torch.tensor(azi_rot, dtype=torch.float32)
            depth_path = path.replace('.png', '_depth.npy')
            if os.path.isfile(depth_path):
                depth = np.load(depth_path)  # HxWx1
                depth = torch.from_numpy(depth)  # HxWx1
                depth = depth.permute([2, 0, 1])  # 1xHxW
                ret = {'img_data': img_data, 'path': path, 'cam2world': pose, 'azi_rot': azi_rot, 'depth': depth}
            else:
                ret = {'img_data': img_data, 'path': path, 'cam2world': pose, 'azi_rot': azi_rot}
            mask_path = path.replace('.png', '_pred_mask.png' if self.use_eisen_seg else '_mask.png')
            if os.path.isfile(mask_path) and not self.disable_load_mask:
                mask = Image.open(mask_path).convert('RGB')
                # mask_l = mask.convert('L')
                mask_l = self._object_id_hash(mask)
                mask = self._transform_mask(mask)
                ret['mask'] = mask
                mask_l = self._transform_mask(mask_l)
                mask_flat = mask_l.flatten(start_dim=0)  # HW,
                greyscale_dict = mask_flat.unique(sorted=True)  # 8,
                # print(greyscale_dict)
                onehot_labels = mask_flat[:, None] == greyscale_dict  # HWx8, one-hot
                onehot_labels = onehot_labels.type(torch.uint8)
                mask_idx = onehot_labels.argmax(dim=1)  # HW
                bg_color = greyscale_dict[0] if 'tdw' in mask_path or 'bridge' in mask_path else greyscale_dict[2]
                fg_idx = mask_flat != bg_color  # HW
                ret['mask_idx'] = mask_idx
                ret['fg_idx'] = fg_idx
                obj_idxs = []
                for i in range(len(greyscale_dict)):
                    obj_idx = mask_l == greyscale_dict[i]  # 1xHxW
                    obj_idxs.append(obj_idx)
                obj_idxs = torch.stack(obj_idxs)  # Kx1xHxW
                ret['obj_idxs'] = obj_idxs  # KxHxW

                # additional attributes: GT background mask and object masks
                ret['bg_mask'] = mask_l == bg_color
                obj_masks = []
                if self.use_eisen_seg:

                    area = (mask_l == greyscale_dict[:, None, None]).sum(dim=[1, 2])
                    if self.min_num_masks < len(greyscale_dict):
                        _, idx = area.topk(k=self.min_num_masks)
                    else:
                        idx = range(len(greyscale_dict))

                    for i in range(len(greyscale_dict)):
                        if greyscale_dict[i] == bg_color:
                            continue
                        if i in idx:
                            obj_mask = mask_l == greyscale_dict[i]  # 1xHxW
                            obj_masks.append(obj_mask)
                else:
                    for i in range(len(greyscale_dict)):
                        if greyscale_dict[i] == bg_color:
                            continue
                        obj_mask = mask_l == greyscale_dict[i]  # 1xHxW
                        obj_masks.append(obj_mask)


                # Exception handling for empty object masks
                if len(obj_masks) == 0:
                    print('Error reading file: ', path)
                    return self.buffer_rets

                obj_masks = torch.stack(obj_masks)  # Kx1xHxW

                shape = obj_masks.shape[0]
                # if the number of masks is too small, pad with empty masks
                if obj_masks.shape[0] < self.min_num_masks:
                    n, d, h, w = obj_masks.shape
                    obj_masks = torch.cat([obj_masks, torch.zeros(self.min_num_masks-n, d, h, w)], dim=0)

                # Exception handling for empty object masks
                if not 'bridge' in self.dataroot and (shape == 1 or obj_masks.shape[0] > self.min_num_masks):
                    print('Error reading file: ', path)
                    print("here", self.min_num_masks, obj_masks.shape[0], shape)

                    return self.buffer_rets

                ret['obj_masks'] = obj_masks  # KxHxW


                # Get GT mesh
                # - First get 2D segmentation color map of GT masks
                gt_mask_path = path.replace('.png', '_mask.png')
                gt_mask = Image.open(gt_mask_path).convert('RGB')
                seg_color = torch.tensor(np.array(gt_mask)).permute(2, 0, 1)  # [3, H, W]
                seg_color = TF.resize(seg_color, (self.opt.load_size, self.opt.load_size), Image.NEAREST)

                obj_seg_colors = []
                for obj_mask in obj_masks:
                    color, count = (obj_mask * seg_color).flatten(1, 2).unique(dim=-1, return_counts=True)
                    count[color.sum(0) == 0] = 0
                    argmax = count.argmax(-1)
                    color = color[:, argmax]
                    obj_seg_colors.append(torch.tensor(color).view(1, 3))

                obj_seg_colors = torch.stack(obj_seg_colors, dim=0)

                if obj_seg_colors.shape[0] < self.min_num_masks-1:
                    print(self.min_num_masks, 'self.min_num_masks')
                    obj_seg_colors = torch.cat([obj_seg_colors, torch.zeros_like(obj_seg_colors[0:(self.min_num_masks-1-obj_masks.shape[0])])], dim=0)
                    print('Error reading segmentation color: ', path)
                ret['obj_seg_colors'] = obj_seg_colors

            else:
                try:
                    ret['mask'] = torch.zeros_like(rets[-1]['mask'])
                    ret['mask_idx'] = torch.zeros_like(rets[-1]['mask_idx'])
                    ret['fg_idx'] = torch.zeros_like(rets[-1]['fg_idx'])

                    ret['obj_idxs'] = torch.zeros_like(rets[-1]['obj_idxs'])  # KxHxW

                    # additional attributes: GT background mask and object masks
                    ret['bg_mask'] = torch.zeros_like(rets[-1]['bg_mask'])

                    ret['obj_masks'] = torch.zeros_like(rets[-1]['obj_masks']) # KxHxW
                    ret['obj_seg_colors'] = torch.zeros_like(rets[-1]['obj_seg_colors'])
                except:
                    # return self.buffer_rets
                    pass


            # ret['positions'] = torch.tensor(positions)[:, [0, 2, 1]] # switch x and y axis

            rets.append(ret)

            # print(self.opt.dataroot)

            if 'tdw' in self.opt.dataroot:
                ret['dataset'] = 'tdw'
            elif 'bridge' in self.opt.dataroot:
                ret['dataset'] = 'bridge'
            else:
                raise NotImplementedError

            if self.opt.fg_mask:
                ret['use_fg_mask'] = True
                masks = (~ret['bg_mask'] * 1.)
                images = ret['img_data']
                # print('masks.shape', masks.shape)
                # print('images.shape', images.shape)

                img = (images * masks).cpu().detach().numpy()
                rows = np.any(img, axis=1)
                cols = np.any(img, axis=0)
                rnz = np.where(rows)[0]
                cnz = np.where(cols)[0]

                if len(rnz) == 0:
                    cmin = rmin = 0
                    cmax = mask.shape[-1]
                    rmax = mask.shape[-2]
                else:
                    rmin, rmax = rnz[[0, -1]]
                    cmin, cmax = cnz[[0, -1]]
                bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
                # print('\nbbox', bbox)
                ret['bbox'] = bbox

            else:
                ret['use_fg_mask'] = False


        self.buffer_rets = rets
        return rets

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.n_scenes


def collate_fn(batch):
    # "batch" is a list (len=batch_size) of list (len=n_img_each_scene) of dict
    # flat_batch = [item for sublist in batch for item in sublist]
    # img_data = torch.stack([x['img_data'] for x in flat_batch])
    # paths = [x['path'] for x in flat_batch]
    # cam2world = torch.stack([x['cam2world'] for x in flat_batch])
    # azi_rot = torch.stack([x['azi_rot'] for x in flat_batch])
    # if 'depth' in flat_batch[0]:
    #     depths = torch.stack([x['depth'] for x in flat_batch])  # Bx1xHxW
    # else:
    #     depths = None

    flat_batch = [item for sublist in batch for item in sublist]
    img_data = torch.stack([x['img_data'] for x in flat_batch])
    paths = [x['path'] for x in flat_batch]
    cam2world = torch.stack([x['cam2world'] for x in flat_batch])
    azi_rot = torch.stack([x['azi_rot'] for x in flat_batch])
    if 'depth' in flat_batch[0]:
        depths = torch.stack([x['depth'] for x in flat_batch])  # Bx1xHxW
    else:
        depths = None

    ret = {
        'img_data': img_data,
        'paths': paths,
        'cam2world': cam2world,
        'azi_rot': azi_rot,
        'depths': depths
    }
    if 'mask' in flat_batch[0]:
        masks = torch.stack([x['mask'] for x in flat_batch])
        ret['masks'] = masks
        mask_idx = torch.stack([x['mask_idx'] for x in flat_batch])
        ret['mask_idx'] = mask_idx
        fg_idx = torch.stack([x['fg_idx'] for x in flat_batch])
        ret['fg_idx'] = fg_idx
        obj_idxs = flat_batch[0]['obj_idxs']  # Kx1xHxW
        ret['obj_idxs'] = obj_idxs
        bg_mask = torch.stack([x['bg_mask'] for x in flat_batch])
        ret['bg_mask'] = bg_mask
        obj_masks = torch.stack([x['obj_masks'].squeeze(1) for x in flat_batch])
        ret['obj_masks'] = obj_masks
        ret['obj_seg_colors'] = torch.stack([x['obj_seg_colors'] for x in flat_batch])

    # added for pixelnerf
    # print("ret['img_data'].shape", ret['img_data'].shape)
    SBNV, _, H, W = ret['img_data'].shape
    SB = len(batch)

    # print("flat_batch[0]['dataset']", flat_batch[0]['dataset'])
    if flat_batch[0]['dataset'] == 'tdw':
        focal = torch.Tensor([[0.9605 * 128, 0.9605 * 128] for _ in range(len(ret['paths']))]).view(SB, int(SBNV / SB), 2)[:, 0]
    elif flat_batch[0]['dataset'] == 'bridge':
        focal = torch.Tensor([[0.7703 * 128, 0.7703 * 128] for _ in range(len(ret['paths']))]).view(SB, int(SBNV / SB), 2)[:, 0]
    else:
        raise NotImplementedError

    # dtype float32 # 0.9605 tdw / 0.7703 bridge
    if flat_batch[0]['use_fg_mask']:
        ret['bboxes'] = torch.stack([x['bbox'] for x in flat_batch])
        masks = (~ret['bg_mask'] * 1.).view(SB, int(SBNV / SB), 1, H, W)
        images = ret['img_data'].view(SB, int(SBNV / SB), 3, H, W)
        images  = images * masks + (1. - masks) * torch.ones_like(images)
        masks = (~ret['bg_mask'] * 1.).view(SB, int(SBNV / SB), 1, H, W)
        bboxes = ret['bboxes'].view(SB, int(SBNV / SB), 4)
    else:
        images = ret['img_data'].view(SB, int(SBNV / SB), 3, H, W)
        masks = None
        bboxes = None

    data = {
        'path': ret['paths'],
        'img_id': None,
        'focal': focal,
        'images': images,
        'poses': ret['cam2world'].view(SB, int(SBNV/SB), 4, 4),
        'masks': masks,
        'bboxes': bboxes
    }
    # masks = (~ret['bg_mask'] * 1.).view(SB, int(SBNV/SB), 1, H, W)
    # print('data["images"].shape', data["images"].shape)
    # print('masks.shape', masks.shape)
    # print('masks.max()', masks.max())
    # print('masks.min()', masks.min())

    # ret['paths'].view(SB, SBNV/SB)[:, 0]

    return data

