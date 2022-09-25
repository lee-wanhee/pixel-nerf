"""
Mesh reconstruction tools
"""
import mcubes
import torch
import numpy as np
import util
import tqdm
import warnings
import torch.nn.functional as F

from util import repeat_interleave

import os

def marching_cubes(
    occu_net,
    c1=[-1, -1, -1],
    c2=[1, 1, 1],
    reso=[128, 128, 128],
    isosurface=50.0,
    sigma_idx=3,
    eval_batch_size=100000,
    coarse=True,
    device=None, masks=None, src_pose=None, obj_idx=0,
):
    """
    Run marching cubes on network. Uses PyMCubes.
    WARNING: does not make much sense with viewdirs in current form, since
    sigma depends on viewdirs.
    :param occu_net main NeRF type network
    :param c1 corner 1 of marching cube bounds x,y,z
    :param c2 corner 2 of marching cube bounds x,y,z (all > c1)
    :param reso resolutions of marching cubes x,y,z
    :param isosurface sigma-isosurface of marching cubes
    :param sigma_idx index of 'sigma' value in last dimension of occu_net's output
    :param eval_batch_size batch size for evaluation
    :param coarse whether to use coarse NeRF for evaluation
    :param device optionally, device to put points for evaluation.
    By default uses device of occu_net's first parameter.
    """
    # if occu_net.use_viewdirs:
    #     warnings.warn(
    #         "Running marching cubes with fake view dirs (pointing to origin), output may be invalid"
    #     )
    with torch.no_grad():
        # breakpoint()
        if device is None:
            device = next(occu_net.parameters()).device
        # grid = util.gen_grid(*zip(c1, c2, reso), ij_indexing=True)
        grid = util.gen_grid(*zip(c1, c2, reso), ij_indexing=False)
        grid_n_one = torch.concat([grid, torch.ones([grid.shape[0], 1])], dim=-1)[..., None].to(device).clone()
        is_train = occu_net.training

        # print("Evaluating sigma @", grid.size(0), "points")
        occu_net.eval()

        temp_idx = 0
        all_sigmas = []
        grid_spl = torch.split(grid, eval_batch_size, dim=0)
        if occu_net.use_viewdirs:
            fake_viewdirs = -grid / torch.norm(grid, dim=-1).unsqueeze(-1)
            vd_spl = torch.split(fake_viewdirs, eval_batch_size, dim=0)
            for pnts, vd in tqdm.tqdm(zip(grid_spl, vd_spl), total=len(grid_spl)):
                # print('pnts.shape', pnts.shape)
                # breakpoint()
                outputs = occu_net(
                    pnts[None, ...].to(device=device), coarse=coarse, viewdirs=vd[None, ...].to(device=device)
                )
                sigmas = outputs[..., sigma_idx]
                # if temp_idx < 5:
                #     sigmas *= 0.
                all_sigmas.append(sigmas.cpu())
                temp_idx += 1
        else:
            for pnts in tqdm.tqdm(grid_spl):
                outputs = occu_net(pnts.to(device=device), coarse=coarse)
                sigmas = outputs[..., sigma_idx]
                all_sigmas.append(sigmas.cpu())
        sigmas = torch.cat(all_sigmas, dim=0)
        # breakpoint()
        sigmas = sigmas.view(*reso)
        # x_index = torch.logical_and(grid[..., 0] < 2.5, grid[..., 0] > -0.5)
        # y_index = torch.logical_and(grid[..., 1] < 2.5, grid[..., 1] > -0.5)
        dist_xy = torch.sqrt(grid[..., 0]**2 + grid[..., 1]**2)
        pick_some_xy_index = torch.logical_and(dist_xy > -1.5, dist_xy < 1.5)
        # sigmas.view(128, 128, 128)[:, :, :] = 0.0
        sigmas *= pick_some_xy_index.view(*reso)
        sigmas = sigmas.cpu().numpy()

        if masks != None:
            masks = masks.to(device)

            # masks = torch.zeros_like(masks)
            # temp = masks.view(128, 128)[:, :]
            # temp = torch.ones_like(temp)
            # masks.view(128, 128)[:, :] = temp

            # masks = torch.zeros_like(masks)
            # temp = masks.view(128, 128)[:, 64:]
            # temp = torch.ones_like(temp)
            # masks.view(128, 128)[:, 64:] = temp

            # breakpoint()
            import matplotlib.pyplot as plt
            n = 1
            fig, axs = plt.subplots(1, n, figsize=(8, 3))
            for _ in range(n):
                axs.imshow(masks[_].cpu().permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
                # masks[_].flatten(0, 1)[index] = 0.5
                # axs[0, _].imshow(rgb_gt_all[_])
                # axs[1, _].imshow(all_rgb[_])

            isExist = os.path.exists('debug_mask')
            if not isExist:
                os.makedirs('debug_mask')
            plt.savefig(f'debug_mask/mask.png')
            # breakpoint()

            # breakpoint()
            xyz = torch.matmul(src_pose.inverse(), grid_n_one)[None, ...][..., 0]
            print(src_pose.inverse())

            focal = torch.tensor([[122.9440, -122.9440]], device='cuda:0')
            # focal = torch.tensor([[122.9440, -45.]], device='cuda:0')
            focal = torch.tensor([[122.9440, -64.]], device='cuda:0')
            # focal = torch.tensor([[64., -64]], device='cuda:0')
            c = torch.tensor([[64., 64.]], device='cuda:0')


            # (Pdb)
            # self.latent_scaling / image_size
            # tensor([0.0159, 0.0159], device='cuda:0')
            # (Pdb)
            # image_size
            # tensor([128., 128.], device='cuda:0')
            # (Pdb)
            # self.latent_scaling
            # tensor([2.0317, 2.0317], device='cuda:0')

            uv = -xyz[:, :, :2] / (xyz[:, :, 2:] + 1e-5)  # (SB, B, 2)
            # breakpoint()
            uv *= repeat_interleave(
                focal.unsqueeze(1), 1
            )
            uv += repeat_interleave(
                c.unsqueeze(1), 1
            )  # (SB*NS, B, 2)
            # latent = self.encoder.index(
            #     uv, None, self.image_shape
            # )  # (SB * NS, latent, B)
            uv = uv / 128 * 2 - 1
            # uv = uv * 0.0159

            # uv[..., 0] *= -2
            uv[..., 1] *= -1


            # masks ~ torch.Size([1, 512, 64, 64])
            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            # uv = uv.flip(dims=(3,))
            samples = F.grid_sample(
                masks,
                uv,
                align_corners=False,
                mode='bilinear',
                padding_mode='zeros',
            )

            # focal_x = 0.9605 * 128
            # focal_y = 0.9605 * 128
            # bias_x = (128 - 1.) / 2. # TODO: check this (- 1.)
            # bias_y = (128 - 1.) / 2.
            # intrinsic_mat = torch.tensor([[focal_x, 0, bias_x, 0],
            #                               [0, focal_y, bias_y, 0],
            #                               [0, 0, 1, 0],
            #                               [0, 0, 0, 1]])
            #
            # cam2spixel = intrinsic_mat.cuda()
            # spixel2cam = instrinsic_mat.inverse().cuda()
            # breakpoint()

            # pixel_coord = torch.matmul(cam2spixel, cam_coord)
            # pixel_coord = pixel_coord[..., 0]
            # uv = pixel_coord[..., :2] / (pixel_coord[..., 2:3] + 1e-5)
            #
            # uv = (uv + 0.) / 128 * 2 - 1

            # cam_coord = cam_coord[..., 0]
            # cam_coord = cam_coord[..., :2] / (cam_coord[..., 2:3] + 1e-5) #torch.Size([2097152, 2])
            # breakpoint()
            # cam_coord *= torch.Tensor([[1, -1]]).to(device)
            # cam_coord = cam_coord.view([128, 128, 128, 2])
            # cam_coord_ = torch.zeros_like(cam_coord)
            # cam_coord_[..., 0] = torch.flip(cam_coord[..., 0], dims=(0,))
            # cam_coord_[..., 0] = cam_coord[..., 1]
            # cam_coord_[..., 1] = cam_coord[..., 0]
            # cam_coord_[..., 1] = cam_coord[..., 1]
            # breakpoint()
            # cam_coord_ = cam_coord_.flatten(0, 2)
            # cam_coord = cam_coord_.clone()
            # inside_mask = F.grid_sample(masks.to(device), uv[None, ...].to(device), mode='nearest', padding_mode='zeros', align_corners=None)

            inside_mask = samples[:, :, :, 0].permute(0, 2, 1)  # (B, C, N)

            inside_mask = inside_mask.view([128, 128, 128])# TODO: check this permutation
            # sigmas *= inside_mask.cpu().numpy()
            sigmas += inside_mask.cpu().numpy() * 100

        print("Running marching cubes")
        vertices, triangles = mcubes.marching_cubes(sigmas, isosurface)
        # Scale
        c1, c2 = np.array(c1), np.array(c2)
        vertices *= (c2 - c1) / np.array(reso)

    if is_train:
        occu_net.train()
    return vertices + c1, triangles


def save_obj(vertices, triangles, path, vert_rgb=None):
    """
    Save OBJ file, optionally with vertex colors.
    This version is faster than PyMCubes and supports color.
    Taken from PIFu.
    :param vertices (N, 3)
    :param triangles (N, 3)
    :param vert_rgb (N, 3) rgb
    """
    file = open(path, "w")
    if vert_rgb is None:
        # No color
        for v in vertices:
            file.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
    else:
        # Color
        for idx, v in enumerate(vertices):
            c = vert_rgb[idx]
            file.write(
                "v %.4f %.4f %.4f %.4f %.4f %.4f\n"
                % (v[0], v[1], v[2], c[0], c[1], c[2])
            )
    for f in triangles:
        f_plus = f + 1
        file.write("f %d %d %d\n" % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()
    print('mesh saved', path)
