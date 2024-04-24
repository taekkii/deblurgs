#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera, MiniCam
import numpy as np
import math
import imageio
import matplotlib
import torchvision.transforms.functional
from utils.graphics_utils import getWorld2View, getProjectionMatrix
from scene.cameras import get_c2w as cam_to_numpy_c2w
from utils.mvg_utils import mean_camera_pose, get_c2w_from_eye

cmapper = matplotlib.cm.get_cmap('jet_r')

def depth_colorize_with_mask(depthlist, background=(0,0,0), dmindmax=None):
    """ depth: (H,W) - [0 ~ 1] / mask: (H,W) - [0 or 1]  -> colorized depth (H,W,3) [0 ~ 1] """
    print("Depth colorizing...", end="")
    batch, vx, vy = np.where(depthlist!=0)
    if dmindmax is None:
        valid_depth = depthlist[batch, vx, vy]
        dmin, dmax = valid_depth.min(), valid_depth.max()
    else:
        dmin, dmax = dmindmax
    norm_dth = np.ones_like(depthlist)*dmax # [B, H, W]
    norm_dth[batch, vx, vy] = (depthlist[batch, vx, vy]-dmin)/(dmax-dmin)
    
    final_depth = np.ones(depthlist.shape + (3,)) * np.array(background).reshape(1,1,1,3) # [B, H, W, 3]
    final_depth[batch, vx, vy] = cmapper(norm_dth)[batch,vx,vy,:3]
    print(" [DONE]")

    return final_depth

@torch.no_grad()
def depth_colorize(depths:torch.Tensor, z_near:float=0.01, z_far:float=100, clip_percentage:float=1.000):
    """
    ARGUMENTS
    ---------
    depths: tensor [b,h,w]
    z_near, z_far: decides "absolute scale"
    
    RETURNS
    -------
    final_depth: uint8 ndarray [b,h,w,3]
    """
    z_near = max(z_near, depths.min().item())
    z_far = min(z_far, depths.max().item() , depths.reshape((-1,)).sort().values[int((depths.numel()-1)*clip_percentage)].item())
    depths = ( depths - z_near ) / (z_far - z_near)
    depths = depths.clip(0.0, 1.0)

    depths_npy = depths.cpu().numpy()
    final_depth = cmapper(depths_npy)

    return (final_depth * 255).astype(np.uint8)

def depths_to_ndc_z(depths:torch.Tensor, z_near:float, z_far:float):
    """
    ARGUMENTS
    ---------
    depths: tensor [b,h,w]
    z_near, z_far: decides "absolute scale"
    
    RETURNS
    -------
    ndc_depth: tensor [b,h,w]
        ndc_depth. 
    """
    z_near = max(z_near, depths.min().item())
    z_far = min(z_far, depths.max().item())
    depths = depths.clamp(z_near, z_far)

    return ( ( z_far * depths - z_far*z_near ) / (z_far-z_near) ) / depths

def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        return "error"
    return "ok"


@torch.no_grad()
def get_render_path(cameras:list, lookat:np.ndarray, spin_angle=3.5, n_frames=50 , spin_for=2):

    """
    Let view-vector to be avg. of cameras->lookat vector,
    get spiral camera path around the view-vector.

    Arguments
    ---------
    cameras: list of Camera object.
    spin_angle: the angle between view-vector and cameras.
    n_frames: the number of cameras.
    spin_for: the number of spinning.
    """

    # Deg->Radian
    spin_angle = spin_angle*np.pi / 180.0
    
    # Convert to np c2w matrices, get mean c2w.
    cam_c2ws = np.stack([cam_to_numpy_c2w(camera) for camera in cameras]) # (n,4,4)
    mean_c2w = mean_camera_pose(cam_c2ws) 

    # Define pivot c2w matrix.
    up = mean_c2w[:3,1]
    eye = mean_c2w[:3,3]
    c2w_pivot = get_c2w_from_eye(eye, lookat, up)

    # Length between eye and lookat
    l = np.linalg.norm(eye-lookat)

    # get "circle"
    radius_x = math.tan(spin_angle) * l
    radius_y = math.tan(spin_angle) * l

    # make it array.
    radius_x = np.linspace(radius_x/spin_for, radius_x, n_frames * spin_for)
    radius_y = np.linspace(radius_y/spin_for, radius_y, n_frames * spin_for)
    
    x_pivot_coords = np.tile(np.cos(np.linspace(0.0, 2.0*np.pi, n_frames)),spin_for) * radius_x
    y_pivot_coords = np.tile(np.sin(np.linspace(0.0, 2.0*np.pi, n_frames)),spin_for) * radius_y
    z_pivot_coords = np.zeros(n_frames*spin_for)
    
    pivot_coords = np.stack([x_pivot_coords,y_pivot_coords,z_pivot_coords,np.ones_like(z_pivot_coords)],axis=0) #[4,n_frames]
    
    eyes_circle = ((c2w_pivot@pivot_coords).T)[:,:3] # [n_frames, 3]
    c2ws =  np.stack([get_c2w_from_eye(eye_cam, lookat, up) for eye_cam in eyes_circle]) #[n_frames, 3, 3]
    
    result_cams = []
    ref_camera:Camera = cameras[0] # reference for metadata copy.

    for c2w in c2ws:
        w2c = np.linalg.inv(c2w)

        width = ref_camera.image_width
        height = ref_camera.image_height
        rot = w2c[:3,:3].T
        trans = w2c[:3,3]
        fovx = ref_camera.FoVx
        fovy = ref_camera.FoVy
        znear = ref_camera.znear
        zfar = ref_camera.zfar
        world_view_transform = torch.tensor(getWorld2View(rot, trans)).transpose(0, 1).cuda()
        projection_matrix = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0,1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

        result_cams.append( MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform))


    return result_cams
def make_video(imgs, path, fps=32):

    writer = imageio.get_writer(path , fps=fps)
    
    for img in imgs:
        writer.append_data(img)    
    writer.close()

def render_set(model_path,  gaussians:GaussianModel, scene:Scene , background, args):

    input_cameras = scene.camera_motion_module.get_middle_cams()
    look_at = gaussians._xyz.cpu().numpy().mean(axis=0)
    
    views = get_render_path(cameras=input_cameras, 
                            lookat=look_at,
                            spin_for=args.spin_for)
    imgs = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, background)
        img = scene.tone_mapping(render_pkg['render'])
        imgs.append(img)
    imgs = torch.stack(imgs)
    imgs = (imgs.permute(0,2,3,1).cpu().numpy().clip(0.0,1.0) * 255.0 ).astype(np.uint8)

    make_video(imgs, os.path.join(model_path, "render_img.mp4"), args.fps)


@torch.no_grad()
def render_sets(dataset: ModelParams, iteration : int, args):
    
    # [HARDCODING] If hold exists, forcefully turn on the eval mode.
    data_path = dataset.source_path
    if len( [e for e in os.listdir(data_path) if "hold" in e] ) == 1:
        dataset.eval= True
    
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, curve_model=True)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_set(dataset.model_path, gaussians, scene, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fps", default=32, type=int)
    parser.add_argument("--spin_for", default=2, type=int)
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    render_sets(model.extract(args), args.iteration, args)