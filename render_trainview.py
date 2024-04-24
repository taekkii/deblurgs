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
import imageio
import matplotlib
import torchvision.transforms.functional

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

def get_c2w_from_eye(eye, lookat, up):
    # get c2w matrix for pivot camera.
    z_vec = lookat-eye
    x_vec = np.cross(up,z_vec) 
    y_vec = np.cross(z_vec,x_vec)

    x_vec = x_vec/np.linalg.norm(x_vec)
    y_vec = y_vec/np.linalg.norm(y_vec)
    z_vec = z_vec/np.linalg.norm(z_vec)
    
    rot_pivot = np.stack([x_vec,y_vec,z_vec],axis=0).T
    
    c2w = np.eye(4)
    c2w[:3,:3] = rot_pivot
    c2w[:3,3] = eye
    return c2w
@torch.no_grad()
def get_render_path(ref_camera:Camera, eye=None, lookat = [0.,0.,0.], up=[0., 1., 0.], radius_x=(0.5,1.0), radius_y=(0.2,0.5), n_frames=50 , spin_for=2, look_distance=1.0):

    if eye is not None:
        # get c2w matrix for pivot (eye->lookat ^ up matrix)   
        eye = np.array(eye)
        lookat = np.array(lookat)
        up = np.array(up)
        c2w_pivot = get_c2w_from_eye(eye,lookat,up)
    else:
        # get c2w pivot from ref_camera. 
        w2c_pivot=np.eye(4)
        w2c_pivot[:3,:3] = ref_camera.R.T
        w2c_pivot[:3,3] = ref_camera.T
        c2w_pivot = np.linalg.inv(w2c_pivot)
        lookat = c2w_pivot[:3,3] + c2w_pivot[:3,2]*look_distance
        up = c2w_pivot[:3,1]
    
    # get "circle"
    if isinstance(radius_x,float): radius_x = (radius_x*(1/spin_for), radius_x)
    if isinstance(radius_y,float): radius_y = (radius_y*(1/spin_for), radius_y)
    
    radius_x = np.linspace(radius_x[0], radius_x[1], n_frames * spin_for)
    radius_y = np.linspace(radius_y[0], radius_y[1], n_frames * spin_for)
    
    x_pivot_coords = np.tile(np.cos(np.linspace(0.0, 2.0*np.pi, n_frames)),spin_for) * radius_x
    y_pivot_coords = np.tile(np.sin(np.linspace(0.0, 2.0*np.pi, n_frames)),spin_for) * radius_y
    z_pivot_coords = np.zeros(n_frames*spin_for)
    
    pivot_coords = np.stack([x_pivot_coords,y_pivot_coords,z_pivot_coords,np.ones_like(z_pivot_coords)],axis=0) #[4,n_frames]
    
    eyes_circle = ((c2w_pivot@pivot_coords).T)[:,:3] # [n_frames, 3]
    c2ws =  np.stack([get_c2w_from_eye(eye_cam, lookat, up) for eye_cam in eyes_circle]) #[n_frames, 3, 3]
    
    cams = []
    from utils.graphics_utils import getWorld2View, getProjectionMatrix
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

        cams.append( MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform))


    return cams

def make_video(imgs, path, fps=32):

    writer = imageio.get_writer(path , fps=fps)
    
    for img in imgs:
        writer.append_data(img)    
    writer.close()

def render_set(model_path,  gaussians:GaussianModel, scene:Scene , background,z_near,z_far,render_parameter, fps=20):
    render_path = os.path.join(model_path, "video_frame")
    
    makedirs(render_path, exist_ok=True)
    views = scene.getTrainCameras()
    
    DEFAULT_RENDER_PARAMETER = [0.11,0.07,1.0]
    for i in range(len(render_parameter),len(DEFAULT_RENDER_PARAMETER)):
        render_parameter.append(DEFAULT_RENDER_PARAMETER[i])

    radius_x, radius_y, look_distance = render_parameter
    
    # views = get_render_path(ref_camera=train_cameras[len(train_cameras)//2],radius_x=radius_x, radius_y=radius_y, look_distance=look_distance)
    imgs = []
    depths = []
    gts = []

    START = 0
    LENGTH = fps*20
    CROP = 0.95
    # gaussians._features_dc[:]=0.0 #{hardcoding}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if START<=idx<=START+LENGTH:
            render_pkg = render(view, gaussians, background)
            img = scene.tone_mapping(render_pkg['render'])
            gt = view.original_image
            depth = render_pkg['depth']
            imgs.append(img)
            depths.append(depth.squeeze())
            gts.append(gt)
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    imgs = torch.stack(imgs)
    gts = torch.stack(gts)
    # imgs = torch.cat([imgs,gts], dim=2)
    depths = torch.stack(depths) 
    imgs = (imgs.permute(0,2,3,1).cpu().numpy().clip(0.0,1.0) * 255.0 ).astype(np.uint8)
    gts = (gts.permute(0,2,3,1).cpu().numpy().clip(0.0,1.0) * 255.0 ).astype(np.uint8)

    # depths = depths_to_ndc_z(depths, z_near, z_far)
    # disp = 1.0/(depths+1e-6)
    # disp = disp - disp.min()
    # disp = disp - disp.max()
    # disp = disp * 255.0
    # disp = disp.to(dtype=torch.uint8)

    # disp = torchvision.transforms.functional.equalize(disp[:,None,:,:])
    # disp = disp.squeeze().float()

    # # disparity = depth_to_normalized_disparity(depths, z_near, z_far)
    # depths = depth_colorize(depths, 0.0, 1.0)
    
    H,W = imgs.shape[1:3]
    crop_ch, crop_cw = H/2, W/2
    crop_lenh, crop_lenw = H*CROP, W*CROP

    h1 = int(crop_ch - crop_lenh/2)
    h2 = int(crop_ch + crop_lenh/2)

    w1 = int(crop_cw - crop_lenw/2)
    w2 = int(crop_cw + crop_lenw/2)

    imgs = imgs[:,h1:h2,w1:w2,:]
    gts = gts[:,h1:h2, w1:w2,:]

    make_video(imgs, os.path.join(model_path, "render_trainview_img.mp4"), fps)
    make_video(gts, os.path.join(model_path, "render_trainview_gt.mp4"), fps)
    make_video(np.concatenate([gts,imgs],axis=2), os.path.join(model_path, "render_trainview_all.mp4"),fps)
    # make_video(depths, os.path.join(model_path, "render_trainview_depth.mp4"),fps)


@torch.no_grad()
def render_sets(dataset: ModelParams, iteration : int, z_near=0.01, z_far=100.0, render_parameter=[]):
    
    # [HARDCODING] If hold exists, forcefully turn on the eval mode.
    data_path = dataset.source_path
    if len( [e for e in os.listdir(data_path) if "hold" in e] ) == 1:
        dataset.eval= True
    
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, curve_model=False)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_set(dataset.model_path, gaussians, scene, background, z_near, z_far, render_parameter)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_parameter", type=float, nargs="+", default=[], help="radius_x, radius_y, z_distance of cylindrical cone.")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, args.z_near,args.z_far, args.render_parameter)