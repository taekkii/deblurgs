
import os
import sys

import math
import imageio

import numpy as np
import torch

sys.path.append(os.getcwd())

from gaussian_renderer import render

from scene import Scene
from scene.cameras import get_c2w as cam_to_numpy_c2w
from scene.cameras import c2w_to_cam as numpy_c2w_to_cam

import matplotlib
from utils.mvg_utils import mean_camera_pose, get_c2w_from_eye

from scene.cameras import Camera, MiniCam


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
    cmapper = matplotlib.cm.get_cmap('jet_r')
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
    cmapper = matplotlib.cm.get_cmap('jet_r')
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


@torch.no_grad()
def get_render_path(scene:Scene, spin_angle=5.0, n_frames=50 , spin_for=2):

    """
    Let view-vector to be avg. of cameras->lookat vector,
    get spiral camera path around the view-vector.

    Arguments
    ---------
    scene: 3DGS Scene object.
    spin_angle: the angle between view-vector and cameras.
    n_frames: the number of cameras.
    spin_for: the number of spinning.
    """

    # Deg->Radian
    spin_angle = spin_angle*np.pi / 180.0

    # Camera Objects.
    cameras = scene.camera_motion_module.get_middle_cams()
    
    # Reference for metadata copy.
    ref_camera:Camera = cameras[0]

    # Convert to np c2w matrices, get mean c2w.
    cam_c2ws = np.stack([cam_to_numpy_c2w(camera) for camera in cameras]) # (n,4,4)
    mean_c2w = mean_camera_pose(cam_c2ws) 

    # Define pivot c2w matrix.
    up = mean_c2w[:3,1]
    eye = mean_c2w[:3,3]
    # c2w_pivot = get_c2w_from_eye(eye, lookat, up)
    c2w_pivot = mean_c2w.copy()

    # Define "look-at" by average depth of center-cropped rendering.
    camera_pivot = numpy_c2w_to_cam(ref_cam=ref_camera, c2w=c2w_pivot)
    depth_pivot = render(camera_pivot, scene.gaussians, torch.zeros(3).cuda())['depth']
    _,H,W = depth_pivot.shape
    lookat_z = depth_pivot[:, H//4:H*3//4 , W//4:W*3//4].mean().cpu().numpy()
    lookat = eye + lookat_z * c2w_pivot[:3,2]

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

    for c2w in c2ws:
        result_cams.append(numpy_c2w_to_cam(ref_cam=ref_camera, c2w=c2w))

    return result_cams
def make_video(imgs, path, fps=32):

    writer = imageio.get_writer(path , fps=fps)
    
    for img in imgs:
        writer.append_data(img)    
    writer.close()

def center_crop_with_ratio(x, ratio):
    """
    ARUMENTS
    --------
    x: np.ndarray (b,h,w,c) or [h,w,c]

    RETURNS
    -------
    cropped img (b,h',w',c) or (h',w',c)
    """
    assert 3 <= len(x.shape) <= 4
    is_batched = len(x.shape) == 4
    if not is_batched:
        x = x[None,...]
    
    H,W = x.shape[1:3]
    crop_ch, crop_cw = H/2, W/2
    crop_lenh, crop_lenw = H*ratio, W*ratio

    h1 = int(crop_ch - crop_lenh/2)
    h2 = int(crop_ch + crop_lenh/2)

    w1 = int(crop_cw - crop_lenw/2)
    w2 = int(crop_cw + crop_lenw/2)

    x = x[:,h1:h2,w1:w2,:]
    
    if not is_batched:
        x = x[0]
    
    return x