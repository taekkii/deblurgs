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
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from utils.export_utils import make_video, center_crop_with_ratio

def render_set(model_path,  gaussians:GaussianModel, scene:Scene , background, args):
    
    views = scene.camera_motion_module.get_middle_cams()
    imgs = []
    gts = []

    start_idx = args.start_index
    length = args.fps * args.duration
    crop_ratio = args.crop_ratio
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if start_idx<=idx<=start_idx+length:
            render_pkg = render(view, gaussians, background)
            img = scene.tone_mapping(render_pkg['render'])
            gt = scene.getTrainCameras()[idx].original_image
            imgs.append(img)
            gts.append(gt)
        
    imgs = torch.stack(imgs)
    gts = torch.stack(gts)

    imgs = (imgs.permute(0,2,3,1).cpu().numpy().clip(0.0,1.0) * 255.0 ).astype(np.uint8)
    gts = (gts.permute(0,2,3,1).cpu().numpy().clip(0.0,1.0) * 255.0 ).astype(np.uint8)

    imgs = center_crop_with_ratio(imgs, ratio=crop_ratio)
    gts = center_crop_with_ratio(gts, ratio=crop_ratio)

    make_video(imgs, os.path.join(model_path, "render_trainview_img.mp4"), args.fps)
    make_video(gts, os.path.join(model_path, "render_trainview_gt.mp4"), args.fps)
    make_video(np.concatenate([gts,imgs],axis=2), os.path.join(model_path, "render_trainview_all.mp4"), args.fps)


@torch.no_grad()
def render_sets(dataset: ModelParams, iteration : int, args):
    
    # [HARDCODING] If hold exists, forcefully turn on the eval mode.
    data_path = dataset.source_path
    if len( [e for e in os.listdir(data_path) if "hold" in e] ) == 1:
        dataset.eval= True
    
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, curve_model=True)
    scene.camera_motion_module.load(dataset.model_path)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_set(dataset.model_path, gaussians, scene, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fps", default=10, type=int)
    parser.add_argument("--start_index", default=0, type=int)
    parser.add_argument("--duration", default=20.0, type=float)
    parser.add_argument("--crop_ratio", default=0.95, type=float)
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    render_sets(model.extract(args), args.iteration, args)