
import os
import shutil

from scene import Scene, GaussianModel
from scene.cameras import Camera, MiniCam
from scene.cameras import get_c2w
from utils.graphics_utils import getProjectionMatrix
from gaussian_renderer import render
from arguments import ModelParams, get_combined_args

from utils.image_utils import psnr
from utils.loss_utils import ssim, l1_loss
from lpipsPyTorch import lpips
from scene.colmap_loader import rotmat2qvec, qvec2rotmat
from utils.system_utils import do_system
from utils.graphics_utils import fov2focal
from utils.graphics_utils import getWorld2View, getProjectionMatrix

import torch
import torch.nn as nn
import torch.optim as optim
import roma

from tqdm import tqdm
import copy
import argparse

import random
import math
from PIL import Image
import utils.general_utils 
import utils.colorize
import torchvision.utils
import numpy as np
from scripts.colmap_visualization import read_poses
from utils.colmap_reoder import read_db

class OptimPoseModel(nn.Module):

    def __init__(self, cams:list):
        
        """
        
        Aliases:
        C : curve order
        n : # imgs.
        f : # subframes.
        """
        super().__init__()

        print("Optim Pose Model...")
        
        rots = []
        transes = []
        
        self.cams = cams
        for cam in cams:
            cam:Camera    
            rots.append(torch.from_numpy(cam.R).cuda())
            transes.append(torch.from_numpy(cam.T).cuda())

        rots = torch.stack(rots)
        transes = torch.stack(transes)

        rots_unitquat = roma.rotmat_to_unitquat(rots)
        
        self._rot = nn.Parameter(rots_unitquat.float().clone().contiguous().requires_grad_(True)) # [n,4]
        self._trans = nn.Parameter(transes.float().clone().contiguous().requires_grad_(True)) # [n,3]

        
    def forward(self,idx):
        cam: Camera
        cam = copy.deepcopy(self.cams[idx])
        
        quat = self._rot[idx] + 1e-8# [4]
        
        unitquat = quat / quat.norm() # [4]
        rotmat = roma.unitquat_to_rotmat(unitquat[None,:]).squeeze() # [3,3]
        trans = self._trans[idx] # [3]

        cam.world_view_transform = torch.eye(4).cuda()
        cam.world_view_transform[:3,:3] = rotmat.T
        cam.world_view_transform[:3, 3] = trans
        cam.world_view_transform = cam.world_view_transform.transpose(0,1)

        cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy).transpose(0,1).cuda()
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]

        return cam

@torch.no_grad()
def evaluate(cams:list, scene: Scene, gaussians:GaussianModel,bg_color:torch.Tensor, vis_dir:str=None):
    """
    Evaluation using test cams.

    RETURNS
    -------
    psnr, ssim, lpips: (float each) metric of current settings.

    """

    if vis_dir is not None:
        vis_path = os.path.join(scene.model_path, vis_dir)
        shutil.rmtree(vis_path, ignore_errors=True)
        os.makedirs(vis_path)
    else:
        vis_path = None
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    n = len(cams)
    for i, cam in enumerate(cams):
        
        gt_image = cam.original_image
        image = scene.tone_mapping(render(cam, gaussians, bg_color)["render"] )
        psnr_test += psnr(image, gt_image).mean().item()
        ssim_test += ssim(image, gt_image).mean().item()
        lpips_test += lpips(image, gt_image, net_type='alex').mean().item()

        if vis_path is not None:
            errormap = utils.colorize.colorize(torch.abs(gt_image - image).permute(1,2,0).mean(dim=-1)).permute(2,0,1)
            torchvision.utils.save_image(gt_image, os.path.join(vis_path, f"{i:03d}_gt.png"))
            torchvision.utils.save_image(image, os.path.join(vis_path, f"{i:03d}_render.png"))
            torchvision.utils.save_image(errormap, os.path.join(vis_path, f"{i:03d}_error.png"))
                    
    
    return psnr_test/n, ssim_test/n, lpips_test/n

def optimize_test_pose(scene: Scene, gaussians:GaussianModel, bg_color:torch.Tensor, num_iter_per_view:int=2000):
    """
    Run iNeRF-like pose optimization for test veiws.
    Note that test camera pose is not accurate for curve-optimized 3DGS scene, so this process is essential. 
    
    RETURNS
    -------
    optimized_cams: list of Camera object, fit to current scene.
    """
    torch.cuda.empty_cache()

    test_cameras = scene.getTestCameras()
    n = len(test_cameras)

    optim_model = OptimPoseModel(test_cameras)
    optim_param_group = [{"params":[optim_model._rot],   'lr': 5e-5, 'name':"rot"},
                         {"params":[optim_model._trans], 'lr': 5e-4, 'name':"trans"}]

    optimizer = optim.Adam(optim_param_group, lr=5e-4, eps=1e-15)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=num_iter_per_view//20,gamma=0.9)
    
    pbar = tqdm(range(num_iter_per_view), desc="Optimizing...")

    l2_error_ema = 0.0
    
    for iteration in pbar:
        
        idx_list = list(range(n))
        random.shuffle(idx_list)
        
        # Run 1 Epoch.
        while len(idx_list) > 0:        
            # Choose one test view.
            idx = idx_list.pop()
            viewpoint_cam = optim_model(idx)
            
            # Loss.
            gt_image = viewpoint_cam.original_image
            image = render(viewpoint_cam, gaussians, bg_color)["render"]
            image = scene.tone_mapping(image).clamp(0.0,1.0)
            loss = l1_loss(image, gt_image)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                l2_error_ema = l2_error_ema * 0.6 + ((gt_image-image)**2).mean().item() * 0.4
    
        lr_scheduler.step()
        if iteration % 20 == 0:
            with torch.no_grad():
                current_psnr = 20 * math.log10(1.0 / math.sqrt(l2_error_ema))
                pbar.set_description(f"Optimizing...PSNR ={current_psnr:6.2f} lr = {optimizer.param_groups[0]['lr']:10.6f}")
                
    return [optim_model(i) for i in range(n)]

@torch.no_grad()
def initialize_test_pose(args:ModelParams, scene: Scene, gaussians:GaussianModel, bg_color:torch.Tensor,exclude = [], old_version=False):
    """
    Only functions when testing without known pose.
    (i.e.) not llffhold-style.
    """
    source_path = args.source_path
    model_path = args.model_path
    if len(scene.getTestCameras()) > 0:
        return
    
    print("Not LLFFHOLD style dataset... Looking for test image without poses.")
    test_image_dir = os.path.join(source_path, "test_images")
    if not os.path.exists(test_image_dir):
        print("No test image detected... Exiting")
        exit()

    # Prepare temporary colmap workspace.
    tmp_colmap_workspace = os.path.join(model_path, "render_colmap")
    shutil.rmtree(tmp_colmap_workspace, ignore_errors=True)
    os.makedirs(tmp_colmap_workspace)

    db_path = os.path.join(tmp_colmap_workspace, "database.db")

    tmp_images = os.path.join(tmp_colmap_workspace, "images_rendered")
    os.makedirs(tmp_images)

    tmp_sparse = os.path.join(tmp_colmap_workspace, "sparse", "1")
    os.makedirs(tmp_sparse)

    flag_EAS = 1

    # Load cams.
    scene.camera_motion_module.load(os.path.join(model_path, "cm.pth"))
    cams = [cam for i,cam in enumerate(scene.camera_motion_module.get_middle_cams()) if i not in exclude]
    
    # Render from train view, save.
    print("Rendering from training view...")
    for i, cam in enumerate(cams):
        cam: MiniCam
        
        # Render and Save.
        render_filename = f"{i:03d}_render.png"
        rendered = render(cam, gaussians, bg_color)["render"]
        rendered = scene.tone_mapping(rendered)
        torchvision.utils.save_image(rendered, os.path.join(tmp_images, render_filename))

    # Save extrinsic => we will do later to keep track with database order.
    # Save intrinsic in COLMAP convention.
    with open(os.path.join(tmp_sparse, "cameras.txt"), "w") as fp:
        print("# \n"*3, end='', file=fp)
        cam = cams[0]
        w = cam.image_width
        h = cam.image_height
        fx = fov2focal(cam.FoVx, w)
        fy = fov2focal(cam.FoVy, h)
        cx = w/2
        cy = h/2

        # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        print(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}", file=fp)        

    # Create Empty pointcloud file.
    with open(os.path.join(tmp_sparse, "points3D.txt"), "w") as fp:
        pass

    # Run colmap with rendered images only.
    do_system("colmap feature_extractor "
              f"--database_path {db_path} " 
              f"--image_path {tmp_images} "
              f"--SiftExtraction.estimate_affine_shape {flag_EAS} "
              f"--SiftExtraction.domain_size_pooling {flag_EAS} "
              f"--ImageReader.single_camera 1 "
              f"--ImageReader.camera_model PINHOLE "
              f"--SiftExtraction.use_gpu 0 "
              f'''--ImageReader.camera_params "{fx},{fy},{cx},{cy}" ''')
    
    do_system(f"colmap exhaustive_matcher "
              f"--database_path {db_path} "
              f"--SiftMatching.guided_matching {flag_EAS} "
              f"--SiftMatching.use_gpu 0 ")
    
    tmp_sparse_pcd = os.path.join(tmp_colmap_workspace,"sparse","2")
    os.makedirs(tmp_sparse_pcd, exist_ok=True)

    # Save Extrinsic.
    with open(os.path.join(tmp_sparse, "images.txt"), "w") as fp:
        print("# \n"*4, end='', file=fp)
        extr_dic = {}
        for i, cam in enumerate(cams):
            cam: MiniCam
            
            # Render and Save.
            render_filename = f"{i:03d}_render.png"

            # Save pose in COLMAP convention.
            c2w = get_c2w(cam)
            w2c = np.linalg.inv(c2w)
            qvec = rotmat2qvec(w2c[:3,:3])
            tvec = w2c[:3,3]
            
            extr_dic[render_filename] = (qvec,tvec)
        _, image_tuples = read_db(db_path=db_path)

        # Follow Database order.
        for i, image_tuple in enumerate(image_tuples):
            render_filename = image_tuple[1]
            qvec, tvec = extr_dic[render_filename]
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            print(i+1, *qvec, *tvec, 1, render_filename, end="\n\n", file=fp)

    # Triangulation. (get PCD)
    inputpath_arg = "--input_path" if not old_version else "--import_path"
    outputpath_arg = "--output_path" if not old_version else "--export_path"
    cmd = f"colmap point_triangulator " + \
              f"--database_path {db_path} " + \
              f"--image_path {tmp_images} " + \
              f"{inputpath_arg} {tmp_sparse} " + \
              f"{outputpath_arg} {tmp_sparse_pcd}"
    do_system(cmd)
    # Prepare test images
    test_image_files = os.listdir(test_image_dir)
    test_image_files.sort()

    tmp_test_images = os.path.join(tmp_colmap_workspace, "test_images")
    shutil.rmtree(tmp_test_images, ignore_errors=True)
    os.makedirs(tmp_test_images)

    for i, test_image_file in enumerate(test_image_files):
        test_image_path = os.path.join(test_image_dir, test_image_file)
        img_pil = Image.open(test_image_path)
        img_pil.save(os.path.join(tmp_test_images,f"{i:03d}.png"))
        print("[DONE]", test_image_path)
    
    # feature extraction and match.
    do_system("colmap feature_extractor "
              f"--database_path {db_path} " 
              f"--image_path {tmp_test_images} "
              f"--SiftExtraction.estimate_affine_shape {flag_EAS} "
              f"--SiftExtraction.domain_size_pooling {flag_EAS} "
              f"--ImageReader.single_camera 1 "
              f"--ImageReader.camera_model PINHOLE "
              f"--SiftExtraction.use_gpu 0 "
              f'''--ImageReader.camera_params "{fx},{fy},{cx},{cy}" ''')
    
    do_system(f"colmap exhaustive_matcher "
              f"--database_path {db_path} "
              f"--SiftMatching.guided_matching {flag_EAS} "
              f"--SiftMatching.use_gpu 0 ")
    

    tmp_sparse_final = os.path.join(tmp_colmap_workspace,"sparse","0")
    shutil.rmtree(tmp_sparse_final, ignore_errors=True)
    os.makedirs(tmp_sparse_final)

    do_system(f"colmap image_registrator "
              f"--database_path {db_path} "
              f"{inputpath_arg} {tmp_sparse_pcd} "
              f"{outputpath_arg} {tmp_sparse_final}")

    tmp_sparse_txt = os.path.join(tmp_colmap_workspace,"sparse_txt")
    shutil.rmtree(tmp_sparse_txt, ignore_errors=True)
    os.makedirs(tmp_sparse_txt)

    do_system(f"colmap model_converter "
              f"--input_path {tmp_sparse_final} "
              f"--output_path {tmp_sparse_txt} "
              f"--output_type TXT")

    do_system(f"python scripts/colmap_visualization.py --path {tmp_colmap_workspace} ")
        
    # Get test images and poses.
    image_txtfile = os.path.join(tmp_sparse_txt, "images.txt")
    
    with open(image_txtfile, 'r') as f:
        lines = f.readlines()

    lines = lines[4:]
    lines = lines[::2]

    test_cams = []

    one_cam = scene.getTrainCameras()[0]
    
    for line in lines:
        tokens = line.strip().split()
        img_name = tokens[-1]
        if "render" in img_name:
            continue

        test_image_path = os.path.join(tmp_test_images, img_name)
        img_pil = Image.open(test_image_path)
        img = utils.general_utils.PILtoTorch(img_pil, img_pil.size).cuda()

        qvec = np.array(list(map(float, tokens[1:5])))
        tvec = np.array(list(map(float, tokens[5:8])))

        R = qvec2rotmat(qvec).T
        T = np.array(tvec)

        
        # world_view_transform = torch.tensor(getWorld2View(R, T)).transpose(0, 1).cuda()
        # projection_matrix = getProjectionMatrix(znear=one_cam.znear, zfar=one_cam.zfar, fovX=one_cam.FoVx, fovY=one_cam.FoVy).transpose(0,1).cuda()
        # full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        # camera_center = world_view_transform.inverse()[3, :3]
        
        test_cam = Camera(1, R, T, one_cam.FoVx, one_cam.FoVy, img, None, img_name, 1)
        # test_cam.original_image = img
        test_cams.append(test_cam)
    
    scene.test_cameras[1.0] = test_cams

    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Testing script parameters")
    model_params = ModelParams(parser, sentinel=False)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--exclude_indice", nargs="+", default=[] , type=int)
    parser.add_argument("--colmap_old_ver", action="store_true")
    args = get_combined_args(parser)
    print("Evaluating...")

    bg_color = torch.ones(3).cuda()
    
    scene_args = model_params.extract(args)

    # [HARDCODING] If hold exists, forcefully turn on the eval mode.
    data_path = scene_args.source_path
    if len( [e for e in os.listdir(data_path) if "hold" in e] ) == 1:
        scene_args.eval= True

    gaussians = GaussianModel(scene_args)
    scene = Scene(scene_args, gaussians, load_iteration=args.iteration, shuffle=False)
    initialize_test_pose(scene_args, scene, gaussians, bg_color, 
                         exclude=args.exclude_indice, 
                         old_version=args.colmap_old_ver)

    # Before fitting test pose.
    before_psnr, before_ssim, before_lpips = evaluate(scene.getTestCameras(), scene, gaussians, bg_color, vis_dir="eval_before")
    print(f"!!! (Unfit) PSNR: {before_psnr:.2f} SSIM: {before_ssim:.3f} LPIPS: {before_lpips:.3f}")

    fit_cams = optimize_test_pose(scene, gaussians, bg_color=torch.ones(3).cuda())

    after_psnr, after_ssim, after_lpips = evaluate(fit_cams, scene, gaussians, bg_color, vis_dir="eval_after")
    print(f"!!! (Fit) PSNR: {after_psnr:.2f} SSIM: {after_ssim:.3f} LPIPS: {after_lpips:.3f}")

    with open(os.path.join(scene_args.model_path, "eval.txt"), "w") as fp:
        print(f"PSNR: {after_psnr:.2f}", file=fp)
        print(f"SSIM: {after_ssim:.3f}", file=fp)
        print(f"LPIPS: {after_lpips:.3f}", file=fp)
        