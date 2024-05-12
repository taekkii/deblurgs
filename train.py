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

import os
import torch
import numpy as np
from utils.loss_utils import l1_loss, batchwise_smoothness_loss , hinge_l2, tv_loss
import sys
from scene import Scene, GaussianModel
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, OptimizationParams
from utils.visualization import Visualizer
import utils.general_utils as general_utils
from utils.system_utils import do_system
from utils.logger import Logger
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


        
def training(dataset:ModelParams, opt:OptimizationParams, args):
  
    saving_iterations = args.save_iterations
    checkpoint_iterations = args.checkpoint_iterations
    checkpoint = args.start_checkpoint
    
    render_iterations = args.render_iterations
    is_visualizing_curve = not args.disable_curve_visualize
    
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, load_path=args.load_path)
    cam_motion_module = scene.camera_motion_module

    gaussians.training_setup(opt)
    
    # Load camera motion parameters if path is given.
    if args.load_camera_motion_path is not None: 
        cam_motion_module.load(args.load_camera_motion_path)


    # Add pose params to the optimizer.
    cam_motion_module.add_training_setup(gaussians=gaussians, lr_dict={'curve_rot':opt.curve_rotation_lr,
                                                                       'curve_trans':opt.curve_controlpoints_lr,
                                                                       'curve_alignment':opt.curve_alignment_lr})
    # Gaussian Loader.
    first_iter = 0
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    first_iter += 1
    training_time_sec = 0.0

    # Background color is for Visualizer only.
    # (We provide random background color for training as we want the influence of background to be 0.)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") 
    
    # Prepare logger, visualizer.
    scene_name = [e for e in args.source_path.split("/") if len(e.strip())>0][-1]
    progress_bar = tqdm(range(first_iter, opt.iterations+1), desc=scene_name, ncols=200)
    logger = Logger(progress_bar, ema_weight=0.6)
    visualizer = Visualizer(opt, scene, gaussians, background, vis_cam_idx=args.vis_cam_idx)
    
    # Schedulers
    densify_threshold_func = general_utils.get_expon_lr_func(opt.densify_grad_threshold_init,
                                                             opt.densify_grad_threshold_final,
                                                             max_steps=opt.densify_annealing_until)

    lambda_t_smooth_func = general_utils.get_expon_lr_func( opt.lambda_t_smooth_init,
                                                            opt.lambda_t_smooth_final,
                                                            max_steps=opt.iterations)
    noise_func = general_utils.get_expon_lr_func(opt.noise_init,
                                                 opt.noise_final,
                                                 max_steps=opt.iterations)
    
    alignment_func = general_utils.get_scheduler(lr_init=opt.curve_alignment_lr,
                                                 lr_final=1e-7,
                                                 warmup_ratio=0.0,
                                                 step_warmup=opt.curve_alignment_start,
                                                 step_final=opt.iterations)
    # alignment_func = get_expon_lr_func(opt.curve_alignment_lr,
    #                                    0.0,
    #                                    lr_delay_steps=int(opt.densify_until_iter*opt.drop_alignment),
    #                                    max_steps=opt.iterations)
    
    # alignment_func = get
    # Turn off camera motion optimizer.
    cam_motion_module.alternate_optimization() 

    for iteration in range(first_iter, opt.iterations + 1):        
        
        t0 = time.time()

        # Update scheduled hyperparameters.
        gaussians.update_learning_rate(iteration, opt,alignment_lr=alignment_func(iteration))
        densification_threshold = densify_threshold_func(iteration) if args.flag != 1 else opt.densify_grad_threshold_init
        lambda_t_smooth = lambda_t_smooth_func(iteration)
        
        # Turn on/off camera motion optimizer. 
        if iteration == opt.curve_start_iter or iteration == opt.curve_end_iter:    
            cam_motion_module.alternate_optimization()
        
        # Turn off random sampling pose on the camera motion.
        if iteration == opt.random_sample_until:    
            cam_motion_module.curve_random_sample = False 

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
         
        # Get camera idx and sub-frame indice.        
        cam_idx = scene.get_random_cam_idx()
        if iteration>=opt.curve_start_iter:
            subframe_indice = "all"
        else:
            subframe_indice = 1

        # Render
        retrieved = cam_motion_module.query(cam_idx=cam_idx, 
                                            subframe_indice=subframe_indice)
        
        blur = retrieved['blurred']
        subframes = retrieved['subframes']
        subframe_depths = retrieved['depths']
        gt = retrieved['gt']
        render_pkgs = retrieved['render_pkgs']
        
        # (Optional) Add noise to the GT if desired.
        noise = noise_func(iteration)
        gt = scene.tone_mapping.inverse()(gt) + torch.randn_like(gt)*noise
        
        # [========== Loss =========] #
        Ll1 = l1_loss(blur, gt)
        L_t_smooth = batchwise_smoothness_loss(subframes)
        
        # Depth Smoothness (Optional). Not written in the paper.
        if opt.lambda_depth_tv>0.0:
            L_depth_tv = tv_loss(subframe_depths[:,None,:,:])
        else:
            L_depth_tv = 0.0
            
        # Penalize opacity and t out-of-range (not written in the paper.)
        # (We have replaced opacity activation from sigmoid to identity.)
        if opt.lambda_hinge > 0.0:
            L_hinge = hinge_l2(gaussians._opacity) # + hinge_l2(scene.camera_motion_module._nu)
        loss = Ll1 + \
               lambda_t_smooth * L_t_smooth + \
               opt.lambda_depth_tv * L_depth_tv + \
               opt.lambda_hinge * L_hinge
            
        loss.backward()

        
        with torch.no_grad():
            # Progress bar
            logger.update( {"l1":(Ll1,"ema",".5f"),
                            "smooth":(L_t_smooth,"ema",".7f"),
                            "hinge":(L_hinge,"ema",".7f"),
                            "vel":(alignment_func(iteration),"update",".4f"),
                            "#pts":(gaussians._xyz.shape[0],"update","7d")})
            if iteration % 10 == 0:
                logger.show()

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                scene.camera_motion_module.save(os.path.join(scene.model_path, "cm.pth"))

            # Densification
            if iteration < opt.densify_until_iter:
                # Now that we have more than 1 image in a single training iter, iterate over all viewpoint tensors.
                for render_pkg in render_pkgs:
                    viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, 1.0/len(render_pkgs))

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.densify_and_prune(densification_threshold, scene.cameras_extent)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
       

            # Optimizer step
            if iteration < opt.iterations:
                if opt.clip_grad>0.0:
                    torch.nn.utils.clip_grad_value_([e['params'][0] for e in gaussians.optimizer.param_groups],opt.clip_grad)

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # 
            time_sec = time.time() - t0
            training_time_sec = training_time_sec + time_sec
            # Save and visualize.
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            if iteration in render_iterations:
                visualizer.traj_render(iteration)
            
            if is_visualizing_curve:
                visualizer.run(iteration)

    if is_visualizing_curve:
        visualizer.save_video()
        
    with open(os.path.join(args.model_path,"time.txt") ,"w") as fp:
        print(f"Training Time = {training_time_sec:7.5f}sec" , file=fp)
        
    for rendercode in ["render_spiral", "render_trainview"]:
        do_system(f"python {rendercode}.py --model {args.model_path} --source {args.source_path} "
                f"--resolution {args.resolution} --tone_mapping {args.tone_mapping_type} "
                f"--sh_degree {args.sh_degree} --activation {args.activation}")


def print_args(args):
    path = os.path.join(args.model_path, "args.txt")
    with open(path, "w") as fp:
        for k,v in args.__dict__.items():
            print(k,":",v,file=fp)

def set_output_folder(args):    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
if __name__ == "__main__":

    torch.set_printoptions(precision=4, sci_mode=False)
    np.set_printoptions(precision=4, suppress=True)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
     
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[150_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[50_000, 100_000, 150_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--render_iterations", nargs="+", type=int, default=[25_000, 50_000, 75_000, 100_000, 125_000, 150_000])
    
    parser.add_argument("--disable_curve_visualize", action="store_true", help="Do not use visualizer.")
    parser.add_argument("--vis_cam_idx", type=int, default=None, help="visualizer will focus on [VIS_CAM_IDX]-th camera rendering, instead of overall view.")
    parser.add_argument("--load_camera_motion_path", type=str, default=None, help="Load motion parameters, either .pth file or the workspace directory.")
    parser.add_argument("--load_path", type=str, default=None, help="Load gaussian from.")
    
    parser.add_argument("--flag", type=int, default=None, help="custom flag for hard-coding experiment.")
    args = parser.parse_args(sys.argv[1:])
        
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    os.makedirs(args.model_path, exist_ok=True)

    print_args(args)
    set_output_folder(args)

    training(lp.extract(args), op.extract(args), args )

    # All done
    print("\nTraining complete.")
