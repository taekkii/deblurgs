
import os
from scene.cameras import Camera, get_c2w, c2w_to_cam
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import OptimizationParams

from scene.motion import CameraMotionModule
import torch
import torchvision
import torch.nn as nn
import numpy as np
from gaussian_renderer import render
import math
import cv2
import matplotlib.pyplot as plt
import render_spiral
import shutil
import utils.colorize
import open3d as o3d
import utils.mvg_utils


class Visualizer:
    
    def __init__(self, opt:OptimizationParams, scene: Scene, gaussians: GaussianModel, 
                 bg_color, 
                 alignment_vis_folder="vis_alignment", traj_vis_folder = "vis_traj",
                 n_visualize_shots:int=200,
                 exponential:float=1.7,
                 vis_cam_idx = None):
        """
        Visualization class.
        
        ARGUMENTS
        ---------
        scene: Scene class
        gaussians: Gaussian Splatting model.
        n_visualization_shots:
            number of visualization shots.
        exponential:
            visualization iteration will be decided by shape of function f(x)={x^exponential}
        """

        print("Initializing Visualizer...")
        self.gaussians = gaussians
        self.scene = scene
        self.bg_color = bg_color
        self.draw_camera = True
        self.num_visualize_subframes = 3

        self.vis_iters = self._get_vis_iteration(n_visualize_shots=n_visualize_shots, alpha=exponential, n_iters=opt.iterations)

        # Prepare Alignment Visualizer.
        n = len(scene.camera_motion_module)
        possible_indice = np.arange(n)
        self.selected_indice = np.random.choice(possible_indice, size=9, replace=False)
        self.selected_indice.sort()
        
        # Prepare Trajectory Visualizer.
        self.ref_camera = self._get_visualization_camera(scene, gaussians, vis_cam_idx)
        self.cam_scale = self._get_camera_scale(scene, gaussians)

        # Prepare directory.
        self.alignment_path = os.path.join(scene.model_path, alignment_vis_folder)
        self.traj_vis_path = os.path.join(scene.model_path, traj_vis_folder)

        shutil.rmtree(self.alignment_path, ignore_errors=True)
        os.makedirs(self.alignment_path)
        
        shutil.rmtree(self.traj_vis_path, ignore_errors=True)
        os.makedirs(self.traj_vis_path)
        print("[Done] Initializing Visualizer.")
        

    def _get_vis_iteration(self, n_visualize_shots, alpha, n_iters=30000):
        a = n_iters / n_visualize_shots ** (alpha)

        visualize_iters = a * (np.arange(1, n_visualize_shots+1).astype(float))**alpha
        visualize_iters = visualize_iters.astype(int)
        return visualize_iters
        
    def _get_visualization_camera(self, scene:Scene, gaussians:GaussianModel, vis_cam_idx=None, threshold = 0.5):
        """
        obtain "reasonable" camera to watch observation process.
        """
        
        if vis_cam_idx is not None:
            self.draw_camera = False
            return self.sample_subframe_cams(idx=vis_cam_idx, num_subframes=1)[0]
        
        print(" ==> searching for reasonable camera")
        
        lookat = gaussians._xyz.detach().cpu().numpy().mean(axis=0)
        pts = np.stack([cam.camera_center.cpu().numpy() for cam in scene.getTrainCameras()]) # (n,3)
        
        # Binary search for the lowest zoom which can see all cameras.

        zoom_lb, zoom_ub = 1.5, 100.0

        while zoom_ub - zoom_lb >= 1e-3:
            zoom = (zoom_lb+zoom_ub) / 2.0
            c2ws = np.stack([get_c2w(cam) for cam in scene.getTrainCameras()])
            mean_c2w = utils.mvg_utils.mean_camera_pose(c2ws)
            eye = mean_c2w[:3,3]
            up = mean_c2w[:3,1]
            zoomout_eye = lookat + zoom * (eye-lookat)
            zoomout_c2w = utils.mvg_utils.get_c2w_from_eye(zoomout_eye,lookat,up)
            zoomout_cam = c2w_to_cam(ref_cam=scene.getTrainCameras()[0], c2w=zoomout_c2w)
            W,H  = zoomout_cam.image_width, zoomout_cam.image_height

            pts_hom = np.pad(pts, ((0,0),(0,1)), 'constant', constant_values=1.0) # (n,4)
            pts_cam_hom = pts_hom @ zoomout_cam.world_view_transform.cpu().numpy() # (n,4)
            pts_cam = pts_cam_hom[:,:3] / pts_cam_hom[:,3:] # (n,3)
            pts_cheirality = pts_cam[:,2] >= 0.1 # (n,)

            pts_ndc_hom = pts_hom @ zoomout_cam.full_proj_transform.cpu().numpy() # (n,4)
            pts_ndc = pts_ndc_hom[:,:3] / pts_ndc_hom[:,3:] # (n,3)
            
            pts_pix = (( pts_ndc[:,:2] + 1.0) * np.array([zoomout_cam.image_width,zoomout_cam.image_height]).astype(float) -1.0) * 0.5 # (n,2)

            pts_inside = np.logical_and( np.logical_and( pts_pix[:,0] >= -threshold*W , pts_pix[:,0] <= (1.0+threshold)*W) , 
                                         np.logical_and( pts_pix[:,1] >= -threshold*H , pts_pix[:,1] <= (1.0+threshold)*H) )
            
            pts_good = np.logical_and(pts_inside, pts_cheirality)

            if pts_good.all():
                zoom_ub = zoom
            else:
                zoom_lb = zoom
        
        return zoomout_cam

    def _get_camera_scale(self, scene, gaussians):
        return 0.5   

    @torch.no_grad()
    def draw_cone_on_render_img(self, cam_render: Camera, rendered_img:np.ndarray, cams_for_draw:list, scale=1.0, color=np.array([0,0,255])):
        """
        Draw camera cone on the rendered_img, which is rendered from cam.
        
        ARGUMENTS
        ---------
        cam_render: Camera object used for render 'rendered_img'
        rendered_img: np.array (H,W,3)
        cam_for_draw: Camera object to be painted. 
        scale: float. Decides how large the cone is.
        color: RGB format
        RETURNS
        -------
        rendered_img with cam cone.
        """
        if not self.draw_camera:
            return rendered_img
        
        color = np.ascontiguousarray(color[::-1]) # to BGR format for cv2
        H,W,_ = rendered_img.shape
        
        c2ws_draw = np.stack([get_c2w(cam) for cam in cams_for_draw])
        for cam_draw, c2w_draw in zip(cams_for_draw, c2ws_draw):
            cone_x, cone_y = math.tan(cam_draw.FoVx/2), math.tan(cam_draw.FoVy/2)

            cone_camera_draw_space_homog = np.pad(np.array([[ 0.0   ,   0.0  , 0.0],
                                                            [ cone_x,  cone_y, 1.0],
                                                            [ cone_x, -cone_y, 1.0], 
                                                            [-cone_x, -cone_y, 1.0],
                                                            [-cone_x,  cone_y, 1.0]]) * scale , 
                                                ((0,0),(0,1)),
                                                'constant',
                                                constant_values=1.0) # (5,4)
            
            cone_world_space_homog = cone_camera_draw_space_homog @ c2w_draw.T # (5,4)
            
            cam_hom = cone_camera_draw_space_homog @ cam_render.world_view_transform.cpu().numpy() # (5,4)
            if (cam_hom[:,2]/cam_hom[:,3] < 0.1).any():
                continue
            
            ndc_hom = cone_world_space_homog @ cam_render.full_proj_transform.cpu().numpy() # (5,4)
            ndc = ndc_hom[:,:3] / ndc_hom[:,3:] # [5,3]

            pix = (( ndc[:,:2] + 1.0) * np.array([W,H]).astype(float) -1.0) * 0.5 # [5,2]
            connectivity = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
            for i,j in connectivity:
                try:
                    rendered_img = cv2.line(rendered_img, pix[i].astype(int).tolist(), pix[j].astype(int).tolist(), color.tolist(), thickness=1)
                except Exception as e:
                    pass # TODO do something later
                    # print("[ERROR]" ,e)
        return rendered_img
            
    @torch.no_grad()
    def render_gaussian_and_cams(self, iteration):
        
        tonemapping = self.scene.tone_mapping

        rendered = tonemapping(render(self.ref_camera, self.gaussians, self.bg_color)["render"]).permute(1,2,0).cpu().numpy()
        rendered = np.ascontiguousarray((rendered * 255.0).clip(0.0,255.0).astype(np.uint8)[:,:,::-1])
        
        color1 = np.array([0,255,255])
        color2 = np.array([255,255,0])
        t = np.linspace(0, 1, len(self.scene.camera_motion_module))[:,None]
        colors = ((1-t)*color1 + t*color2).astype(np.uint8)
        
        for i ,color in enumerate(colors):
            subframe_cams = self.sample_subframe_cams(i, num_subframes=5)
            rendered = self.draw_cone_on_render_img(self.ref_camera, rendered, subframe_cams ,scale=self.cam_scale,color=color)

        cv2.imwrite(os.path.join(self.traj_vis_path, f"{iteration:05d}.png" ), rendered) # RGB2BGR

    def sample_subframe_cams(self, idx, num_subframes=None):
        t = self.scene.camera_motion_module._sample_nu_from_alignment(idx)
        if num_subframes is not None:
            subfr_idx = torch.linspace(0,t.shape[0]-1, num_subframes, device=t.device).long()
            t = t[subfr_idx]
        traj = self.scene.camera_motion_module.get_trajectory(idx, t)
        
        return traj
    @torch.no_grad()
    def visualize_alignment(self, iteration):
        
        # Create a 3x3 grid of subplots
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))

        # Generate and plot data in each subplot
        for i in range(3):
            for j in range(3):

                # get camera index for visualization.
                idx = self.selected_indice[i*3+j]
                
                nu = self.scene.camera_motion_module._sample_nu_from_alignment(idx).detach().cpu().numpy()
                nu_pivot = self.scene.camera_motion_module._nu[idx].sort().values.detach().cpu().numpy()
                y = np.linspace(0.0,1.0,nu.shape[0])
                # Plotting the histogram in the current subplot
                
                axes[i,j].plot(nu, y, 'o', markersize=2)

                y = np.linspace(0.0, 1.0,nu_pivot.shape[0])
                axes[i,j].plot(nu_pivot, y, 'o', color="red", markersize=3)
                
                # Adding labels and title to each subplot
                axes[i, j].set_xlabel('nu')
                axes[i, j].set_title(f'Idx {idx}')

                axes[i, j].set_ylim(-0.1,1.1)

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Save the entire figure as an image file 
        plt.savefig(os.path.join( self.alignment_path, f"{iteration:05d}.jpg") )

        plt.close()

    @torch.no_grad()
    def run(self, current_iter):
        if current_iter in self.vis_iters:
            self.render_gaussian_and_cams(current_iter)
            self.visualize_alignment(current_iter)


    @torch.no_grad()
    def traj_render(self, current_iter):

        render_path = f"{self.scene.model_path}/traj_render_{current_iter:05d}" 
        shutil.rmtree(render_path, ignore_errors=True)
        os.makedirs(render_path)
        
        print(f"Iter {current_iter} : render traj.")
        cmm:CameraMotionModule = self.scene.camera_motion_module
        
        for i in range(len(self.scene.camera_motion_module)):
            
            blur_retrieve:dict = cmm.query(cam_idx=i, 
                                           subframe_indice="all", 
                                           post_process=self.scene.tone_mapping)
            blurred = blur_retrieve["blurred"]
            gt = blur_retrieve["gt"]

            subframe_renders = cmm.query(cam_idx=i, 
                                         subframe_indice=self.num_visualize_subframes, 
                                         post_process=self.scene.tone_mapping)["subframes"]
            for j, subframe_render in enumerate(subframe_renders):
                torchvision.utils.save_image(self.scene.tone_mapping(subframe_render).clamp(0.0,1.0), 
                                             os.path.join(render_path, f"{i:03d}_{j:02d}.png"))
            
            torchvision.utils.save_image(blurred.clamp(0.0,1.0), os.path.join(render_path, f"{i:03d}_blur.png"))
            torchvision.utils.save_image(gt, os.path.join(render_path, f"{i:03d}_gt.png"))

            errormap = utils.colorize.colorize(torch.abs(blurred - gt).permute(1,2,0).mean(dim=-1)).permute(2,0,1)
            torchvision.utils.save_image(errormap, os.path.join(render_path, f"{i:03d}_l1.png"))


            

    def save_video(self):
        for video_frame_path in [self.alignment_path, self.traj_vis_path]:
            files = [e for e in os.listdir(video_frame_path) if e.endswith(".png") or e.endswith(".jpg")]
            files.sort(key=lambda x:int(x.split(".")[0]))
            imgs = []
            for file in files:
                full_path = os.path.join(video_frame_path, file)            
                img = cv2.imread(full_path)[:,:,::-1]
                imgs.append(img)
            
            imgs = np.stack(imgs)
            video_name = video_frame_path.split("/")[-1].strip()
            
            render_spiral.make_video(imgs, os.path.join(self.scene.model_path,f"{video_name}.mp4"),fps=20)

