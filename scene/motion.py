
import os

import torch
import torch.nn as nn
import roma
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import CameraInfo
from scene.bezier import BezierModel
from arguments import ModelParams
import utils.pytorch3d_functions as torch3d
from scene.cameras import Camera, MiniCam
from utils.camera_utils import cameraList_from_camInfos
from gaussian_renderer import render

class CameraMotionModule:
    
    def __init__(self, cam_infos:list, args:ModelParams):
        
        """
        Aliases:
        C : curve order
        n : # imgs.
        f : # subframes.
        """
        print("Initializing Curve Model...")
        
        C = self.curve_order = args.curve_order
        f = self.n_subframes = args.num_subframes
        self.curve_type = args.curve_type
        self.curve_random_sample = args.curve_random_sample
        self.gaussians = None

        self.original_cam = cameraList_from_camInfos(cam_infos, 
                                                     resolution_scale=1, 
                                                     args=args)
        rotations = []
        translations = []
        
        for cam_info in cam_infos:
            cam_info: CameraInfo

            rotations.append(torch.from_numpy(cam_info.R)) # originally transposed
            translations.append(torch.from_numpy(-cam_info.T@cam_info.R.T)) # translation vec of c2w: cam location.

        rotations = torch.stack(rotations).cuda()
        translations = torch.stack(translations).cuda() 

        # Initial Parameters
        self._set_initial_parameters(rotations, translations) 

        # Alignment Parameters
        n = len(self)
        self._nu = nn.Parameter( torch.linspace(1/(f-1), 1.0-(1/(f-1)), f-2)[None,:].repeat(n,1).cuda().contiguous().requires_grad_(True))
    
    def link_gaussian(self, gaussians:GaussianModel):
        """
        Register GaussianModel Object.
        """
        self.gaussians = gaussians

    def add_training_setup(self, gaussians:GaussianModel, lr_dict:dict):
        """
        Extend gaussianmodel optimizer
        by adding pose_optimizer parameters.
        """
        for group in gaussians.optimizer.param_groups:
            if "curve_" in group['name'] and group['params'][0] in gaussians.optimizer.state:
                del gaussians.optimizer.state[group['params'][0]]
        gaussians.optimizer.param_groups = [e for e in gaussians.optimizer.param_groups if 'curve_' not in e['name']]
        
        gaussians.optimizer.add_param_group({'params': self._rot.parameters(),   'lr': lr_dict['curve_rot'], 'name': 'curve_rot'})
        if hasattr(self, "_trans"):
            gaussians.optimizer.add_param_group({'params': self._trans.parameters(),   'lr': lr_dict['curve_trans'], 'name': 'curve_trans'})
        gaussians.optimizer.add_param_group({'params': [self._nu],  'lr': lr_dict['curve_alignment'], 'name': 'curve_alignment'})

    def query(self, cam_idx:int, 
                    subframe_indice="all", 
                    post_process=None, 
                    background="random"):
        """
        Main query method.
        Render a blurry view, and retrieve additional queried data.

        ARGUMENTS
        ---------
        - cam_idx: int
            camera index

        - subframe_indice: "all", list[int] or int
            If "all" (Default), render all subframes.
            If list(or iterable) of int, this indicates subframe indice.
            If int, this indicates the number of subframes to be rendered; indice are evenly-spaced.
        
        - post_process: None or Callable.
            Postprocess (e.g. gamma-correction) for blurry view. Do nothing if None.
        
        - background: "random" or torch.tensor[3]
            background color. random or color in [0.0, 1.0]
        
        RETURNS
        -------
        - retrieved: dictionary
             dictionary of answered query, whose keys are
            - 'blurred': synthesized blurry view. Post_process will be applied here.
            - 'gt': gt observation. (Default)
            - 'subframes': all subframe renderings.
            - 'render_pkgs': list of render_pkgs from 3DGS render function.
            - 'depths': all subframe depth renderings.
        """ 

        assert hasattr(self, "gaussians") and isinstance(self.gaussians, GaussianModel)

        gaussians = self.gaussians
        
        # Configure background color.
        if background == "random":
            bg = torch.rand(3,device=gaussians._xyz.device)
        else:
            bg = background

        # Configure sub-frame cams.
        if subframe_indice == "all":
            subframe_cams = self.get_trajectory(cam_idx)
        else:
            nu = self._sample_nu_from_alignment(cam_idx)
            if isinstance(subframe_indice, int):
                if subframe_indice == 1:
                    subfr_idx = [len(self)//2]
                subfr_idx = torch.linspace(0,nu.shape[0]-1, subframe_indice, device=nu.device).long()
            else:
                subfr_idx = subframe_indice
            nu = nu[subfr_idx]
            subframe_cams = self.get_trajectory(cam_idx, nu)
        
            
        # Main code for render.
        render_pkg_subframes = []
        
        for cam in subframe_cams:
            render_pkg = render(cam, gaussians, bg)
            render_pkg_subframes.append(render_pkg)        
        
        render_subframes = torch.stack([render_pkg['render'] for render_pkg in render_pkg_subframes]) # [f,3,h,w], f is num_subframes.

        # Return Values
        retrieved_dic = {}

        blurred = render_subframes.mean(dim=0) # [3,h,w]
        if post_process is not None:
            blurred = post_process(blurred)
        
        retrieved_dic['blurred'] = blurred
        retrieved_dic['gt'] = self.get_gt_image(cam_idx) # [3,h,w]
        retrieved_dic['subframes'] = render_subframes
        retrieved_dic['depths'] = torch.stack([render_pkg['depth'] for render_pkg in render_pkg_subframes])
        retrieved_dic['render_pkgs'] = render_pkg_subframes
    
        return retrieved_dic

    def get_trajectory(self, idx, t=None):
        """
        idx: int
        t: None or torch.tensor of size [f (#_of_frames)]. 
           (tensor of) position on the trajectory in the range of [0,1].
           if None, sample from alignment parameter "t" of this model.
        RETURN
        ------
        list of MiniCam type objects (which can be used in rasterization later.)
        corresponding to camera idx.
        """

        # sample subframe c2w_rotations, c2w_translations.
        rot_interp, trans_interp = self._sample_c2w_from_nu(idx, t)

        # Convert to list of Minicam objects, and returns.
        return self._c2w_to_minicam(rot_interp, trans_interp, self.original_cam[0])
    
    def _set_initial_parameters(self, rotations, translations):
        """
        set initial parameters.

        ARGUMENTS
        ---------
        rotations: rotation part of c2w matrix [n, 3, 3]
        translations: camera origin (or equivalently, translation part of c2w matrix [n,3])
        """
        n = rotations.shape[0]

        if self.curve_type == "quarternion_cartesian":
            rot_params = roma.rotmat_to_unitquat(rotations) # [n,4]
            self._rot = BezierModel(rot_params, self.curve_order)
            self._trans = BezierModel(translations, self.curve_order, initial_noise=0.01)

        elif self.curve_type == "se3":
            # NOTE: transpose for torch3d convention
            c2w = torch.zeros(n,4,4).cuda()
            c2w[:,:3,:3] = rotations.transpose(-2,-1)
            c2w[:,3,:3] = translations
            c2w[:,3,3] = 1.0

            params = torch3d.se3_log_map(c2w) 
            self._rot = BezierModel(params[:,3:], self.curve_order)
            self._trans = BezierModel(params[:,:3], self.curve_order)
        else:
            raise NotImplementedError
        
    def _sample_nu_from_alignment(self, idx):
        
        device = self._nu.device
        
        nu_mid = self._nu[idx] # [f-2]
        if self.curve_random_sample:
            nu_mid = nu_mid + torch.rand_like(nu_mid) / self.n_subframes - (1/(2*self.n_subframes)) # add some "uncertainty"
     

        # return nu_mid.sort().values # HARDCODING.
        return torch.cat([torch.zeros(1, device=device), nu_mid, torch.ones(1, device=device)]).clamp(0.0, 1.0).sort().values # [f]

    def _sample_c2w_from_nu(self, idx, nu=None):
        """
        ARGUMENTS
        ---------
        idx: curve index.
        t: Tensor of shape [num_subframes,], ranging in [0.0,1.0] 

        RETURNS
        -------
        c2w_rotations: Tensor of shape [num_subframes, 3, 3] 
        c2w_translations: Tensor of shape [num_subframes, 3]
        """
        
        if nu is None:
            nu = self._sample_nu_from_alignment(idx)
        elif torch.is_tensor(nu):
            nu = nu.to(self.device)
        else:
            raise NotImplementedError

        
        if self.curve_type == "quarternion_cartesian":
            rot_quaternion = self._rot(nu,idx) # [f,4]
            rot_quaternion = rot_quaternion / rot_quaternion.norm(dim=1, keepdim=True) # [f,4]
            c2w_rotations = roma.unitquat_to_rotmat(rot_quaternion) # [f,3,3]
            c2w_translations = self._trans(nu,idx) # [f,3]

        elif self.curve_type == "se3":
            se3 = torch.cat([self._trans(nu,idx), self._rot(nu, idx)], dim=1) # [f,6]
            c2w = torch3d.se3_exp_map(se3) # [f,4,4]
            c2w_rotations = c2w[:,:3,:3].transpose(-2,-1)
            c2w_translations = c2w[:,3,:3]

        else:
            raise NotImplementedError
        return c2w_rotations, c2w_translations
        
    def _c2w_to_minicam(self, rots, transes, ref_cam:Camera):
        """
        given batch of rotation and translation in c2w poses,
        returns minicam object.

        ARGUMENTS
        ---------
        rots: [b,3,3]
        transes: [b,3]
        ref_cam: Camera or Minicam object. 
                 Additional attributes (znear, zfar, fov, etc...) will be duplicated from here.
        RETURNS
        -------
        list of minicam objects
        """

        minicam_list = []
        for i, (rot,trans) in enumerate(zip(rots,transes)): # c2w
            
            world_view_transform = torch.eye(4, device=self.device)
            world_view_transform[:3,:3] = rot # NOTE rot.T.T 
            world_view_transform[3,:3] = -trans@rot # NOTE: not [:3,3] for world-view transform.
            
            projection_matrix = ref_cam.projection_matrix
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            minicam_list.append(
                MiniCam(width=ref_cam.image_width,
                        height=ref_cam.image_height,
                        fovy=ref_cam.FoVy,
                        fovx=ref_cam.FoVx,
                        znear=ref_cam.znear,
                        zfar=ref_cam.zfar,
                        world_view_transform=world_view_transform,
                        full_proj_transform=full_proj_transform,)
            )
        
        return minicam_list
    
    def get_gt_image(self, idx):
        return self.original_cam[idx].original_image
        
    def get_depth(self, idx):
        return self.original_cam[idx].original_depth
    
    def __len__(self):
        return len(self._rot)

    @property
    def device(self):
        return self._rot.device
    
    def is_optimizing(self):
        return self._rot._control_points.requires_grad
    
    def alternate_optimization(self):
        """
        Stop optimizing if it was doing. Start optimizing if optimizing process was stopped.
        """
        new_state = not self.is_optimizing()
        
        print("Curve gradient:" , "[On]" if new_state else "[Off]")
        for optimizable in [self._rot, self._trans, self._nu]:
            optimizable.requires_grad_(new_state)
        
    @torch.no_grad()
    def get_middle_cams(self):
        """
        get list of "middle" from the trajectory.
        """
        cams = []
        for i in range(len(self)):
            nu = self._sample_nu_from_alignment(i)
            mid_idx = nu.shape[0]//2
            nu_mid = nu[mid_idx: mid_idx+1]
            cam = self.get_trajectory(i,nu_mid)[0]
            cams.append(cam)
        return cams
    
    
    def save(self, state_dict_path:str):
        """
        Save camera motion parameters.
        """
        
        assert(state_dict_path.endswith(".pth"))
        
        sdict = {"rot": self._rot.state_dict(),
                 "trans": self._trans.state_dict(),
                 "nu": self._nu}

        torch.save(sdict, state_dict_path)
        print("[SAVED] Camera Motion")

    def load(self, path:str):
        """
        Load camera motion parameters.
        """

        if path.endswith(".pth"):
            state_dict_path = path
        else:
            state_dict_path = os.path.join(path,"cm.pth")

        sdict = torch.load(state_dict_path)
        self._rot.load_state_dict(sdict["rot"])
        self._trans.load_state_dict(sdict["trans"])
        self._nu = sdict['nu']
        print("[LOADED] Camera Motion")
