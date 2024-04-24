

import torch
import torch.nn as nn
import roma
import numpy as np
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import CameraInfo
from arguments import OptimizationParams
from arguments import ModelParams
import utils.pytorch3d_functions as torch3d
from scene.cameras import Camera, MiniCam
from utils.camera_utils import cameraList_from_camInfos
from scipy.spatial.transform import Rotation
import open3d as o3d
import os
import scipy.special
from utils.general_utils import inverse_sigmoid
import random


class BezierModel(nn.Module):

    def __init__(self, initial_points, curve_order, initial_noise=0.001):
        """
        ARGUMENTS
        ---------
        initial_points: torch tensor [n,d]. Starting points for curves.
            where n is number of curves
            and d is dimension of the space.
            e.g.) 5 curves on cartesian space: 
                needs shape of [5,3]
        curve_order (int):
           order of bezier curve.
           set to 1 for linear model.
        initial_noise (float): initial noise for optimization.
        """ 
        super().__init__()
        
        self.curve_order = curve_order
        
        initial_points = initial_points.float().cuda() #  [n,d]
        initial_points = initial_points[:,None,:].repeat(1, curve_order+1, 1) # [n,c+1,d]
        initial_points = initial_points + torch.randn_like(initial_points)*initial_noise # [n,c+1,d]

        self._control_points = nn.Parameter(initial_points.clone().contiguous().requires_grad_(True))
    
        self._bezier_binom_coeff = torch.tensor([scipy.special.binom(self.curve_order, k) for k in range(self.curve_order+1)]).cuda() # [C+1]

    @property
    def device(self):
        return self._control_points.device
    
    def _get_bezier_coeff(self, t):
        """
        ARGUMENTS
        ---------
        t: tensor size of [f, ], ranging from 0.0 to 1.0
        """
        C = self.curve_order

        coeff = (t[:,None] ** torch.arange(C,-1,-1, device=self.device)) * ( (1-t)[:,None] ** torch.arange(0,C+1, device=self.device) ) * self._bezier_binom_coeff # [f, C+1]
        
        return coeff
    
    def forward(self, t:torch.Tensor, idx:int):
        """
        ARGUMENTS
        ---------
        t: tensor [num_samples]
            float tensor in the range of [0,1]
        idx: curve idx.
        
        RETURNS
        -------
        sample_points: [num_samples, dimension]
        """
        if isinstance(idx, int):
            idx = torch.tensor([idx],device=self.device)
        
        sample_points = (self._get_bezier_coeff(t)[:,:,None] * self._control_points[idx]).sum(dim=1) # [f,c+1,1] * [c+1,d] = [f,c+1,d]--(sum)-->[f,d]
        
        return sample_points

    def __len__(self):
        return self._control_points.shape[0]