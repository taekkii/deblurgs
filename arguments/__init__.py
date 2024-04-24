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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 2
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.llffhold = 0
        self.num_initial_pcd = -1
        
        self.num_subframes = 21
        self.curve_order = 9
        self.curve_type = "se3" # ["quarternion_cartesian", "se3"]
        
        self.z_near = 0.2
        self.z_far = 100.0

        self.random_init = False
        self.alpha_lower_bound=0.0 # rm
        self.scale_lb=0.0 # rm
        self.scale_ub=-1.0 # rm
        self.tone_mapping_type = "gamma"
        self.activation = "relu"
        self.use_isotrophic = False
        self.curve_random_sample = False
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 150_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.noise_init = 0.0
        self.noise_final = 0.0 
        self.lambda_t_smooth_init = 1e-4
        self.lambda_t_smooth_final = 1e-5 
        
        self.lambda_depth_tv = 0.001
        self.lambda_hinge = 0.1
        
        self.densification_interval = 200
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 75_000
        self.densify_grad_threshold_init = 4e-4
        self.densify_grad_threshold_final = 2e-4
        self.densify_annealing_until = 25_000
        self.clip_grad = -1.0
                
        # curve optimization factors.
        self.curve_controlpoints_lr = 1e-2
        self.curve_rotation_lr = 1e-3
        self.curve_alignment_lr = 3e-3
        self.curve_lr_half_iter = 15_000
        self.curve_start_iter = 1000
        self.curve_end_iter = 100_000
        self.random_sample_until = 100000
        self.drop_alignment = 0.25
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
