
import numpy as np
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.mvg_utils import  build_K, get_normalized_coords, normalized_coords_to_cam_coords, cam_to_world_coords

def random_pcd_init(cam_infos, near=0.0, far=8.0, num_pcd=100_000,bds=None):
    """
    generate pcd along to camera frustrum

    """
    all_xyz = []
    d = 50 # num of points per ray
    num_pcd_per_cam = num_pcd // (max(len(cam_infos)-5,1)) + 2
    if bds is not None:
        print(bds.mean(axis=0))  
    for i, cam_info in enumerate(cam_infos):
        rot = cam_info.R.T # NOTE: rotation is transposed for glm-library in CUDA.
        trans = cam_info.T
        
        w2c = np.eye(4)
        w2c[:3,:3] = rot
        w2c[:3,3] = trans
        c2w = np.linalg.inv(w2c)

        w = cam_info.width
        h = cam_info.height
        fx = fov2focal(cam_info.FovX, w)
        fy = fov2focal(cam_info.FovY, h)
        K = build_K(fx*0.8, fy*0.8, w/2, h/2) # spread little bit wider area than original field of view.

        stride_coeff = num_pcd_per_cam**(-1/3)
        stride_h = int(h*stride_coeff)
        stride_w = int(w*stride_coeff)
        # stride_d = int(d*stride_coeff)

        pixel_coords = np.stack( np.meshgrid(np.linspace(0,w-1,w), np.linspace(0,h-1,h), indexing="xy"),axis=-1)
        pixel_coords = pixel_coords[::stride_h, ::stride_w]
        pixel_coords = pixel_coords.reshape((-1,2))
        
        norm_coords = get_normalized_coords(pixel_coords, K)
        
        norm_coords = np.tile(norm_coords, (d*2,1))
        
        cam_near = max(near, bds[i,0] if bds is not None else 0.0)
        cam_far = min(far, bds[i,1] if bds is not None else 999999999.9)

        depth = np.random.random((norm_coords.shape[0]))*(cam_far-cam_near)+cam_near
        cam_coords = normalized_coords_to_cam_coords(norm_coords, depth)[:num_pcd_per_cam]
        
        xyz_world = cam_to_world_coords(cam_coords, c2w)
        all_xyz.append(xyz_world)

    return np.concatenate(all_xyz,axis=0)[:num_pcd]