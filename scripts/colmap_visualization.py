
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d

import argparse
import math
import cv2
import shutil

sys.path.append(os.getcwd())
from utils.system_utils import do_system
from utils.mvg_utils import build_K, to_w2c

def read_intrinsic(camera_path):

    with open(camera_path, 'r') as f:
        lines = f.readlines()

    content = lines[3]
    elements = content.strip().split()
    print(*elements)

    type_cam = elements[1]
    print(f"CAM_TYPE: {type_cam}")

    if type_cam == "PINHOLE":
        fx, fy, cx, cy = map(float, elements[-4:] )
    elif type_cam == "SIMPLE_PINHOLE":
        fx, cx, cy = map(float, elements[-3:] )
        fy = fx
    else:
        raise NotImplementedError
    
    return fx, fy, cx, cy

def read_poses(images_path):
    if not os.path.exists(images_path):
        raise Exception(f"No such file : {images_path}")

    with open(images_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise Exception(f"Invalid cameras.txt file : {images_path}")

    comments = lines[:4]
    contents = lines[4:]

    data = []
    
    for i, content in enumerate(contents[::2]):
        content_items = content.split(' ')
        q_xyzw = np.array(content_items[2:5] + content_items[1:2], dtype=np.float32) # colmap uses wxyz
        t_xyz = np.array(content_items[5:8], dtype=np.float32)
        img_name = content_items[9]

        R = Rot.from_quat(q_xyzw).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, -1] = t_xyz

        data.append((img_name, T))
        
    data.sort(key=lambda x:int( ''.join(c for c in x[0] if c.isdigit())) )
    poses = np.stack([pose for img_name, pose in data])
    return poses

def read_point3d_txt(point3d_path):
    if not os.path.exists(point3d_path):
        raise Exception(f"No such file : {point3d_path}")

    with open(point3d_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise Exception(f"Invalid cameras.txt file : {point3d_path}")

    comments = lines[:3]
    contents = lines[3:]

    XYZs = []
    RGBs = []
    candidate_ids = {}

    for pt_idx, content in enumerate(contents):
        content_items = content.split(' ')
        pt_id = content_items[0]
        XYZ = content_items[1:4]
        RGB = content_items[4:7]
        error = content_items[7],
        candidate_id = content_items[8::2]
        XYZs.append(np.array(XYZ, dtype=np.float32).reshape(1,3))
        RGBs.append(np.array(RGB, dtype=np.float32).reshape(1, 3) / 255.0)
        candidate_ids[pt_id] = candidate_id
    XYZs = np.concatenate(XYZs, axis=0)
    RGBs = np.concatenate(RGBs, axis=0)

    return XYZs, RGBs, candidate_ids


def write_pose(poses, out_path, stride=5):
    """
    poses: [n,3,4] or [n,4,4] cam2world matrix
    stride: # of camera group per saving (just pass large number like 1000000 to make it one file)
    """
    n = poses.shape[0]
    for i in range(0,n,stride):

        poses_partial = poses[i:i+stride]
        m_cam = None

        for j,pose in enumerate(poses_partial):
            m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            m.transform(pose)
            if m_cam is None:
                m_cam = m
            else:
                m_cam += m

        o3d.io.write_triangle_mesh(os.path.join(out_path, f"cam_{i:03d}.ply"), m_cam)
    # Save the camera coordinate frames as meshes for visualization
    # o3d.io.write_triangle_mesh(filename, m_cam)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./davis_preprocess")
    parser.add_argument("--scale", type=float, default=0.1)
    args = parser.parse_args()


    fx, fy, cx, cy = read_intrinsic(os.path.join(args.path, "sparse_txt", "cameras.txt"))

    poses = read_poses(os.path.join(args.path, "sparse_txt", "images.txt"))

    xyz, rgb, _ = read_point3d_txt(os.path.join(args.path, "sparse_txt", "points3D.txt"))
    print("points", xyz.shape)

    # SCALE
    xyz *= args.scale
    poses[:,:3,3] = poses[:,:3,3] * args.scale
        
    # VISUALIZATION - PCD
    vis_path = os.path.join(args.path, "visualization")

    if os.path.exists(vis_path):
        shutil.rmtree(vis_path)
    os.makedirs(vis_path, exist_ok=True)
    
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    path = os.path.join(vis_path, "pcd.ply")
    o3d.io.write_point_cloud(path, pcd)

    # FOR MY VISUALIZATION - CAMS
    write_pose(np.stack([np.linalg.inv(w2c) for w2c in poses]), vis_path, stride=1)
