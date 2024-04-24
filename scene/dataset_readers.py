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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB, RGB2SH
from scene.gaussian_model import BasicPointCloud

from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

from scene.pcd_init import random_pcd_init

import copy
import open3d as o3d


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info, pcd):

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3])
    cam_centers = np.stack(cam_centers) # (n,3)
    if pcd is not None:
        xyzs = pcd.points
        center = xyzs.mean(axis=0)
        dist = np.linalg.norm(cam_centers-center, axis=1)
        radius1 = np.percentile(dist, 10.0) # heuristic
    else:
        dist_matrix = np.linalg.norm(cam_centers - cam_centers[:,None,:] , axis=-1) # (n,n,3) -> (n,n)
        radius1 = np.percentile(dist_matrix,90)
        
    def get_center_and_diag(cam_centers):
        cam_centers = np.stack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=0, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=1)
        diagonal = np.max(dist)
        return center.flatten(), diagonal


    center, diagonal = get_center_and_diag(cam_centers)
    radius2 = diagonal * 1.1

    radius = min(radius1, radius2)
    print(f"pcd-cam radius : {radius1:.2f}")
    print(f"cam-center radius : {radius2:.2f}")
    print(f"Scene Radius = {radius:.2f}")

    return {"translate": None, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    
    permu_idx = [0 for _ in cam_extrinsics]

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        try:
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)
        except FileNotFoundError:
            image_path = image_path[:-4]+".jpg"
            image = Image.open(image_path)
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, depth=None)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# TODO move to somewhere else.    
def get_bds(cam_infos, pcd):
    """
    cam infos
    pcd: (n_pts,3)
    hwf: (3)
    
    RETURNS
    ------
    bds: (n_cam,2)
    """
    h = cam_infos[0].height
    w = cam_infos[0].width
    fx = fov2focal(cam_infos[0].FovX, w)
    fy = fov2focal(cam_infos[0].FovY, h)
    
    K = np.array([[fx,0.0,w/2],[0.0,fy,h/2],[0.0,0.0,1.0]])

    bds = []
    for cam_info in cam_infos:
        
        w2c = np.eye(4)
        w2c[:3,:3] = cam_info.R.T
        w2c[:3,3] = cam_info.T

        pcd_homog = np.pad(pcd,((0,0),(0,1)),mode='constant',constant_values=1.0) # (n,4)
        
        cam_coords = (pcd_homog @ w2c.T)[:,:3] # (n,3)

        depths = cam_coords[:,2] # (n)
        valid = depths>0.01 # (n)
        
        pixel_coords_homog = cam_coords @ np.linalg.inv(K) # (n,3)
        pixel_coords = pixel_coords_homog[:,:2] / pixel_coords_homog[:,2:] # (n,2)

        valid = np.logical_and( valid, pixel_coords[:,0] >= 0)
        valid = np.logical_and( valid, pixel_coords[:,0] < w)
        valid = np.logical_and( valid, pixel_coords[:,1] >= 0)
        valid = np.logical_and( valid, pixel_coords[:,1] < h)

        depths = depths[valid]
        
        near = np.percentile(depths, 0.1)
        far = np.percentile(depths, 99.9)
        bds.append([near,far])
    
    return np.array(bds)
        
def readColmapSceneInfo(args):
    path = args.source_path
    images = args.images
    eval = args.eval
    llffhold = args.llffhold

    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # If llffhold is not specified, try locating "hold=n" file. If such file is detected, use it.
    if llffhold == 0:
        maybe_llff_file = [e for e in os.listdir(path) if "hold=" in e]
        assert len(maybe_llff_file) <= 1, "more than two llffhold indicator detected."
        if len(maybe_llff_file):
            llffhold = int( (maybe_llff_file[0].strip().split("="))[-1] )
            print(f"LLFF Hold is not specified, but we can detect indiactor file: llffhold={llffhold}")

    depths = None
    for i,cam_info in enumerate(cam_infos):
        image_id = int(''.join(c for c in cam_infos[i].image_name if c.isdigit())) # extract numeric part only.
        cam_infos[i] = cam_infos[i]._replace(depth=depths[image_id] if depths is not None else None)
    
    if eval and llffhold>0:
        train_cam_infos = [cam_info for idx, cam_info in enumerate(cam_infos) if int(cam_info.image_name) % llffhold != 0]
        test_cam_infos = [cam_info for idx, cam_info in enumerate(cam_infos) if int(cam_info.image_name) % llffhold == 0]
    else:
        if llffhold > 0 or eval:
            print("[ERROR] One of eval and llffhold is set, while the other is off. Check if something is wrong.")
            exit(1)
        train_cam_infos = cam_infos
        test_cam_infos = []


    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # if not args.random_init:
    # if not os.path.exists(ply_path):
    print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    try:
        xyz, rgb, error = read_points3D_binary(bin_path)
    except:
        xyz, rgb, error = read_points3D_text(txt_path)

    # [Prune high error pcds]
    if args.num_initial_pcd > 0:

        error = error.reshape((-1,))
        percent = min( args.num_initial_pcd / xyz.shape[0] * 100, 100.0)
        error_filter_threshold = np.percentile(error, percent)
        valid_idx = error < error_filter_threshold
        
        xyz = xyz[valid_idx]
        rgb = rgb[valid_idx]
    
    storePly(ply_path, xyz, rgb)
    if args.random_init:
        ply_path = os.path.join(path, "sparse/0/points3D_random_init.ply")
        # if not os.path.exists(ply_path):
            # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        # xyz = np.random.random((num_pts, 3)) * radius * 2 - radius + center[None,:]
        bound_near = (args.z_far-args.z_near)*0.01
        bound_far = (args.z_far-args.z_near)*0.30
        bds = get_bds(train_cam_infos, xyz)
        xyz = random_pcd_init(train_cam_infos, near=args.z_near + bound_near, far=args.z_far - bound_far, num_pcd=num_pts, bds=bds)
        shs = RGB2SH(np.ones((num_pts, 3))*0.01, use_sigmoid=args.activation=="sigmoid")
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs, use_sigmoid=args.activation=="sigmoid"), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs, use_sigmoid=args.activation=="sigmoid") * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    nerf_normalization = getNerfppNorm(train_cam_infos, pcd=None if args.random_init else pcd)

    # filter_pcd(pcd, train_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],depth=None))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", center= np.array([0., 0. ,0.]), radius=1.3):
    # np.array([12.164,-4.05, 10.7]) np.array([0., 0. ,0.])
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []


    ply_path = os.path.join(path, "points3d.ply")

    
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        # xyz = np.random.random((num_pts, 3)) * radius * 2 - radius + center[None,:]
        xyz = random_pcd_init(train_cam_infos, near=2.0, far=8.0, num_pcd=num_pts)
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
        
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    nerf_normalization = getNerfppNorm(train_cam_infos, pcd=None)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


        

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}