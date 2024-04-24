
import os
import sys
sys.path.append(os.getcwd())
from utils.system_utils import do_system

import numpy as np
from scene.cameras import Camera, MiniCam
from typing import Iterable, Union
import torchvision.utils
import argparse
from arguments import ModelParams
from scene.gaussian_model import GaussianModel
from scene import Scene
import shutil
from utils.camera_utils import fov2focal
import sqlite3
from scene.cameras import get_c2w
from scene.colmap_loader import rotmat2qvec, qvec2rotmat

def read_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM images")
    images_tuples = c.fetchall()

    c.execute("SELECT * FROM cameras")
    cameras_tuples = c.fetchall()

    return cameras_tuples, images_tuples


def triangulate(cams:Iterable[Camera], output_path:str):
    """
    Using information in cams, run colmap triangulation and create a new colmap workspace.
    i.e. fix camera intrinsic&extrinsic and add point cloud.
    Useful for converting blender (and possibly llff) to colmap.
    
    ARGUMENTS
    ---------
    cams: list of Camera
      - note that it should be Camera, not Minicam object, as this script uses ground-truth image information to begin with.
    output_path
      - path for new workspace of COLMAP.
    """    

    # Workspace.
    image_path = os.path.join(output_path, "images")
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(image_path)

    # Save Images.
    for cam in cams:
        print(cam.image_name)
        image_path_file = os.path.join(image_path, f"{cam.image_name}.png")
        torchvision.utils.save_image(cam.original_image, image_path_file)

    # Configuration.
    db_path = os.path.join(output_path, "database.db")
    sparse_path = os.path.join(output_path, "sparse_txt_tmp")
    shutil.rmtree(sparse_path, ignore_errors=True)
    os.makedirs(sparse_path)
    
    fx = fov2focal(cams[0].FoVx, cams[0].image_width)
    fy = fov2focal(cams[0].FoVy, cams[0].image_height)
    cx, cy = cams[0].image_width/2.0 , cams[0].image_height/2.0
    flag_EAS = 1

    # Feature Extract & Matching.
    do_system("colmap feature_extractor "
              f"--database_path {db_path} " 
              f"--image_path {image_path} "
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

    # Save intrinsic in COLMAP convention.
    with open(os.path.join(sparse_path, "cameras.txt"), "w") as fp:
        print("# \n"*3, end='', file=fp)
        cam = cams[0]
        w = cam.image_width
        h = cam.image_height

        # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        print(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}", file=fp)        

    # Create Empty pointcloud file.
    with open(os.path.join(sparse_path, "points3D.txt"), "w") as fp:
        pass

    # Save Extrinsic.
    with open(os.path.join(sparse_path, "images.txt"), "w") as fp:
        print("# \n"*4, end='', file=fp)
        extr_dic = {}
        for i, cam in enumerate(cams):            
            # Render and Save.
            render_filename = f"{cam.image_name}.png"

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
    sparse_result_path = os.path.join(output_path, "sparse", "0")
    shutil.rmtree(sparse_result_path, ignore_errors=True)
    os.makedirs(sparse_result_path)

    do_system(f"colmap point_triangulator "
              f"--database_path {db_path} "
              f"--image_path {image_path} "
              f"--input_path {sparse_path} "
              f"--output_path {sparse_result_path}")
    
    # Remove pointcloud-less sparse path.
    shutil.rmtree(sparse_path)
    sparse_path = sparse_result_path


    sparse_txt_path = os.path.join(output_path,"sparse_txt")
    shutil.rmtree(sparse_txt_path, ignore_errors=True)
    os.makedirs(sparse_txt_path)

    do_system(f"colmap model_converter "
              f"--input_path {sparse_path} "
              f"--output_path {sparse_txt_path} "
              f"--output_type TXT")
    
    do_system(f"python scripts/colmap_visualization.py --path {output_path} ")
      
    print("[DONE]")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Triangulation script parameters")
    model_params = ModelParams(parser, sentinel=False)
    parser.add_argument("--result_path", type=str, required=True, help="new colmap directory.")

    args = parser.parse_args()

    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    triangulate(scene.getTrainCameras(),output_path=args.result_path)
