
import cv2
import argparse
import os
import sys

import numpy as np
import shutil
from pathlib import Path

sys.path.append(os.getcwd())
from utils.system_utils import do_system
from scene.dataset_readers import read_intrinsics_text

def get_parser()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_path", type=str, required=True)

    parser.add_argument("--image_path", type=str, default=None, 
                        help="Data directory where images are located.")
    parser.add_argument("--video_path", type=str, default=None, 
                        help="Video path. Ignored if --image_path is given.")
    parser.add_argument("--mask_path", type=str, default=None,
                        help="Path for mask. Optional.")
    parser.add_argument("--reverse_mask", action="store_true",
                        help="Reverse mask (1 and 0)")
    parser.add_argument("--resize_factor", type=float, default=None,
                        help="Resize factor")

    parser.add_argument("--video_frame_min", type=int, default=None,
                        help="[Optional] frame clipping min (inclusive).")
    parser.add_argument("--video_frame_max", type=int, default=None,
                        help="[Optional] frame clipping max (exclusive).")
    parser.add_argument("--video_skip", type=int, default=1,
                        help="[Optional] frame skip rate.")
    # parser.add_argument("--mask_path", type=str, default=None,
    #                     help="Optional: motion mask for colmap.")
    
    parser.add_argument("--no_colmap", action="store_true",
                        help="not running colmap")
    parser.add_argument("--keep_image_name", action="store_true",
                        help="disable re-labeling image filename (default: renaming from 0 to n-1.jpg)")

    parser.add_argument("--forceful", action="store_true",
                        help="disable to ask whether the system want to overwrite result directory.")
    # colmap.
    parser.add_argument("--colmap_matcher", default="exhaustive", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images")
    parser.add_argument("--intrinsic_path", default=None, type=str, help="Reference '[COLMAP workspace]/cameras.txt' for intrinsic information. Prior 1: providing this will ignore all intrinsic arguments below.")
    parser.add_argument("--focal_length", nargs="+", default=None, type=float, help="[Optional] provide focal length will fix it and no longer optimize.")
    parser.add_argument("--principal_points", nargs="+", default=None, type=float, help="[Optional] provide principal points will fix it and no longer optimize.")
    
    parser.add_argument("--radial", nargs="+", type=float, default=None, help="[Optional] providing radial parameter (k1, k2, p1, p2) will fix it and no longer optimize.")

    parser.add_argument("--camera_model", default="OPENCV", help="COLMAP camera model.")

    return parser

def maybe_resize(img, args):
    if args.resize_factor is not None:
        w = int(img.shape[1] / args.resize_factor)
        h = int(img.shape[0] / args.resize_factor)
        
        img = cv2.resize(img, (w,h))
    return img

def get_images(args):
    """
    Get images from path.
    """

    imgs = []
    if args.image_path is not None:
        print("Loading images from", args.image_path)
        for i, filename in enumerate(sorted([e for e in os.listdir(args.image_path) if ".jpg" in e or ".png" in e])):
            full_path = os.path.join(args.image_path, filename)
            print(f"{i:03d}")
            print(f"processing: {full_path}")

            img = cv2.imread(full_path)
            img = maybe_resize(img, args)
            imgs.append(img)

        imgs = np.stack(imgs)
        
    elif args.video_path is not None:
        
        assert not args.keep_image_name

        print("Loading video from", args.video_path)
        
        vidcap = cv2.VideoCapture(args.video_path)
        
        success = 1
        while success:
            success, img = vidcap.read()
            if success: 
                img = maybe_resize(img, args)
                imgs.append(img)
        imgs = np.stack(imgs)
        
    else:
        raise Exception("At least one of --image_path or --video_path required.")
    clipping_min = args.video_frame_min if args.video_frame_min is not None else 0
    clipping_max = args.video_frame_max if args.video_frame_max is not None else len(imgs)
    
    imgs = imgs[clipping_min:clipping_max:args.video_skip]

    return imgs

def write_images(args, imgs, folder='images', ext="png"):
    
    image_write_path = os.path.join(args.result_path, folder)
    shutil.rmtree(image_write_path, ignore_errors=True)
    os.makedirs(image_write_path, exist_ok=True)
    
    original_filenames = os.listdir(args.image_path)
    original_filenames.sort()

    for i, img in enumerate(imgs):
        filename = original_filenames[i] if args.keep_image_name else f"{i:05d}.{ext}"
        full_path = os.path.join(image_write_path, filename)
        
        print(f"{i:03d}")
        print(f"writing: {full_path}")

        cv2.imwrite(full_path, img)

def read_sparse_txt(path):
    with open(path, "r") as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            line = line.strip()
            if line[0] == "#":
                continue

            tokens = line.split()

            params = tokens[4:]


def get_camera_param_format(args, imgs):
    
    if args.intrinsic_path is not None:
    
        raise NotImplementedError
        return
    assert args.focal_length is not None
    
    if len( args.focal_length ) == 1:
        fx = fy = args.focal_length[0]
    elif len( args.focal_length) == 2:
        fx, fy = args.focal_length

    if args.principal_points is None:
        h, w = imgs.shape[1:3]
        cx = w/2
        cy = h/2
    else:
        assert len(args.principal_points) == 2
        cx, cy = args.principal_points
    
    if args.radial is not None:
        k1, k2, p1, p2 = args.radial
    else:
        k1, k2, p1, p2 = 0.0, 0.0, 0.0, 0.0

    
    # NOTE
    # Colmap parameter: 
    # SIMPLE_PINHOLE: f cx cy
    # PINHOLE: fx fy cx cy
    # SIMPLE_RADIAL: f cx cy k1
    # RADIAL: f cx cy k1 k2
    # OPENCV: fx fy cx cy k1 k2 p1 p2

    if args.camera_model == "SIMPLE_PINHOLE":
        return f"{fx:.10f},{cx:.10f},{cy:.10f} "
    
    elif args.camera_model == "PINHOLE":
        return f"{fx:.10f},{fy:.10f},{cx:.10f},{cy:.10f} "
    
    elif args.camera_model == "SIMPLE_RADIAL":
        return f"{fx:.10f},{cx:.10f},{cy:.10f},{k1:.10f} "
    
    elif args.camera_model == "RADIAL":
        return f"{fx:.10f},{cx:.10f},{cy:.10f},{k1:.10f},{k2:.10f} "
    
    elif args.camera_model == "OPENCV":
        return f"{fx:.10f},{fy:.10f},{cx:.10f},{cy:.10f},{k1:.10f},{k2:.10f},{p1:.10f},{p2:.10f} "
    


def run_colmap(args, imgs):
    db = os.path.join(args.result_path, "database.db")
    images = os.path.join(args.result_path, "images")
    text = os.path.join(args.result_path, "sparse_txt")
    text_distortion = os.path.join(args.result_path, "sparse_distortion_txt")
    
    mask = os.path.join(args.result_path, "colmap_masks")
    undistortion_tmpdir = os.path.join(args.result_path, "dense")
    cam_model = args.camera_model

    flag_EAS = 1
    is_refining_focal = int(args.focal_length is None)
    is_refining_extra_params = int('PINHOLE' not in cam_model and args.radial is None) 
    is_refining_principal = 0 #int('PINHOLE' not in cam_model and args.focal_length is None)
    
    sparse = os.path.join(args.result_path, "sparse")

    print(f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")
    print(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n) Y")
    # if (input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
    #     sys.exit(1)
    if os.path.exists(db):
        os.remove(db)

    if os.path.exists(mask):
        fextract_additional_command = f" --ImageReader.mask_path {mask}"
    else:
        fextract_additional_command = ""

    if args.focal_length is not None:    
        camera_param_format = get_camera_param_format(args, imgs)
        fextract_additional_command += f" --ImageReader.camera_params {camera_param_format}"
    
    fextract_additional_command += " --SiftExtraction.use_gpu 0 "
    do_system( f"colmap feature_extractor "
               f"--ImageReader.camera_model {cam_model} "
               f"--SiftExtraction.estimate_affine_shape {flag_EAS} "
               f"--SiftExtraction.domain_size_pooling {flag_EAS} "
               f"--ImageReader.single_camera 1 "
               f"--database_path {db} "
               f"--image_path {images} "
                "--SiftExtraction.max_num_features 8192 "
               f"{fextract_additional_command}")
    
    do_system(f"colmap {args.colmap_matcher}_matcher --SiftMatching.guided_matching {flag_EAS} --database_path {db} --SiftMatching.use_gpu 0")
    
    shutil.rmtree(sparse, ignore_errors=True)

    do_system(f"mkdir {sparse}")
    do_system(f"colmap mapper --database_path {db} --image_path {images} --output_path {sparse} "
                "--Mapper.abs_pose_max_error 20 " # 12
                "--Mapper.init_max_error 12 " # 4
                "--Mapper.filter_max_reproj_error 8 " # 4
                "--Mapper.init_max_reg_trials 5 "
                "--Mapper.max_reg_trials 5 "
                "--Mapper.min_num_matches 5 "
                "--Mapper.init_min_num_inliers 30 " # 100
                "--Mapper.abs_pose_min_num_inliers 15 " # 30
                "--Mapper.abs_pose_min_inlier_ratio 0.12 " # 0.25
                "--Mapper.tri_ignore_two_view_tracks 1 "
                "--Mapper.ba_local_max_num_iterations 100 "
                "--Mapper.ba_global_max_num_iterations 100 "
               f"--Mapper.ba_refine_focal_length {is_refining_focal} "
               f"--Mapper.ba_refine_principal_point {is_refining_principal} "
               f"--Mapper.ba_refine_extra_params {is_refining_extra_params} ")
    
    do_system(f"colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 "
              f"--BundleAdjustment.refine_principal_point {is_refining_principal} "
              f"--BundleAdjustment.refine_extra_params {is_refining_extra_params} "
              f"--BundleAdjustment.refine_focal_length {is_refining_focal}")
    try:
        shutil.rmtree(text)
    except:
        pass


    # Undistortion if needed
    if "PINHOLE" not in cam_model:
        # Save Distortion Parameters.
        do_system(f"mkdir {text_distortion}")
        do_system(f"colmap model_converter --input_path {sparse}/0 --output_path {text_distortion} --output_type TXT")
        
        os.makedirs(undistortion_tmpdir)

        do_system(f"colmap image_undistorter --image_path {images} --input_path {sparse}/0 --output_path {undistortion_tmpdir}")

        # Remove distorted images.
        do_system(f"rm -rf {images}")
        do_system(f"rm -rf {sparse}")
        
        os.makedirs(sparse)
        
        do_system(f"mv {undistortion_tmpdir}/images {args.result_path}")
        do_system(f"mv {undistortion_tmpdir}/sparse {sparse}")
        do_system(f"mv {sparse}/sparse {sparse}/0")        
        
        do_system(f"rm -rf {undistortion_tmpdir}")
        

    do_system(f"mkdir {text}")
    do_system(f"colmap model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")
def move_mask(args, imgs):
    """
    Move mask images from path to result.
    """

    mask_write_path = os.path.join(args.result_path, "masks")
    os.makedirs(mask_write_path, exist_ok=True)
    
    if not args.no_colmap:
        colmap_mask_path = os.path.join(args.result_path, "colmap_masks")
        os.makedirs(colmap_mask_path, exist_ok=True)
        
    if args.mask_path is not None:
        print("Loading masks from", args.mask_path)
        
        for i, filename in enumerate(sorted(os.listdir(args.mask_path))):
            full_path = os.path.join(args.mask_path, filename)
            print(f"{i:03d}")
            print(f"processing: {full_path}")

            mask = cv2.imread(full_path)
            # mask = maybe_resize(mask, args)
        
            full_write_path = os.path.join(mask_write_path, filename)        
            print(f"writing: {full_write_path}")

            cv2.imwrite(full_write_path, mask)
            if not args.no_colmap:
                mask = (np.sum(mask,axis=-1)==0).astype(float)
                mask = cv2.resize(mask,(imgs.shape[2],imgs.shape[1]) )
                
                mask = (mask!=0).astype(int)*255
                h,w = mask.shape
                # mask = np.broadcast_to(mask.reshape((h,w,1)),((h,w,3)))

                filename_jpg = f"{i:05d}.jpg"
                full_colmap_mask_path = os.path.join(colmap_mask_path, f"{filename_jpg}.png")
                if args.reverse_mask:
                    mask = 255 - mask

                print(f"writing: {full_colmap_mask_path}")
                
                cv2.imwrite(full_colmap_mask_path, mask )


def warn_and_overwrite(path, forceful=False):
    
    if not os.path.exists(path):
        return
    if forceful:
        print(f"Forceful mode. Overwriting {path} ...")
    else:
        image_path = os.path.join(path, "images")
        if os.path.exists(image_path):
            print("Found previous COLMAP workspace. Overwriting...")
        else:
            user_answer = input(f"{path} does not look like COLMAP workspace. Do you want to Overwrite (y/N)? ").strip().lower()
            
            if user_answer != "y":
                print(f"Not overwriting {path}. Halting...")
                exit(0)

    shutil.rmtree(path)
    os.makedirs(path)

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    warn_and_overwrite(args.result_path, forceful=args.forceful)
    imgs = get_images(args)

    write_images(args, imgs)
    
    if args.mask_path is not None:
        move_mask(args, imgs)
    
    if not args.no_colmap:
        run_colmap(args, imgs)
        do_system(f"python scripts/colmap_visualization.py --path {args.result_path}")
