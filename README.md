# DeblurGS: Gaussian Splatting for Camera Motion Blur
Jeongtaek Oh, Jaeyoung Chung, Dongwoo Lee and Kyoung Mu Lee <br>

🚧 Under Construction: Please wait for the cleanup.🚧

### Setup

#### Local Setup

```shell
git clone https://github.com/taekkii/deblurgs.git --recursive
cd deblurgs
conda env create --file environment.yml
conda activate deblurgs
```

### Quickstart

#### LLFFHOLD Convention Dataset 
```shell
CUDA_VISIBLE_DEVICES=$gpu python train.py --source_path $PATH_TO_DATASET --model_path $PATH_TO_OUTPUT --eval --resolution $RESOLUTION
```

#### Custom Dataset
```shell
CUDA_VISIBLE_DEVICES=$gpu python train.py --source_path $PATH_TO_DATASET --model_path $PATH_TO_OUTPUT
```



### Arguments


<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --resolution / -r
  Resolution
  #### --num_subframes
  Number of subframes per blurry observation.
  #### --random_init
  (Optional) Spread random point clouds within camera frustrums, uniformly.
  

  ### Camera Motion

  #### --curve_controlpoints_lr
  Learning rate for bezier curve control points.
  #### --curve_rotation lr
  Learning rate for bezier curve rotation.
  #### --curve_alignment_lr.
  Learning rate for alignment parameter
  #### --curve_lr_half_iter.
  Learning rate for ```CURVE_CONTROLPOINTS_LR``` and ```CURVE_ROTATION_LR``` exponentially decays to be half for each ```CURVE_LR_HALF_ITER```. 
  #### --curve_start_iter.
  Runs original 3DGS pipeline before ```CURVE_START_ITER```.
  #### --curve_order
  Bezier curve order.

  ### Important Parameters

  #### --densify_grad_threshold_init / --densify_grad_threshold_init
  Gaussian Densification Annealing Strategy: the threshold exponentially decays from ```DENSIFY_GRAD_THRESHOLD_INIT``` to ```DENSIFY_GRAD_THRESHOLD_FINAL```
  
  #### --lambda_t_smooth_init / --lambda_t_smooth_final
  Temporal smoothness: regularizes abrupt change between adjacent sub-frame renderings. Exponentially decays from ```LAMBDA_T_SMOOTH_INIT``` to ```LAMBDA_T_SMOOTH_FINAL```
  
  #### --lambda_depth_tv
  (Optional) 2D TV loss for depth map. Helpful for smooth geometry, but trades reconstruction quality off.
  
 
</details>
<br>


## Field-captured Video.


```shell
python ./scripts/run_colmap.py --video_path VIDEO/PATH/video.mp4 --result_path COLMAP_WORKSPACE --focal_length OPTIONAL_FOCAL_LENGTH_IF_KNOWN
```

Do not include ```--eval```

```shell
CUDA_VISIBLE_DEVICES=$gpu python train.py --source_path COLMAP_WORKSPACE --model_path $PATH_TO_OUTPUT
```
