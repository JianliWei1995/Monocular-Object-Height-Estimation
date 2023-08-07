# MOHE Net
Estimating the heights of objects in the field of view has applications in many tasks such as robotics, autonomous platforms and video surveillance. Object height is a concrete and indispensable characteristic people or machine could learn and capture. Many actions such as vehicle avoiding obstacles will be taken based on it. Traditionally, object height can be estimated using laser ranging, radar or stereo camera. Depending on the application, cost of these techniques may inhibit their use, especially in autonomous platforms. Use of available sensors with lower cost would make the adoption of such techniques at higher rates. Our approach to height estimation requires only a single 2D image. To solve this problem we introduce the Monocular Object Height Estimation Network (MOHE-Net).

# Requirements
Python 3.8 or later with all requirements.txt dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```

# Pretrained weights
YOLOv5 pretrained weights include
* yolov5s.pt
* yolov5m.pt
* yolov5l.pt
* yolov5x.pt

They are available at [`one drive`](https://buckeyemailosu-my.sharepoint.com/:f:/r/personal/wei_909_buckeyemail_osu_edu/Documents/YOLOv5%20Pre-trained%20Models?csf=1&web=1&e=AUQf3e). Please download and save them into `model\yolov5\weights`.


# Inference
`test.py` runs inference on a variety of image folders. After saving weights, run:
```bash
$ python test.py --source = './datasets/'       # Inference dataset
                 --seq_num = '00'               # Inference folder within source
                 --weights = 'yolov5s.pt'       # YOLOv5 pre-trained weights
                 --img_size = 640               # inference image size (pixels), 1280 for OST strategy
                 --conf_thres = 0.25            # object confidence threshold
                 --iou_thres = 0.45             # IOU threshold for NMS
                 --augment                      # augmented inference
                 --agnostic-nms                 # class-agnostic NMS
```
Our code also supports your own dataset. Please conduct camera calibration and homography estimation before your inference.

# Camera calibration
Our repo support camera calibration using chessboard algorithm. Please save chessboard images into `camera_cali/cali` folder and run [`camera_calibration.ipynb`](https://github.com/OSUPCVLab/Ford2019/blob/master/Monocular%20Object%20Height%20Estimation%20Network%20Using%20Deep%20Learning%20and%20Scene%20Geometry/camera_cali/camera_calibration.ipynb). It returns camera intrinsic matrix and distortion coefficients. Please update `Proj_matrix` and `dist_coeffs` in [`test.py`](https://github.com/OSUPCVLab/Ford2019/blob/master/Monocular%20Object%20Height%20Estimation%20Network%20Using%20Deep%20Learning%20and%20Scene%20Geometry/test.py) with those.

# Homography estimation
We provide homography estimation code. You need to pick up one image clearly having cone markers on the ground and save it into `homography estimation` folder. Then run [`homo.ipynb`](https://github.com/OSUPCVLab/Ford2019/blob/master/Monocular%20Object%20Height%20Estimation%20Network%20Using%20Deep%20Learning%20and%20Scene%20Geometry/homography%20estimation/homo.ipynb). It outputs H matrix. Then update H matrix within [`utils.py`](https://github.com/OSUPCVLab/Ford2019/blob/master/Monocular%20Object%20Height%20Estimation%20Network%20Using%20Deep%20Learning%20and%20Scene%20Geometry/utils_general/utils.py) `get_H_A0_roi` function. If you had a predefined roi, please also update it in this function.

# Disclaimer
Feel free to open an issue if you get stuck anywhere.
