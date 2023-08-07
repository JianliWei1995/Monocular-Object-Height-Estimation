## This use homography estimation matrix
from model.yolov5.models.experimental import attempt_load
from model.yolov5.utils.datasets import LoadImages, LoadStreams
from model.yolov5.utils.datasets import LoadStreams, LoadImages
from model.yolov5.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from model.yolov5.utils.plots import plot_one_box
from model.yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
import torch
import torchvision
import os
import numpy as np
import cv2
import argparse
import time
import matplotlib.pyplot as plt
from datetime import datetime

from model import HE
from utils_general.utils import *

import sys
sys.path.insert(0, './model/yolov5')


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def main(opt):

    # set up dirs
    path_result = os.path.join('results', opt.seq_num)
    os.makedirs(path_result, exist_ok=True)

    # Predefine iPhone 8p camera intrinsic matrix and distortion coefficients
    # please replace them with your camera parameters
    Proj_matrix = np.array([[1.18414502e+03, 0.00000000e+00, 6.38458469e+02],
                            [0.00000000e+00, 1.18404440e+03, 3.52082609e+02],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    invProjmatrix = np.asmatrix(Proj_matrix).I
    dist_coeffs = np.array([[3.15766976e-01, -2.21015226e+00, -1.52525537e-03, 3.42399952e-05, 7.63710136e+00]])
    
    # predefined homography matrix and road coordinates origion A0
    H, A0, roi_pts, roi_mask = get_H_A0_roi(opt.seq_num)

    # initialize object detector
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load ODNet
    ODNet = attempt_load('./model/yolov5/weights/'+opt.weights, map_location=device)  # load FP32 model
    ODNet.to(device).eval()
    imgsz = check_img_size(opt.img_size, s=ODNet.stride.max())  # check img_size
    if half:
        ODNet.half()  # to FP16

    # Get names and colors
    names = ODNet.module.names if hasattr(ODNet, 'module') else ODNet.names
    # detect over 80 object classes
    target_classes = np.arange(80).tolist()
    COLORS = [[255,0,0]]*80
    
    ## Only detect some specific objects
    # target_classes = [1, 3, 8] # person-1, car-3, truck-8
    # COLORS = []
    # for class_ in target_classes:
        # COLORS.append(compute_color_for_labels(class_+1))
    

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = ODNet(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    # initilize object height estimater
    HENet = HE.HENet_yolov5_undist(detector, Proj_matrix, dist_coeffs, invProjmatrix, H, A0, roi_mask)
    HENet.to(device=device)
    HENet.eval()
    
    # Set Dataloader
    img_dir = opt.source+opt.seq_num
    dataset = LoadImages(img_dir, img_size=imgsz)

    for frame, (path, img, img0, vid_cap) in enumerate(dataset):
        # counting process time
        t0 = time.time()
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = ODNet(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=target_classes, agnostic=opt.agnostic_nms)[0]
        t1  =time.time()

        if len(pred):
            # Rescale boxes from img_size to img0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape)

            # apply HE
            obj_heights = HENet(pred)

            # plot boundingbox and estimated height on raw image
            for i in range(len(pred)):
                height = obj_heights[i]
                x1, y1, x2, y2, _, cls = pred[i].round()
                x1, y1, x2, y2, cls = x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist(), int(cls.tolist())
                
                # label = '{} {}'.format(names[cls], height)
                label = '{}'.format(height)
                plot_obj_line([x1,y1,x2,y2], img0, label=label, color=COLORS[cls], line_thickness=2) if height is not None else None
        # plot roi
        pts = roi_pts.reshape((-1, 1, 2))
        cv2.polylines(img0, [pts[1:-1,:]], False, (0, 0, 255),2)
        cv2.imwrite(os.path.join(path_result, '%06d.png' % frame), img0)

        t2 = time.time()
            
        # update loop info
        print('OD Inference Time: %.3f' %(t1 - t0))
        print('HE Inference Time: %.3f' %(t2 - t1))
        print('=======================')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./datasets/', help='source')  # file/folder
    parser.add_argument('--seq_num', type=str, default='00', help='Inference folder within source')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    
    detector = 'yolov5'
    print('========== %s ===========' %detector)

    main(opt)