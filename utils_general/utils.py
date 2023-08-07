import numpy as np
from scipy import stats
import cv2
from skimage import io
from skimage import feature
import matplotlib.pyplot as plt

coco_80 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

coco_91 = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
def coco91_to_coco80_class(target_classes):  
    # converts 91-index (paper) to 80-index (val2014)
    map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    x = []
    for class_ in target_classes:
        x.append(map.index(class_)) if class_ in map else None
    return x

def undistort(img,K,D,DIM,scale=1,imshow=False):
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if dim1[0]!=DIM[0]:
        img = cv2.resize(img,DIM,interpolation=cv2.INTER_AREA)
    Knew = K.copy()
    if scale:#change fov
        Knew[(0,1), (0,1)] = scale * Knew[(0,1), (0,1)]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if imshow:
        cv2.imshow("undistorted", undistorted_img)
    return undistorted_img

def load_txt(file_dir):
    f = open(file_dir,"r")
    temps = f.readlines()
    lines = []
    for temp in temps:
        lines.append(temp.strip())
    return lines

def get_calibration_cam_to_image(cab_f):
    for line in open(cab_f):
        if 'P_rect_02:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3, 4))
            return cam_to_img

def get_H_A0_roi(seq_num):
    if seq_num=='00':
        # seq 01
        H = np.array([[1.21961126e-02, 1.65809204e-04, -8.29530469e+00],
                      [1.00464148e-03, -6.67761076e-02, 3.53378267e+01],
                      [-4.64163373e-04, 1.30278405e-02, -3.94269947e+00]])
        A0 = np.array([0, 5.1816]) # x right, y forward
        # ROI
        mask = np.zeros((720,1280), dtype="uint8")
        roi_pts = np.array([[   0,  720], [   0,  421], [  82,  402], [ 329,  358],
                            [ 428,  340], [ 480,  332], [ 559,  332], [ 819,  336],
                            [ 869,  347], [ 964,  366], [1197,  414], [1280,  433],
                            [1280,  720]], np.int32)
        cv2.fillPoly(mask,[roi_pts],1)
    elif seq_num=='01':
        # seq 01
        H = np.array([[1.24391002e-02, 5.65712742e-04, -8.42510078e+00],
                      [-9.49488509e-04, -4.15196807e-03, 1.65831516e+01],
                      [-3.12230543e-04, 1.26880943e-02, -3.56463311e+00]])
        A0 = np.array([0,0]) # x right, z forward
        # ROI
        mask = np.zeros((720,1280), dtype="uint8")
        roi_pts = np.array([[   0,  720], [   0,  413], [  94,  393], [ 284,  363],
                            [ 413,  342], [ 459,  334], [ 492,  327], [ 830,  336],
                            [ 882,  346], [ 977,  366], [1101,  393], [1280,  432],
                            [1280,  720]])
        cv2.fillPoly(mask,[roi_pts],1)
    return H, A0, roi_pts, mask

def yolov5_ouput_regularization(output):
    # Regularize YOLOv5 output so that to be the same as faster_rcnn output
    predictions = {}
    boxes = output[:,0:4]
    scores = output[:,4]
    labels = output[:,5]
    predictions['boxes'] = boxes
    predictions['scores'] = scores
    predictions['labels'] = labels.int()
    return [predictions]
 
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def plot_obj_line(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(img, c1, c2, color, thickness=max(tl - 2, 1), lineType=cv2.LINE_AA)
    # Plots object head and foot points and their connection line
    pt_head, pt_foot = (int((x[0]+x[2])/2), int(x[1])), (int((x[0]+x[2])/2), int(x[3]))
    cv2.circle(img, pt_head, tl+1, color, -1)
    cv2.circle(img, pt_foot, tl+1, [0,0,255], -1)
    cv2.line(img, pt_head, pt_foot, color, tl-1)
    # Plots label
    if label:
        tf = max(tl, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c1 = pt_head[0] - int(t_size[0]/2), pt_head[1]
        c2 = pt_head[0] + int(t_size[0]/2), pt_head[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def mask_plot_one_obj(pt_foot, pt_head, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    # cv2.rectangle(img, c1, c2, color, thickness=max(tl - 2, 1), lineType=cv2.LINE_AA)
    # Plots object head and foot points and their connection line
    cv2.circle(img, pt_head, tl+2, color, -1)
    cv2.circle(img, pt_foot, tl+2, color, -1)
    cv2.line(img, pt_head, pt_foot, color, tl)
    # Plots label
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c1 = pt_head[0] - int(t_size[0]/2), pt_head[1]
        c2 = pt_head[0] + int(t_size[0]/2), pt_head[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def plot_obj_rect(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=max(tl - 2, 1), lineType=cv2.LINE_AA)
    
    pt_head = (int((x[0]+x[2])/2), int(x[1]))
    pt_foot = (int((x[0]+x[2])/2), int(x[3]))
    cv2.circle(img, pt_head, tl+4, color, -1)
    cv2.circle(img, pt_foot, tl+4, color, -1)
    # Plots label
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c1 = pt_head[0] - int(t_size[0]/2), pt_head[1]
        c2 = pt_head[0] + int(t_size[0]/2), pt_head[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask > 0.9,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def find_head_pt(mask):
    edges1 = feature.canny(mask)
    edges2 = feature.canny(mask, sigma=3)
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(30, 10),
                                    sharex=True, sharey=True)

    ax1.imshow(mask, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('Intance mask', fontsize=20)

    ax2.imshow(edges1, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title(r'Canny filter, $\sigma=1$', fontsize=20)

    ax3.imshow(edges2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title(r'Canny filter, $\sigma=3$', fontsize=20)

    fig.tight_layout()

    plt.show()