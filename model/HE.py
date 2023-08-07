import torch
import numpy as np
import cv2
from torch.nn import Module
from skimage import io
from skimage import feature
import matplotlib.pyplot as plt

class HENet(Module):
    def __init__(self, detector, invProjmatrix, H, A0, thr_score, target_classes):
        super().__init__()
        self.detector = detector
        self.invProj_matrix = torch.from_numpy(invProjmatrix).float()
        self.H = torch.from_numpy(H).float()
        self.A0 = torch.from_numpy(A0).float()
        self.thr_score = thr_score
        self.target_classes = target_classes
        
        self.calc_scale = calc_scale(self.H, self.A0)
        self.cameraToCamera = cameraToCamera(self.invProj_matrix)

    def forward(self, predictions):
        boxes = predictions[0]['boxes']
        classIDs = predictions[0]['labels']
        scores = predictions[0]['scores']
        masks = predictions[0]['masks']  if self.detector=='mask_rcnn' else None
        Height = {}
        
        for i in range(len(boxes)):
            if scores[i] > self.thr_score:
                if classIDs[i] in self.target_classes: # person, car, truck
                    # extract the bounding box coordinates
                    (x1, y1) = (boxes[i][0], boxes[i][1])
                    (x2, y2) = (boxes[i][2], boxes[i][3])    
                    
                    # generate object head and feet position
                    p_foot = (torch.round((x1 + x2)/2), torch.round(y2))
                    
                    # calculate scale or depth via foot point
                    scale = self.calc_scale(p_foot)
                    
                    #back-project foot point
                    world_foot = self.cameraToCamera(p_foot, scale)
                    
                    if self.detector=='mask_rcnn':
                        top_edge = find_top_edge(masks[i][0].detach().numpy(), p_foot)
                        world_top_edge = []
                        for edge_pt in top_edge:
                            world_edge_pt = self.cameraToCamera(edge_pt, scale).detach().numpy()
                            world_top_edge.append([world_edge_pt[0,0], world_edge_pt[1,0]])
                        dist = []
                        for world_edge_pt in world_top_edge:
                            dist.append(abs(world_edge_pt[0]-world_foot[0,0].detach().numpy()))
                        min_dist_idx = dist.index(min(dist))
                        y_head = top_edge[min_dist_idx]
                        p_head = torch.Tensor(y_head)
                    else:
                        p_head = (torch.round((x1 + x2)/2), torch.round(y1))
                        
                    # estimate object height
                    world_head = self.cameraToCamera(p_head, scale)
                    height = world_foot[1,0]-world_head[1,0]
                    if self.detector=='mask_rcnn':
                        Height[i] = [round(float(height.detach().numpy()),2), (int(p_foot[0].detach().numpy()), int(p_foot[1].detach().numpy())), (y_head[0], y_head[1])]
                    else:
                        Height[i] = round(float(height.detach().numpy()),2)

        return Height
        
class HENet_undist(Module):
    def __init__(self, detector, camera_matrix, dist_coeffs, invProjmatrix, H, A0, thr_score, target_classes, roi):
        super().__init__()
        self.detector = detector
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.invProj_matrix = torch.from_numpy(invProjmatrix).float()
        self.H = torch.from_numpy(H).float()
        self.A0 = torch.from_numpy(A0).float()
        self.thr_score = thr_score
        self.target_classes = target_classes
        
        self.calc_scale = calc_scale(self.H, self.A0)
        self.cameraToCamera = cameraToCamera(self.invProj_matrix)
        self.roi = roi

    def forward(self, predictions):
        boxes = predictions[0]['boxes']
        classIDs = predictions[0]['labels']
        scores = predictions[0]['scores']
        masks = predictions[0]['masks']  if self.detector=='mask_rcnn' else None
        Height = {}
        
        for i in range(len(boxes)):
            if scores[i] > self.thr_score:
                if classIDs[i] in self.target_classes: # person, car, truck
                    # extract the bounding box coordinates
                    (x1, y1) = (boxes[i][0], boxes[i][1])
                    (x2, y2) = (boxes[i][2], boxes[i][3])    
                    
                    # generate object head and feet position
                    # p_foot = (torch.round((x1 + x2)/2), torch.round(y2))
                    p_foot = ((x1 + x2)/2, y2)
                    xy = np.array([p_foot[0].tolist(), p_foot[1].tolist()], dtype=np.float32).reshape(1,1,2)
                    xy_undistorted = cv2.undistortPoints(xy, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
                    p_foot = xy_undistorted.reshape(2)
                    if self.roi[int(p_foot[1]), int(p_foot[0])]==1:
                        p_foot = torch.from_numpy(p_foot)
                    
                        # calculate scale or depth via foot point
                        scale = self.calc_scale(p_foot)
                        
                        #back-project foot point
                        world_foot = self.cameraToCamera(p_foot, scale)
                        
                        if self.detector=='mask_rcnn':
                            top_edge = find_top_edge(masks[i][0].detach().numpy(), p_foot)
                            world_top_edge = []
                            for edge_pt in top_edge:
                                world_edge_pt = self.cameraToCamera(edge_pt, scale).detach().numpy()
                                world_top_edge.append([world_edge_pt[0,0], world_edge_pt[1,0]])
                            dist = []
                            for world_edge_pt in world_top_edge:
                                dist.append(abs(world_edge_pt[0]-world_foot[0,0].detach().numpy()))
                            min_dist_idx = dist.index(min(dist))
                            y_head = top_edge[min_dist_idx]
                            p_head = torch.Tensor(y_head)
                        else:
                            # p_head = (torch.round((x1 + x2)/2), torch.round(y1))
                            p_head = ((x1 + x2)/2, y1)
                        
                        # undistortPoints
                        xy = np.array([p_head[0].tolist(), p_head[1].tolist()], dtype=np.float32).reshape(1,1,2)
                        xy_undistorted = cv2.undistortPoints(xy, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
                        p_head = torch.from_numpy(xy_undistorted.reshape(2))
                            
                        # estimate object height
                        world_head = self.cameraToCamera(p_head, scale)
                        height = world_foot[1,0]-world_head[1,0]
                        if self.detector=='mask_rcnn':
                            Height[i] = [round(float(height.detach().numpy()),2), (int(p_foot[0].detach().numpy()), int(p_foot[1].detach().numpy())), (y_head[0], y_head[1])]
                        else:
                            Height[i] = round(float(height.detach().numpy()),2)
                    else:
                        Height[i] = None

        return Height

class HENet_yolov5_undist(Module):
    def __init__(self, detector, camera_matrix, dist_coeffs, invProjmatrix, H, A0, roi):
        super().__init__()
        self.detector = detector
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.invProj_matrix = torch.from_numpy(invProjmatrix).float()
        self.H = torch.from_numpy(H).float()
        self.A0 = torch.from_numpy(A0).float()
        self.roi = roi
        
        self.calc_scale = calc_scale(self.H, self.A0)
        self.cameraToCamera = cameraToCamera(self.invProj_matrix)

    def forward(self, pred):
        Height = []
        
        for x1, y1, x2, y2, _, _ in pred:

            # generate object head and feet position
            p_foot = ((x1 + x2)/2, y2)
            xy = np.array([p_foot[0].tolist(), p_foot[1].tolist()], dtype=np.float32).reshape(1,1,2)
            xy_undistorted = cv2.undistortPoints(xy, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
            p_foot = xy_undistorted.reshape(2)
            if self.roi[int(p_foot[1]), int(p_foot[0])]==1:
                p_foot = torch.from_numpy(p_foot)
                
                # calculate scale or depth via foot point
                scale = self.calc_scale(p_foot)
                
                #back-project foot point
                world_foot = self.cameraToCamera(p_foot, scale)

                p_head = ((x1 + x2)/2, y1)
                
                # undistortPoints
                xy = np.array([p_head[0].tolist(), p_head[1].tolist()], dtype=np.float32).reshape(1,1,2)
                xy_undistorted = cv2.undistortPoints(xy, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
                p_head = torch.from_numpy(xy_undistorted.reshape(2))
                    
                # estimate object height
                world_head = self.cameraToCamera(p_head, scale)
                height = world_foot[1,0]-world_head[1,0]

                Height.append(round(float(height.detach().numpy()),2))
            else:
                Height.append(None)

        return Height

# Calculate scale
class calc_scale(Module):
    def __init__(self, H, A0):
        super().__init__()
        self.H = H
        self.A0 = A0

    def forward(self, point):
        x = torch.zeros(3,1)
        x[0][0] = point[0]
        x[1][0] = point[1]
        x[2][0] = 1.0
        
        x = (self.H).mm(x)
        x = x/x[-1]
        scale = x[1]+self.A0[1]
        return scale

# Project image pixel coordinates back to camera coordinates
class cameraToCamera(Module):
    def __init__(self, invProj_matrix):
        super().__init__()
        self.invProj_matrix = invProj_matrix
        
    def forward(self, point, scale):
        x = torch.zeros(3,1)
        x[0][0] = point[0]
        x[1][0] = point[1]
        x[2][0] = 1.0
        
        worldpt = scale*(self.invProj_matrix.mm(x))
        return worldpt[0:-1,:] # return 3*1 matrix

def find_top_edge(mask, p_foot):
    edge = feature.canny(mask)
    idx = np.where(edge==True)
    l1 = sorted(set(idx[1].tolist()),key=idx[1].tolist().index)
    top_edge = []
    for i in l1:
        if abs(i-p_foot[0])<20:
            xx = np.where(idx[1]==i)[0]
            min_y = min([idx[0][j] for j in xx])
            top_edge.append([i,min_y])
    return top_edge