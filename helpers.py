import math
import numpy
import torch
def angle_difference(pred, label):
    
    pred_x1 = pred[0][0]
    pred_x2 = pred[0][2]
    pred_y1 = pred[0][1]
    pred_y2 = pred[0][3]
    pred_x_distance = pred_x1 - pred_x2
    pred_y_distance = pred_y2 - pred_y1

    pred_angle = math.atan2(pred_y_distance, pred_x_distance)
    act_x1 = label[0][0]
    act_x2 = label[0][2]
    act_y1 = label[0][1]
    act_y2 = label[0][3]
    act_x_distance = act_x1 - act_x2
    act_y_distance = act_y2 - act_y1
    actual_angle = math.atan2(act_y_distance, act_x_distance)
    return (pred_angle - actual_angle)*180/math.pi
    

def startpoint_difference(pred, label):
        
    x_distance = pred[0][2] - label[0][2]
    y_distance = pred[0][3] - label[0][3]
    distance = math.sqrt(x_distance*x_distance + y_distance*y_distance)
    return distance

def endpoint_difference(pred, label):

    x_distance = pred[0][0] - label[0][0]
    y_distance = pred[0][1] - label[0][1]
    distance = math.sqrt(x_distance*x_distance + y_distance*y_distance)
    return distance

def direction_performance(pred, label):
    #alpha and beta need to be adjusted
    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    pred = pred.tolist()
    label = label.tolist()
    angle = math.fabs(angle_difference(pred,label))
    start = math.fabs(startpoint_difference(pred,label))
    end = math.fabs(endpoint_difference(pred,label))
    return angle, start, end

