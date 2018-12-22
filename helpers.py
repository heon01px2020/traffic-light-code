import math
import numpy
import torch
def angle_difference(pred, label):
    #pred is a list in the form [x1, y1, x2, y2]
    #label is a list in the form [x1, y1, x2, y2]

    
    pred_x1 = pred[0][0]
    pred_x2 = pred[2][0]
    pred_y1 = pred[1][0]
    pred_y2 = pred[3][0]
    pred_x_distance = pred_x1 - pred_x2
    pred_y_distance = pred_y2 - pred_y1
    #if pred_x_distance is positive then the direction is pointing towards the right
    #if pred_x_distance is negative then the direction is pointing towards the left
    
    pred_angle = math.atan2(pred_y_distance, pred_x_distance)
    act_x1 = label[0][0]
    act_x2 = label[2][0]
    act_y1 = label[1][0]
    act_y2 = label[3][0]
    act_x_distance = act_x1 - act_x2
    act_y_distance = act_y2 - act_y1
    actual_angle = math.atan2(act_y_distance, act_x_distance)
    return (pred_angle - actual_angle)*180/math.pi
    

def startpoint_difference(pred, label):
    #pred is a list in the form [x1, y1, x2, y2]
    #label is a list in the form [x1, y1, x2, y2]

        
    x_distance = pred[2][0] - label[2][0]
    y_distance = pred[3][0] - label[3][0]
    distance = math.sqrt(x_distance*x_distance + y_distance*y_distance)
    return distance

def endpoint_difference(pred, label):
    #pred is a list in the form [x1, y1, x2, y2]
    #label is a list in the form [x1, y1, x2, y2]


    x_distance = pred[0][0] - label[0][0]
    y_distance = pred[1][0] - label[1][0]
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
    return 0.7*angle+ 0.2*start +0.1*end

