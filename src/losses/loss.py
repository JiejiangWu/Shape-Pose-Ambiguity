import torch.nn as nn
import torch

PI = 3.1415926
def angle_loss(predicted_angles, target_angles):
    radians1 = predicted_angles*PI/180
    radians2 = target_angles*PI/180
    diff_angle = torch.abs(torch.atan2(torch.sin(radians1-radians2),torch.cos(radians1-radians2)))
    return diff_angle.mean()

MSE = torch.nn.MSELoss(reduction='none')
def iou(predict, target, eps=1e-6):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + eps
    return (intersect / union).sum() / intersect.nelement()

def iou_loss(predict, target):
    return 1 - iou(predict, target)


def reprojection_loss(predicts, targets):
    IoU_loss = iou_loss(predicts[:,3,:,:],targets[:,3,:,:])
    pixel_color_loss = MSE(predicts[:,0:3,:,:],targets[:,0:3,:,:]) * targets[:,3,:,:].unsqueeze(1).repeat(1,3,1,1)
    return IoU_loss.mean(),pixel_color_loss.mean()