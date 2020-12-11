import soft_renderer as sr
import soft_renderer.functional as srf
from src.models import models
from src.losses import loss
from src.dataset import datasets
from src.training import training
from src.utils import geo_utils,utils
import torch.optim as optim
import torch
from PIL import Image
import numpy as np
import yaml
import matplotlib.pyplot as plt
import scipy.misc
import os
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
init_obj = './data/sphere_642.obj'

def eval_iou(predicted_v,predicted_f,voxels):
    faces_ = srf.face_vertices(predicted_v, predicted_f).data
    faces_norm = faces_ * 1. * (32. - 1) / 32. + 0.5
    voxels_predict = srf.voxelization(faces_norm, 32, False).cpu().numpy()
    voxels_predict = voxels_predict.transpose(0, 2, 1, 3)[:, :, :, ::-1]
    iou = (voxels * voxels_predict).sum((1, 2, 3)) / (0 < (voxels + voxels_predict)).sum((1, 2, 3))
    return iou

def search_iou_with_rotate(v,f,voxels):
    best_iou = 0
    for azimuth in range(0,360,5):
        v2 = geo_utils.get_rotate_points_from_angles(torch.Tensor([0]).to(device),torch.Tensor([azimuth]).to(device),v) 
        temp_iou = eval_iou(v2,f,voxels)
        if temp_iou[0]>best_iou:
            best_iou = temp_iou[0]
    return best_iou


def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--cfg_file', type=str,default = './configs/shapenet/shapenet_1c_02691156.yaml')
    parser.add_argument('--checkpoint_dir', type=str,default = './data/models/')
    parser.add_argument('--category',type=str,default = '02691156')
    parser.add_argument('--RorE', type=str,default = 'R')
    args = parser.parse_args()

    cfg_file = args.cfg_file
    f = open(cfg_file)
    cfg = yaml.load(f)
    f.close()
    ious = utils.AverageMeter()

    if args.RorE == 'R':
        R_checkpoint_dir = args.checkpoint_dir + args.category + '-R.pth.tar'
    else:
        E_checkpoint_dir = args.checkpoint_dir + args.category + '-E.pth.tar'
        G_checkpoint_dir = args.checkpoint_dir + args.category + '-G.pth.tar'


    
    '''load model'''
    if args.RorE == 'R':
        model_R = models.Reconstructor(init_obj,cfg).to(device)
        R_checkpoint = torch.load(R_checkpoint_dir)
        model_R.load_state_dict(R_checkpoint['model_R'])
        model_R.eval()

    else:
        model_E = models.IEncoder(cfg).to(device)
        E_checkpoint = torch.load(E_checkpoint_dir)
        model_E.load_state_dict(E_checkpoint['model_E'])
        model_E.eval()
        model_G = models.Unconditional_Generator(init_obj,cfg).to(device)
        G_checkpoint = torch.load(G_checkpoint_dir)
        model_G.load_state_dict(G_checkpoint['model_G'])
        model_G.eval()
    
    '''result file'''
    if not os.path.exists('./data/result'):
        os.makedirs('./data/result')
    result_file = open(os.path.join('./data/result',cfg['eid']+'.txt'),"w")
    for class_id in cfg['category']:
        dataset_val = datasets.ShapeNet('./data/dataset',[class_id], 'test')
        
        for i in range(dataset_val.get_dataset_len()):
            im,view,vx = dataset_val.get_specific_vox(i)
            
            '''align shape******************'''
            view = view-90
            im = im.unsqueeze(0)
            images = torch.autograd.Variable(im).cuda()
            voxels = vx.numpy()   
            '''reconstruct'''
            if args.RorE == 'R':
                v,f,c = model_R.reconstruct(images)
            else:
                image_feature = model_E(images[:,0:3,:,:])
                v, f = model_G(image_feature)
            
            
            # for view-aligned model (B/L-s, VPL)
#            v2 = geo_utils.get_rotate_points_from_angles(torch.Tensor([0]).to(device),torch.Tensor([90]).to(device),v)            
#            tempiou = eval_iou(v2,f,voxels)
#            ious.update(tempiou[0],1)
            
            # for view-unaligned model (B/L-u, ours)
            iou = search_iou_with_rotate(v,f,voxels)
            ious.update(iou,1)
            
            print('{0}/{1}\t iou:{iou.val:.3f} ({iou.avg:.3f})\t'.format(i,dataset_val.get_dataset_len(),
                                                                              iou = ious))
        result_file.write(class_id+':'+str(ious.avg))
        result_file.write('\r\n')
    result_file.close()
main()
