from src.models import models
from src.losses import loss
from src.dataset import datasets
import yaml
import argparse,os
from src.training import training
import torch.optim as optim
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--cfg_file', type=str,default = './configs/shapenet/shapenet_1c_02691156.yaml')
    parser.add_argument('--init_obj', type=str,default = './data/sphere_642.obj')
    args = parser.parse_args()
    
    # load cfg
    f = open(args.cfg_file)
    cfg = yaml.load(f)
    f.close()
    
    if cfg['datasource'] == 'shapenet':
        load_dir = os.path.join('./data/checkpoint/shapenet',cfg['eid'])
        save_dir = os.path.join('./data/checkpoint/shapenet',cfg['eid'])
        
    elif cfg['eid'] == 'CUB_64_color':
        load_dir = './data/checkpoint/CUB_64_color'
        save_dir = './data/checkpoint/CUB_64_color'
        
    elif cfg['eid'] == 'CUB_64':
        load_dir = './data/checkpoint/CUB_64'
        save_dir = './data/checkpoint/CUB_64'
    

    if cfg['datasource'] == 'shapenet':
        DATASET_DIRECTORY = './data/dataset'  
        if 'azimuth' in cfg:
            dataset_train = datasets.ShapeNet(DATASET_DIRECTORY, cfg['category'], 'train',view_range=cfg['azimuth'])
        else:    
            dataset_train = datasets.ShapeNet(DATASET_DIRECTORY, cfg['category'], 'train')
    elif cfg['datasource'] == 'CUB':
        dataset_train = datasets.CUB_Dataset('./data/dataset/CUB_200.h5py')
    elif cfg['datasource'] == 'CUB_64':
        dataset_train = datasets.CUB_Dataset('./data/dataset/CUB_200_64.h5py')
    

    # train normal GAN
    NG_load_dir = os.path.join(load_dir,'NormalGan')
    NG_save_dir = os.path.join(save_dir,'NormalGan')
    normal_model_D = models.NormalDiscriminator(cfg['input_size']).to(device)
    normal_model_G = models.Unconditional_Generator(args.init_obj,cfg).to(device)
    
    optimizer_D = optim.Adam(normal_model_D.parameters(),lr=cfg['NormalGan']['D']['lr'])
    optimizer_G = optim.Adam(normal_model_G.parameters(),lr=cfg['NormalGan']['G']['lr'])

    training.train_Normal_GAN(normal_model_D,normal_model_G,
                              optimizer_D,optimizer_G,
                              NG_load_dir,NG_save_dir,
                              cfg,dataset_train)

       
    
    # load trained Normal_G 
    normal_model_G = models.Unconditional_Generator(args.init_obj,cfg).to(device)
    G_load_dir = os.path.join(load_dir,'NormalGan/Temp.pth.tar')
    
    # train view predictor 
    V_load_dir = os.path.join(load_dir,'ViewPrediction')
    V_save_dir = os.path.join(save_dir,'ViewPrediction')
    model_vp = models.view_predictor(cfg).to(device)
    optimizer_vp = optim.Adam(model_vp.parameters(),lr=cfg['ViewPrediction']['lr'])
    training.train_view_predictor(normal_model_G,G_load_dir,model_vp,optimizer_vp,V_load_dir,V_save_dir,cfg)
 
    
    # load view prediction
    model_vp = models.view_predictor(cfg).to(device)
    vp_load_dir = os.path.join(load_dir,'ViewPrediction/Temp.pth.tar')


    # train image encoder
#    E_load_dir = os.path.join(load_dir,'ImageEncoder')
#    E_save_dir = os.path.join(save_dir,'ImageEncoder')
#    model_E = models.IEncoder(cfg).to(device)
#    optimizer_E = optim.Adam(model_E.parameters(),lr=cfg['Reconstruction']['lr'])
#    training.train_imageEncoder(normal_model_G,G_load_dir,model_vp,vp_load_dir,model_E,optimizer_E,E_load_dir,E_save_dir,cfg,dataset_train)
#    
#    # train reconstructor
    R_load_dir = os.path.join(load_dir,'Reconstruction')
    R_save_dir = os.path.join(load_dir,'Reconstruction')
    model_R = models.Reconstructor(args.init_obj,cfg).to(device)
    model_D = models.NormalDiscriminator(cfg['input_size']).to(device)
    optimizer_R = optim.Adam(model_R.parameters(),lr=cfg['Reconstruction']['lr'])
    optimizer_D = optim.Adam(model_D.parameters(),lr=cfg['Reconstruction']['lr'])
    training.train_reconstructor(model_vp,vp_load_dir,model_R,optimizer_R,model_D,optimizer_D,R_load_dir,R_save_dir,cfg,dataset_train)
main()
