from src.models import models
from src.losses import loss
from src.dataset import datasets
import soft_renderer.functional as srf
import yaml
import argparse,os
from src.training import training
import torch.optim as optim
import torch
from scipy import misc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--cfg_file', type=str,default = './configs/shapenet/shapenet_1c_02691156.yaml')
    parser.add_argument('--checkpoint', type=str,default = './data/models/02691156-R.pth.tar')
    parser.add_argument('--init_obj', type=str,default = './data/sphere_642.obj')
    parser.add_argument('--input_path', type=str,default = './data/demo/')

    args = parser.parse_args()
    
    # load cfg
    f = open(args.cfg_file)
    cfg = yaml.load(f)
    f.close()
    
    # make model
    model_R = models.Reconstructor(args.init_obj,cfg).to(device)
    checkpoint = torch.load(args.checkpoint)
    model_R.load_state_dict(checkpoint['model_R'])
    model_R.eval()
    img_files = os.listdir(args.input_path)

    for i in range(len(img_files)):
        print(str(i) + '/' + str(len(img_files)))
        temp_img_path = os.path.join(args.input_path , img_files[i])
        filename, file_extension = os.path.splitext(img_files[i])
        
        if not (file_extension == '.png' or file_extension == '.jpg'):
            continue
        temp_img = misc.imread(temp_img_path)
        temp_img = temp_img.transpose(2,0,1)
        img = torch.FloatTensor(temp_img.astype('float32') / 255.).unsqueeze(0).to(device)
        
    
        vertices, faces,_ = model_R.reconstruct(img)
        srf.save_obj(os.path.join(args.input_path,filename+'.obj'),vertices[0],faces[0])
main()
