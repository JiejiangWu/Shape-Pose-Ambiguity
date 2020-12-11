import os

import soft_renderer.functional as nr
import torch
import numpy as np
import tqdm
import h5py
import random
from torch.utils.data import Dataset
class_ids_map = {
    '02691156': 'Airplane',
    '02828884': 'Bench',
    '02933112': 'Cabinet',
    '02958343': 'Car',
    '03001627': 'Chair',
    '03211117': 'Display',
    '03636649': 'Lamp',
    '03691459': 'Loudspeaker',
    '04090263': 'Rifle',
    '04256520': 'Sofa',
    '04379243': 'Table',
    '04401088': 'Telephone',
    '04530566': 'Watercraft',
}

class CUB_Dataset(object):
    def __init__(self, data_folder=None):
        self.h5file = data_folder
        self.h = h5py.File(data_folder,'r')
        self.imgs = self.h['images'][:]
        self.h.close()
    def get_random_single_batch(self,batch_size):
        idx = np.random.choice(range(len(self.imgs)), batch_size, replace=False)
        imgs = torch.Tensor(self.imgs[idx,:,:,:] / 255.)
        backpixelidx = (imgs[:,3,:,:] < 0.2).unsqueeze(1).repeat(1,4,1,1)
        imgs[backpixelidx]=0
        
        views = torch.zeros(batch_size,3)
        viewpoints = torch.zeros(batch_size,3)
        return imgs,viewpoints,views
    
    def get_specific_img(self,idx):
        return torch.from_numpy(self.imgs[idx].astype('float32')/255.), 0
    def get_dataset_len(self):
        return self.imgs.shape[0]
    
class ShapeNet(object):
    def __init__(self, directory=None, class_ids=None, set_name=None,view_range=None,view_noise = 0):
        self.class_ids = class_ids
        self.set_name = set_name
        self.elevation = 30.
        self.distance = 2.732
        self.view_range = view_range
        self.view_noise = view_noise
        self.class_ids_map = class_ids_map

        images = []
        voxels = []
        self.num_data = {}
        self.pos = {}
        count = 0
        loop = tqdm.tqdm(self.class_ids)
        loop.set_description('Loading dataset')
        for class_id in loop:
            images.append(list(np.load(
                os.path.join(directory, '%s_%s_images.npz' % (class_id, set_name))).items())[0][1])
            voxels.append(list(np.load(
                os.path.join(directory, '%s_%s_voxels.npz' % (class_id, set_name))).items())[0][1])
            self.num_data[class_id] = images[-1].shape[0]
            self.pos[class_id] = count
            count += self.num_data[class_id]
        images = np.concatenate(images, axis=0).reshape((-1, 4, 64, 64))
        images = np.ascontiguousarray(images)
        self.images = images
        self.voxels = np.ascontiguousarray(np.concatenate(voxels, axis=0))
        del images
        del voxels

    @property
    def class_ids_pair(self):
        class_names = [self.class_ids_map[i] for i in self.class_ids]
        return zip(self.class_ids, class_names)

    def get_specific_img(self,idx):
        return torch.from_numpy(self.images[idx].astype('float32')/255.), -(idx %24)*15+90
    
    def get_specific_vox(self,idx):
        return torch.from_numpy(self.images[idx].astype('float32')/255.), -(idx %24)*15+90, torch.from_numpy(self.voxels[idx//24].astype('float32'))
    def get_dataset_len(self):
        return self.images.shape[0]

    def get_random_pair_batch(self, batch_size):
        data_ids_a = np.zeros(batch_size, 'int32')
        data_ids_b = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = torch.zeros(batch_size)
        viewpoint_ids_b = torch.zeros(batch_size)
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.randint(0, self.num_data[class_id])

            viewpoint_id_a = np.random.randint(0, 24)
            viewpoint_id_b = np.random.randint(0, 24)
            data_id_a = (object_id + self.pos[class_id]) * 24 + viewpoint_id_a
            data_id_b = (object_id + self.pos[class_id]) * 24 + viewpoint_id_b
            data_ids_a[i] = data_id_a
            data_ids_b[i] = data_id_b
            viewpoint_ids_a[i] = viewpoint_id_a
            viewpoint_ids_b[i] = viewpoint_id_b

        images_a = torch.from_numpy(self.images[data_ids_a].astype('float32') / 255.)
        images_b = torch.from_numpy(self.images[data_ids_b].astype('float32') / 255.)

        distances = torch.ones(batch_size).float() * self.distance
        elevations_a = torch.ones(batch_size).float() * self.elevation
        elevations_b = torch.ones(batch_size).float() * self.elevation
        viewpoints_a = nr.get_points_from_angles(distances, elevations_a, -viewpoint_ids_a * 15 + 90)
        viewpoints_b = nr.get_points_from_angles(distances, elevations_b, -viewpoint_ids_b * 15 + 90)
        views_a = torch.cat([distances.unsqueeze(0),elevations_a.unsqueeze(0),-viewpoint_ids_a.unsqueeze(0) * 15 + 90])
        views_b = torch.cat([distances.unsqueeze(0),elevations_b.unsqueeze(0),-viewpoint_ids_b.unsqueeze(0) * 15 + 90])
        return images_a, images_b, viewpoints_a, viewpoints_b, views_a, views_b
    
    def get_random_single_batch(self, batch_size):
        data_ids_a = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = torch.zeros(batch_size)
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.randint(0, self.num_data[class_id])

            if self.view_range == None:
                viewpoint_id_a = np.random.randint(0,24)
            else:
                viewpoint_id_a = np.random.choice(self.view_range)
            viewpoint_id_a = np.random.randint(0, 24)
            data_id_a = (object_id + self.pos[class_id]) * 24 + viewpoint_id_a
            data_ids_a[i] = data_id_a
            viewpoint_ids_a[i] = viewpoint_id_a

        images_a = torch.from_numpy(self.images[data_ids_a].astype('float32') / 255.)

        distances = torch.ones(batch_size).float() * self.distance
        elevations_a = torch.ones(batch_size).float() * self.elevation
        noise = (torch.rand(batch_size)-0.5) * 2 * self.view_noise
        viewpoints_a = nr.get_points_from_angles(distances, elevations_a, -viewpoint_ids_a * 15 + 90 + noise)
        views_a = torch.cat([distances.unsqueeze(0),elevations_a.unsqueeze(0),-viewpoint_ids_a.unsqueeze(0) * 15 + 90 + noise])
        return images_a, viewpoints_a, views_a
    
    
    def get_specific_view_batch(self, batch_size,view_id):
        data_ids_a = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = torch.zeros(batch_size)
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.randint(0, self.num_data[class_id])

            viewpoint_id_a = view_id
            data_id_a = (object_id + self.pos[class_id]) * 24 + viewpoint_id_a
            data_ids_a[i] = data_id_a
            viewpoint_ids_a[i] = viewpoint_id_a
        images_a = torch.from_numpy(self.images[data_ids_a].astype('float32') / 255.)

        distances = torch.ones(batch_size).float() * self.distance
        elevations_a = torch.ones(batch_size).float() * self.elevation
        viewpoints_a = nr.get_points_from_angles(distances, elevations_a, -viewpoint_ids_a * 15 + 90)
        views_a = torch.cat([distances.unsqueeze(0),elevations_a.unsqueeze(0),-viewpoint_ids_a.unsqueeze(0) * 15 + 90])
        return images_a, viewpoints_a,views_a

    def get_all_batches_for_evaluation(self, batch_size, class_id):
        data_ids = np.arange(self.num_data[class_id]) + self.pos[class_id]
        viewpoint_ids = np.tile(np.arange(24), data_ids.size)
        data_ids = np.repeat(data_ids, 24) * 24 + viewpoint_ids

        distances = torch.ones(data_ids.size).float() * self.distance
        elevations = torch.ones(data_ids.size).float() * self.elevation
        viewpoints_all = nr.get_points_from_angles(distances, elevations, -torch.from_numpy(viewpoint_ids).float() * 15)

        for i in range((data_ids.size - 1) // batch_size + 1):
            images = torch.from_numpy(self.images[data_ids[i * batch_size:(i + 1) * batch_size]].astype('float32') / 255.)
            voxels = torch.from_numpy(self.voxels[data_ids[i * batch_size:(i + 1) * batch_size] // 24].astype('float32'))
            yield images, voxels
            
class rgba_Shape_Dataset(Dataset):
    def __init__(self, data_folder, idx_file,transform=None):
        self.h = h5py.File(data_folder, 'r')
        self.ips = self.h.attrs['images_per_shape']
        self.imgs = self.h['images']
        self.viewpoints = self.h['viewpoints']
#        self.shape_ids = self.h['shape_ids']
        self.transform = transform
        with open(idx_file, 'r') as myfile:
            self.shape_idx = myfile.read().splitlines()
        
        
        # Total number of datapoints
        self.dataset_size = len(self.imgs)
        
    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i] / 255.)
        viewpoint = torch.LongTensor(self.viewpoints[i])
        shape_idx = torch.LongTensor([i//self.ips])
        shape_id = self.shape_idx[i//self.ips]
        return img, viewpoint, shape_idx,shape_id
    
    def __len__(self):
        return self.dataset_size
    
   
class randsub_rgbaPairDataset(Dataset):
    def __init__(self, data_folder, rand_num_ips = 24 ,transform=None):
        self.h5fil = data_folder
        self.h = h5py.File(data_folder, 'r')
        self.ips = self.h.attrs['images_per_shape']
        self.rand_num_ips = rand_num_ips
        self.imgs = self.h['images']
        self.viewpoints = self.h['viewpoints']
        
        '''generate random sub'''
        self.shape_num = int(len(self.imgs) / self.ips)
        self.sub_idx = np.zeros([self.shape_num,self.rand_num_ips])
        for i in range(self.shape_num):
            temp_sub_idx = np.random.choice(range(0,self.ips),rand_num_ips,replace = False)
            self.sub_idx[i,:] = temp_sub_idx
            
#        self.shape_ids = self.h['shape_ids']
        self.transform = transform
        
        
        # Total number of datapoints
        self.dataset_size = len(self.imgs)
        

    def __getitem__(self, i):
#        print(111)
        with h5py.File(self.h5fil, 'r') as db:
            idx = random.sample(range(0,self.rand_num_ips),2)
            idx0 = int(i*self.ips + self.sub_idx[i,idx[0]])
            idx1 = int(i*self.ips + self.sub_idx[i,idx[1]])
            img0 = torch.FloatTensor(db['images'][idx0] / 255.)
            img1 = torch.FloatTensor(db['images'][idx1] / 255.)
            viewpoint0 = torch.LongTensor(db['viewpoints'][idx0])
            viewpoint1 = torch.LongTensor(db['viewpoints'][idx1])
        return img0,img1, viewpoint0,viewpoint1
    
    def __len__(self):
        return self.shape_num
    
class view_Dataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.h5file = data_folder
        self.h = h5py.File(data_folder,'r')
        self.imgs = self.h['images'][:]
        self.viewpoints = self.h['viewpoints']
        
        self.transform = transform
        self.dataset_size = len(self.imgs)
        
    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i])
        viewpoint = torch.LongTensor(self.viewpoints[i])
        return img,viewpoint
    def __len__(self):
        return self.dataset_size

