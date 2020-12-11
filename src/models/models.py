import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import soft_renderer as sr
import soft_renderer.functional as srf
import math

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def normalize_imagenet(x):
    ''' Normalize input images according to ImageNet standards.

    Args:
        x (tensor): input images
    '''
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x
   
class NormalDiscriminator(nn.Module):
    def __init__(self, img_size=64):
        super(NormalDiscriminator,self).__init__()

        self.img_size = img_size 
        
        self.convs1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=4, stride=2, padding=1)
        self.convs2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=4, stride=2, padding=1)
        self.convs3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=4, stride=2, padding=1)
        self.convs4 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=4, stride=2, padding=1)
        self.convs5 = nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size=4, stride=2, padding=0)
        self.m = nn.AdaptiveAvgPool2d(1)
#        self.linear_v = nn.Linear(3,32,bias=False)
    def forward(self, imgs):    
        out = imgs.view(-1,1,self.img_size,self.img_size)
        out = F.relu(self.convs1(out))
#        view_feature = F.relu(self.linear_v(view))
#        h = torch.cat((image_feature,view_feature.view(-1,32,1,1).repeat(1,1,32,32)),1)
        out = F.relu(self.convs2(out))
        out = F.relu(self.convs3(out))
        out = F.relu(self.convs4(out))
        out = self.convs5(out)
        out = self.m(out)
        return out
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)



class ResnetEncoder(nn.Module):
    r''' ResNet-18 encoder network for image(rgb) input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = torchvision.models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out
 

class SimpleEncoder(nn.Module):
    def __init__(self, dim_in=3, dim_out=512, dim1=64, dim2=1024, im_size=64):
        super(SimpleEncoder, self).__init__()
        dim_hidden = [dim1, dim1*2, dim1*4, dim2, dim2]

        self.conv1 = nn.Conv2d(dim_in, dim_hidden[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(dim_hidden[0], dim_hidden[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(dim_hidden[1], dim_hidden[2], kernel_size=5, stride=2, padding=2)

        self.bn1 = nn.BatchNorm2d(dim_hidden[0])
        self.bn2 = nn.BatchNorm2d(dim_hidden[1])
        self.bn3 = nn.BatchNorm2d(dim_hidden[2])

        self.fc1 = nn.Linear(dim_hidden[2]*math.ceil(im_size/8)**2, dim_hidden[3])
        self.fc2 = nn.Linear(dim_hidden[3], dim_hidden[4])
        self.fc3 = nn.Linear(dim_hidden[4], dim_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)
        return x
class IEncoder(nn.Module):
    def __init__(self, cfg):
        super(IEncoder, self).__init__()
        feature_dim = cfg['Reconstruction']['f_dim']
        img_size = cfg['input_size']
        if cfg['Reconstruction']['encoder_type'] == 'resnet':
            self.model_E = ResnetEncoder(c_dim = feature_dim)
        elif ['Reconstruction']['encoder_type'] == 'simple':
            self.model_E = SimpleEncoder(dim_out = feature_dim, im_size=img_size)
        self.final_fc = nn.Linear(feature_dim,cfg['NormalGan']['G']['z_dim'])
    def forward(self,imgs):
        return self.final_fc(self.model_E(imgs))
#color_num = 10
class Decoder(nn.Module):
    def __init__(self, filename_obj, color_rec, dim_in=512, centroid_scale=0.1, bias_scale=1.0, centroid_lr=0.1, bias_lr=1.0,color_num=5):
        super(Decoder, self).__init__()
        # load .obj
        self.template_mesh = sr.Mesh.from_obj(filename_obj)
        self.register_buffer('vertices_base', self.template_mesh.vertices.cpu()[0])#vertices_base)
        self.register_buffer('faces', self.template_mesh.faces.cpu()[0])#faces)

        self.nv = self.vertices_base.size(0)
        self.nf = self.faces.size(0)
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 0.5

        dim = 1024
        dim_hidden = [dim, dim*2]
        self.fc1 = nn.Linear(dim_in, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc_centroid = nn.Linear(dim_hidden[1], 3)
        self.fc_bias = nn.Linear(dim_hidden[1], self.nv*3)
        
        self.color_rec = color_rec
        self.fc_sampling = nn.Linear(dim_hidden[1],64*64*color_num)
        self.fc_selection = nn.Linear(dim_hidden[1],color_num*self.nf)
        self.color_num = color_num
#        self.fc_color = nn.Linear(dim_hidden[1],self.nf*3)
    def forward(self, x, imgs):
        color_num = self.color_num
        batch_size = x.shape[0]
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        # decoder follows NMR
        centroid = self.fc_centroid(x) * self.centroid_scale

        bias = self.fc_bias(x) * self.bias_scale
        bias = bias.view(-1, self.nv, 3)

        base = self.vertices_base * self.obj_scale

        sign = torch.sign(base)
        base = torch.abs(base)
        base = torch.log(base / (1 - base))

        centroid = torch.tanh(centroid[:, None, :])
        scale_pos = 1 - centroid
        scale_neg = centroid + 1

        vertices = torch.sigmoid(base + bias) * sign
        vertices = F.relu(vertices) * scale_pos - F.relu(-vertices) * scale_neg
        vertices = vertices + centroid
        vertices = vertices * 0.5
        faces = self.faces[None, :, :].repeat(batch_size, 1, 1)
        
        if self.color_rec:
            color_palette = torch.bmm(imgs[:,0:3,:,:].view(batch_size,3,-1),F.softmax(self.fc_sampling(x).view(batch_size,-1,color_num),dim=1)) #b * 3 *np
            color = torch.bmm(color_palette,F.softmax(self.fc_selection(x).view(batch_size,color_num,-1),dim=1)) # b * 3 * nf
            color = color.permute(0,2,1)# b * nf * 3
            color = color.view(batch_size,self.nf,1,3)
#            color = F.sigmoid(self.fc_color(x))*255.
#            color = color.view(batch_size,self.nf,1,3)
        else:
            color = None
        return vertices, faces, color
    def no_color_forward(self,x):
        batch_size = x.shape[0]
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        # decoder follows NMR
        centroid = self.fc_centroid(x) * self.centroid_scale

        bias = self.fc_bias(x) * self.bias_scale
        bias = bias.view(-1, self.nv, 3)

        base = self.vertices_base * self.obj_scale

        sign = torch.sign(base)
        base = torch.abs(base)
        base = torch.log(base / (1 - base))

        centroid = torch.tanh(centroid[:, None, :])
        scale_pos = 1 - centroid
        scale_neg = centroid + 1

        vertices = torch.sigmoid(base + bias) * sign
        vertices = F.relu(vertices) * scale_pos - F.relu(-vertices) * scale_neg
        vertices = vertices + centroid
        vertices = vertices * 0.5
        faces = self.faces[None, :, :].repeat(batch_size, 1, 1)
        return vertices, faces
class Unconditional_Generator(nn.Module):
    def __init__(self, filename_obj, cfg):
        super(Unconditional_Generator, self).__init__()
        z_dim = cfg['NormalGan']['G']['z_dim']
        self.decoder = Decoder(filename_obj,color_rec=False,dim_in = z_dim)
        self.renderer = sr.SoftRenderer(image_size= cfg['input_size'], sigma_val=cfg['SoftRender']['SIGMA_VAL'], 
                                        aggr_func_rgb='hard', camera_mode='look_at', viewing_angle=15,
                                        dist_eps=1e-10)
        self.laplacian_loss = sr.LaplacianLoss(self.decoder.vertices_base, self.decoder.faces)
        self.flatten_loss = sr.FlattenLoss(self.decoder.faces)
    def forward(self, z_code):
        return self.decoder.no_color_forward(z_code)
    def model_param(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())
    def set_sigma(self, sigma):
        self.renderer.set_sigma(sigma)
    
class Reconstructor(nn.Module):
    def __init__(self, filename_obj, cfg):
        super(Reconstructor, self).__init__()
        feature_dim = cfg['Reconstruction']['f_dim']
        img_size = cfg['input_size']
        if cfg['Reconstruction']['encoder_type'] == 'resnet':
            self.encoder = ResnetEncoder(c_dim = feature_dim)
        elif ['Reconstruction']['encoder_type'] == 'simple':
            self.encoder = SimpleEncoder(dim_out = feature_dim, im_size=img_size)
        else:
            raise Exception('encoder type must be resnet or simple')
            
        self.color_rec = cfg['Reconstruction']['color_rec']
        self.decoder = Decoder(filename_obj,self.color_rec,color_num=cfg['Reconstruction']['color_num'])
        if self.color_rec:
            self.renderer = sr.SoftRenderer(image_size=img_size, sigma_val=cfg['SoftRender']['SIGMA_VAL'], 
                                        aggr_func_rgb='softmax', camera_mode='look_at', viewing_angle=15,
                                        dist_eps=1e-10)
        else:
            self.renderer = sr.SoftRenderer(image_size= img_size, sigma_val=cfg['SoftRender']['SIGMA_VAL'], 
                                        aggr_func_rgb='hard', camera_mode='look_at', viewing_angle=15,
                                        dist_eps=1e-10)
        self.laplacian_loss = sr.LaplacianLoss(self.decoder.vertices_base, self.decoder.faces)
        self.flatten_loss = sr.FlattenLoss(self.decoder.faces)

    def model_param(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def set_sigma(self, sigma):
        self.renderer.set_sigma(sigma)

    def reconstruct(self, images):
        vertices, faces, color = self.decoder(self.encoder(images[:,0:3,:,:]),images)
        return vertices, faces, color
    def geometry_loss(self,vertices):
        laplacian_loss = self.laplacian_loss(vertices)
        flatten_loss = self.flatten_loss(vertices)
        return laplacian_loss,flatten_loss


class view_predictor(nn.Module):
    def __init__(self,  cfg):
        super(view_predictor,self).__init__()
        
        self.img_size = cfg['input_size']
        self.type = cfg['ViewPrediction']['type']
        
        self.convs1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=4, stride=2, padding=1)
        self.convs2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=4, stride=2, padding=1)
        self.convs3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=4, stride=2, padding=1)
        self.convs4 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=4, stride=2, padding=1)
        self.convs5 = nn.Conv2d(in_channels = 512, out_channels = 100, kernel_size=4, stride=2, padding=0)
        self.m = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(100,50,bias=True)
        
        if self.type == 'regress':
            self.linear2 = nn.Linear(50,1,bias=True)
        elif self.type == 'classify':
            self.linear2 = nn.Linear(50,cfg['ViewPrediction']['class_num'])
        else:
            raise Exception('view prediction type must be regress or classify')
        
#        self.linear_v = nn.Linear(3,32,bias=False)
    def forward(self, imgs):    
        out = imgs.view(-1,1,self.img_size,self.img_size)
        out = F.relu(self.convs1(out))
#        view_feature = F.relu(self.linear_v(view))
#        h = torch.cat((image_feature,view_feature.view(-1,32,1,1).repeat(1,1,32,32)),1)
        out = F.relu(self.convs2(out))
        out = F.relu(self.convs3(out))
        out = F.relu(self.convs4(out))
        out = F.relu(self.convs5(out))
        out = self.m(out)
        out = out.squeeze()
        out = F.relu(self.linear1(out))
        if self.type == 'regress':
            view = F.sigmoid(self.linear2(out)) * 360
        elif self.type == 'classify':
            view = self.linear2(out) # output view class logit, need max after.
        return view
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)    


class ViewDiscriminator(nn.Module):
    def __init__(self, img_size=64):
        super(ViewDiscriminator,self).__init__()

        self.img_size = img_size 
        
        self.convs1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=4, stride=2, padding=1)
        self.convs2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=4, stride=2, padding=1)
        self.convs3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=4, stride=2, padding=1)
        self.convs4 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=4, stride=2, padding=1)
        self.convs5 = nn.Conv2d(in_channels = 256, out_channels = 1, kernel_size=4, stride=2, padding=0)
        self.m = nn.AdaptiveAvgPool2d(32)
        self.linear_v = nn.Linear(3,32,bias=False)
    def forward(self, imgs, view):    
        out = imgs.view(-1,1,self.img_size,self.img_size)
        image_feature = F.leaky_relu(self.convs1(out))
        image_feature = self.m(image_feature)
        view_feature = F.leaky_relu(self.linear_v(view))
        h = torch.cat((image_feature,view_feature.view(-1,32,1,1).repeat(1,1,32,32)),1)
        out = F.leaky_relu(self.convs2(h))
        out = F.leaky_relu(self.convs3(out))
        out = F.leaky_relu(self.convs4(out))
        out = self.convs5(out)
        return torch.sigmoid(out)
    
    def extract_feature(self,imgs,view):
        batch_size = imgs.shape[0]
        pool = torch.nn.AdaptiveAvgPool2d(2)
        out = imgs.view(-1,1,self.img_size,self.img_size)
        image_feature = F.leaky_relu(self.convs1(out))
        image_feature = self.m(image_feature)
        view_feature = F.leaky_relu(self.linear_v(view))
        h = torch.cat((image_feature,view_feature.view(-1,32,1,1).repeat(1,1,32,32)),1)
        out1 = F.leaky_relu(self.convs2(h))
        out2 = F.leaky_relu(self.convs3(out1))
        out3 = F.leaky_relu(self.convs4(out2))
        
        feature = torch.cat((pool(out1),pool(out2),pool(out3)),1)        
        feature = feature.view(batch_size,-1).contiguous()
        return feature
        
        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

#    
#    def predict_multiview(self, image_a, image_b, viewpoint_a, viewpoint_b):
#        batch_size = image_a.size(0)
#        # [Ia, Ib]
#        images = torch.cat((image_a, image_b), dim=0)
#        # [Va, Va, Vb, Vb], set viewpoints
#        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)
#        self.renderer.transform.set_eyes(viewpoints)
#
#        vertices, faces = self.reconstruct(images)
#        laplacian_loss = self.laplacian_loss(vertices)
#        flatten_loss = self.flatten_loss(vertices)
#
#        # [Ma, Mb, Ma, Mb]
#        vertices = torch.cat((vertices, vertices), dim=0)
#        faces = torch.cat((faces, faces), dim=0)
#
#        # [Raa, Rba, Rab, Rbb], cross render multiview images
#        silhouettes = self.renderer(vertices, faces)
#        return silhouettes.chunk(4, dim=0), laplacian_loss, flatten_loss
#
#    def evaluate_iou(self, images, voxels):
#        vertices, faces = self.reconstruct(images)
#
#        faces_ = srf.face_vertices(vertices, faces).data
#        faces_norm = faces_ * 1. * (32. - 1) / 32. + 0.5
#        voxels_predict = srf.voxelization(faces_norm, 32, False).cpu().numpy()
#        voxels_predict = voxels_predict.transpose(0, 2, 1, 3)[:, :, :, ::-1]
#        iou = (voxels * voxels_predict).sum((1, 2, 3)) / (0 < (voxels + voxels_predict)).sum((1, 2, 3))
#        return iou, vertices, faces
#
#    def forward(self, images=None, viewpoints=None, voxels=None, task='train'):
#        if task == 'train':
#            return self.predict_multiview(images[0], images[1], viewpoints[0], viewpoints[1])
#        elif task == 'test':
#            return self.evaluate_iou(images, voxels)
