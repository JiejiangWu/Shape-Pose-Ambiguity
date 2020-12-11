import torch
import os
import torch.autograd as autograd
from src.utils import utils
from src.losses import loss
from src.training.train_utils import adjust_learning_rate,normal_calc_gradient_penalty,conditional_calc_gradient_penalty
import soft_renderer.functional as srf
import imageio
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def train_Normal_GAN(model_D, model_G, optimizer_D, optimizer_G,load_dir,save_dir,cfg,dataset):
    if os.path.exists(os.path.join(load_dir,'Temp.pth.tar')):
            checkpoint = torch.load(os.path.join(load_dir,'Temp.pth.tar'))
            model_D.load_state_dict(checkpoint['model_D'])
            model_G.load_state_dict(checkpoint['model_G'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            start_iteration = checkpoint['iteration']+1
            losslog = checkpoint['losslog']
            w_distances = losslog['w_distance']
            d_costs = losslog['d_cost']
            g_costs = losslog['g_cost']
    else:
        start_iteration = 0
        w_distances = utils.AverageMeter()
        d_costs = utils.AverageMeter()
        g_costs = utils.AverageMeter()
        losslog = {'w_distance':w_distances,'d_cost':d_costs,'g_cost':g_costs}
    
    if not os.path.exists(os.path.join(save_dir,'demo')):
        os.makedirs(os.path.join(save_dir,'demo'))
    batch_size = cfg['NormalGan']['batch_size']
    z_dim = cfg['NormalGan']['G']['z_dim']
    one = torch.FloatTensor([1])
    mone = one * -1
    one = one.to(device)
    mone = mone.to(device)
    for iteration in range(start_iteration,cfg['NormalGan']['iterations']):
        imgs,_,_ = dataset.get_random_single_batch(batch_size)
        imgs = imgs.to(device)
        
        # random views
        views = torch.zeros(3,batch_size)
        views = views.to(device)
        views[0] = 2.732
        views[1] = 30
        views[2] = torch.rand(batch_size)*360 # only consider azimuth
        
        '''-----------train D-----------'''
        for p in model_D.parameters(): 
            p.requires_grad = True        
        optimizer_D.zero_grad()
        
        '''train with real img'''
        real_img = imgs[:,3,:,:] # only input silhouettes
        D_real_result = model_D(real_img)
        D_real_result = D_real_result.mean()
        D_real_result.backward(mone)  
        '''train with fake img'''
        z_noise = torch.empty(batch_size, z_dim).normal_(0,0.33).to(device) # z_noise range 0-0.33
        # render img of generated 3D shapes
        viewpoints = srf.get_points_from_angles(views[0],views[1],views[2])
        model_G.set_sigma(cfg['SoftRender']['SIGMA_VAL'])
        model_G.renderer.transform.set_eyes(viewpoints)
        vertices, faces = model_G(z_noise)
        rgba_fake_img = model_G.renderer(vertices, faces)
        fake_img=rgba_fake_img[:,3,:,:]
        D_fake_result = model_D(fake_img)
        D_fake_result = D_fake_result.mean()
        D_fake_result.backward(one)
        '''train with gradient penalty'''
        gradient_penalty = normal_calc_gradient_penalty(model_D, real_img.data, fake_img.data,cfg,False)
        gradient_penalty.backward()        
        optimizer_D.step() 
        D_cost = D_fake_result - D_real_result + gradient_penalty
        Wasserstein_D = D_real_result - D_fake_result
        d_costs.update(D_cost.item())
        w_distances.update(Wasserstein_D.item())
        
        '''-----------train G-----------'''
        if iteration % cfg['NormalGan']['D_G_rate'] == 0:
            optimizer_G.zero_grad()
            views[2] = torch.rand(batch_size)*360 # only consider azimuth
            z_noise = torch.empty(batch_size, z_dim).normal_(0,0.33).to(device)
            # render img of generated 3D shapes
            viewpoints = srf.get_points_from_angles(views[0],views[1],views[2])
            model_G.set_sigma(cfg['SoftRender']['SIGMA_VAL'])
            model_G.renderer.transform.set_eyes(viewpoints)
            vertices, faces = model_G(z_noise)
            rgba_fake_img = model_G.renderer(vertices, faces)
            fake_img=rgba_fake_img[:,3,:,:]
            D_result = model_D(fake_img)
            D_result = D_result.mean()
            D_result.backward(mone,retain_graph=True)
            temp_mean_GAN_loss = abs(model_G.decoder.fc_bias.weight.grad.mean().item())
            '''train with geometric reguliazer'''
            laplacian_loss = model_G.laplacian_loss(vertices).mean()*cfg['NormalGan']['G']['LAMBDA_LAPLACIAN']
            flatten_loss = model_G.flatten_loss(vertices).mean()*cfg['NormalGan']['G']['LAMBDA_LAPLACIAN']
            geometry_loss = (flatten_loss+laplacian_loss)*(temp_mean_GAN_loss/2e-5)
            
            geometry_loss.backward()
#            (flatten_loss+laplacian_loss).backward()
            optimizer_G.step()
            G_cost = -D_result 
            g_costs.update(G_cost.item()) 
        '''save checkpoints'''
        if iteration % cfg['checkpoint_freq'] == 0:
            model_path = os.path.join(save_dir, 'checkpoint_%07d.pth.tar'%iteration)
            temp_path = os.path.join(save_dir,'Temp.pth.tar')
            state = {'model_D':model_D.state_dict(),
                     'model_G':model_G.state_dict(),
                     'optimizer_D':optimizer_D.state_dict(),
                     'optimizer_G':optimizer_G.state_dict(),
                     'iteration': iteration,
                     'losslog':losslog,
                    }
            torch.save(state,model_path)
            torch.save(state,temp_path)
        '''decay lr'''
        if iteration >= cfg['NormalGan']['decay_begining'] and iteration % cfg['NormalGan']['decay_step'] == 0:
            adjust_learning_rate(optimizer_G, decay_rate=cfg['NormalGan']['decay_rate'])
            adjust_learning_rate(optimizer_D, decay_rate=cfg['NormalGan']['decay_rate'])
        
        # save demo images
        if iteration % cfg['demo_freq'] == 0:
            
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_fake.png' % iteration), utils.img_cvt(rgba_fake_img[0]))

        if iteration % cfg['print_freq'] == 0:
            print('Train_I: [{0}/{1}]\t'
                               'D_cost {loss0.val:.1f} ({loss0.avg:.1f})\t'
                               'G_cost {loss1.val:.1f} ({loss1.avg:.1f})\t'
                               'W_dist {loss2.val:.1f} ({loss2.avg:.1f})\t'.format(iteration, cfg['NormalGan']['iterations'],
                                                                                loss0=d_costs,loss1=g_costs,loss2=w_distances))
            
def train_view_predictor(model_G,G_load_dir,model_vp,optimizer_vp,load_dir,save_dir,cfg):
    if G_load_dir:
        checkpoint_G = torch.load(G_load_dir)
        
        pre_dict = checkpoint_G['model_G']
        model_dict = model_G.state_dict()
        pretrained_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model_G.load_state_dict(model_dict)
    else:
        raise Exception('cant find checkpoint')
        
        
    if os.path.exists(os.path.join(load_dir,'Temp.pth.tar')):
        checkpoint = torch.load(os.path.join(load_dir,'Temp.pth.tar'))
        model_vp.load_state_dict(checkpoint['model_vp'])
        optimizer_vp.load_state_dict(checkpoint['optimizer_vp'])
        start_iteration = checkpoint['iteration']+1
        losslog = checkpoint['losslog']
        losstype = losslog['losstype']
        losses = losslog['losses']
    else:
        start_iteration = 0
        losses = utils.AverageMeter()
        losstype = cfg['ViewPrediction']['type']
        losslog = {'losses':losses,'losstype':losstype}
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    batch_size = cfg['ViewPrediction']['batch_size']
    z_dim = cfg['NormalGan']['G']['z_dim']    
    for iteration in range(start_iteration,cfg['ViewPrediction']['iterations']):
        '''-----------generate fake training data-----------'''
        # random views
        views = torch.zeros(3,batch_size)
        views = views.to(device)
        views[0] = 2.732
        views[1] = 30
        views[2] = torch.rand(batch_size)*360 # only consider azimuth
        z_noise = torch.empty(batch_size, z_dim).normal_(0,0.33).to(device) # z_noise range 0-0.33
        # render img of generated 3D shapes
        viewpoints = srf.get_points_from_angles(views[0],views[1],views[2])
        model_G.set_sigma(cfg['SoftRender']['SIGMA_VAL'])
        model_G.renderer.transform.set_eyes(viewpoints)
        vertices, faces = model_G(z_noise)
        rgba_fake_img = model_G.renderer(vertices, faces)
        fake_img=rgba_fake_img[:,3,:,:]    
        
        '''-----------train VP-----------'''
        predicted_view = model_vp(fake_img)
        CE = torch.nn.CrossEntropyLoss()
        if cfg['ViewPrediction']['type'] == 'regress':
            view_loss = loss.angle_loss(predicted_view.view(batch_size),views[2].view(batch_size)).mean()
        elif cfg['ViewPrediction']['type'] == 'classify':
            views_label = views[2] // (360/cfg['ViewPrediction']['class_num'])
            view_loss = CE(predicted_view,views_label)
        
        optimizer_vp.zero_grad()
        view_loss.backward()
        optimizer_vp.step()
        losses.update(view_loss.item()) 
        
        '''save checkpoints'''
        if iteration % cfg['checkpoint_freq'] == 0:
            model_path = os.path.join(save_dir, 'checkpoint_%07d.pth.tar'%iteration)
            temp_path = os.path.join(save_dir,'Temp.pth.tar')
            state = {'model_vp':model_vp.state_dict(),
                     'optimizer_vp':optimizer_vp.state_dict(),
                     'iteration': iteration,
                     'losslog':losslog,
                    }
            torch.save(state,model_path)
            torch.save(state,temp_path)
            
        '''decay lr'''
        if iteration >= cfg['ViewPrediction']['decay_begining'] and iteration % cfg['ViewPrediction']['decay_step'] == 0:
            adjust_learning_rate(optimizer_vp, decay_rate=cfg['ViewPrediction']['decay_rate'])
        
        
        '''print training info'''
        for param_group in optimizer_vp.param_groups:
            lr = param_group['lr']
            break
        if iteration % cfg['print_freq'] == 0:
            print('Train_I: [{0}/{1}]\t'
                               'view_loss {loss0.val:.1f} ({loss0.avg:.1f})\t'
                               'lr {lr:.5f}\t'.format(iteration, cfg['ViewPrediction']['iterations'],
                                                                                loss0=losses,lr=lr))        
    

def train_imageEncoder(model_G,G_load_dir,model_vp,vp_load_dir,model_E,optimizer_E,load_dir,save_dir,cfg,dataset):
    batch_size = cfg['Reconstruction']['batch_size']
    
    if G_load_dir:
        checkpoint_G = torch.load(G_load_dir)
        pre_dict = checkpoint_G['model_G']
        model_dict = model_G.state_dict()
        pretrained_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model_G.load_state_dict(model_dict)
    else:
        raise Exception('cant find checkpoint')
    if vp_load_dir:
        checkpoint_vp = torch.load(vp_load_dir)
        model_vp.load_state_dict(checkpoint_vp['model_vp'])
    else:
        raise Exception('cant find checkpoint')
         
    if os.path.exists(os.path.join(load_dir,'Temp.pth.tar')):
        checkpoint = torch.load(os.path.join(load_dir,'Temp.pth.tar'))
        model_E.load_state_dict(checkpoint['model_E'])
        optimizer_E.load_state_dict(checkpoint['optimizer_E'])
        start_iteration = checkpoint['iteration']+1
        losslog = checkpoint['losslog']
        IoU_losses = losslog['IoU_losses']
    else:
        start_iteration = 0
        IoU_losses = utils.AverageMeter()
        losslog = {'IoU_losses':IoU_losses}
    
    if not os.path.exists(os.path.join(save_dir,'demo')):
        os.makedirs(os.path.join(save_dir,'demo'))
  
    for iteration in range(start_iteration,cfg['Reconstruction']['iterations']):
        imgs,_,_ = dataset.get_random_single_batch(batch_size)
        imgs = imgs.to(device)    
        
        # predict views
        views = torch.zeros(3,batch_size)
        views = views.to(device)
        views[0] = 2.732
        views[1] = 30
        if cfg['Reconstruction']['use_vp']:
            views[2] = model_vp(imgs[:,3,:,:]).view(batch_size)   
        else:
            views[2] = 0    
            '''-----------train E-----------'''
        for p in model_E.parameters(): 
            p.requires_grad = True               
        
        '''perform reconstruction'''
        image_feature = model_E(imgs[:,0:3,:,:])
        vertices, faces = model_G(image_feature)
        viewpoints = srf.get_points_from_angles(views[0],views[1],views[2])
        model_G.set_sigma(cfg['SoftRender']['SIGMA_VAL'])
        model_G.renderer.transform.set_eyes(viewpoints)
        
        rgba_output_img = model_G.renderer(vertices, faces)
        
        # reprojection loss
        IoU_loss,pixel_color_loss = loss.reprojection_loss(rgba_output_img,imgs)
        IoU_loss=IoU_loss.mean()
        pixel_color_loss=pixel_color_loss.mean()
        
        optimizer_E.zero_grad()
        IoU_loss.backward()
        optimizer_E.step()
        IoU_losses.update(IoU_loss.item())
        
        '''save checkpoints'''
        if iteration % cfg['checkpoint_freq'] == 0:
            model_path = os.path.join(save_dir, 'checkpoint_%07d.pth.tar'%iteration)
            temp_path = os.path.join(save_dir,'Temp.pth.tar')
            state = {'model_E':model_E.state_dict(),
                     'optimizer_E':optimizer_E.state_dict(),
                     'iteration': iteration,
                     'losslog':losslog,
                    }
            torch.save(state,model_path)
            torch.save(state,temp_path)
            
        '''decay lr'''
        if iteration >= cfg['Reconstruction']['decay_begining'] and iteration % cfg['Reconstruction']['decay_step'] == 0:
            adjust_learning_rate(optimizer_E, decay_rate=cfg['Reconstruction']['decay_rate'])
        
        # save demo images
        if iteration % cfg['demo_freq'] == 0:
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_real.png' % iteration), utils.img_cvt(imgs[0]))
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_fake1.png' % iteration), utils.img_cvt(rgba_output_img[0]))
        if iteration % cfg['print_freq'] == 0:
            print('Train_I: [{0}/{1}]\t'
                               'IoU_loss {loss0.val:.3f} ({loss0.avg:.3f})\t'.format(iteration, cfg['Reconstruction']['iterations'],
                                                                                loss0=IoU_losses))
            
def train_reconstructor(model_vp,vp_load_dir,model_R,optimizer_R,model_D,optimizer_D,load_dir,save_dir,cfg,dataset):
    
    batch_size = cfg['Reconstruction']['batch_size']
    color_rec = cfg['Reconstruction']['color_rec']    
    if vp_load_dir:
        checkpoint_vp = torch.load(vp_load_dir)
        model_vp.load_state_dict(checkpoint_vp['model_vp'])
    else:
        raise Exception('cant find checkpoint')
        
    if os.path.exists(os.path.join(load_dir,'Temp.pth.tar')):
        checkpoint = torch.load(os.path.join(load_dir,'Temp.pth.tar'))
        model_R.load_state_dict(checkpoint['model_R'])
        optimizer_R.load_state_dict(checkpoint['optimizer_R'])
        model_D.load_state_dict(checkpoint['model_D'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        start_iteration = checkpoint['iteration']+1
        losslog = checkpoint['losslog']
        IoU_losses = losslog['IoU_losses']
        mixed_color_losses = losslog['mixed_color_losses']
        d_costs = losslog['d_costs']
        w_distances = losslog['w_distances']
        g_costs = losslog['g_costs']
    else:
        start_iteration = 0
        IoU_losses = utils.AverageMeter()
        mixed_color_losses = utils.AverageMeter()
        d_costs = utils.AverageMeter()
        w_distances = utils.AverageMeter()
        g_costs = utils.AverageMeter()
        losslog = {'IoU_losses':IoU_losses,'mixed_color_losses':mixed_color_losses,'d_costs':d_costs,'w_distances':w_distances,'g_costs':g_costs}

    if not os.path.exists(os.path.join(save_dir,'demo')):
        os.makedirs(os.path.join(save_dir,'demo'))

    one = torch.FloatTensor([1])
    mone = one * -1
    one = one.to(device)
    mone = mone.to(device)
    
    for iteration in range(start_iteration,cfg['Reconstruction']['iterations']):
        imgs,_,_ = dataset.get_random_single_batch(batch_size)
        imgs = imgs.to(device)    
        
        # predict views
        views = torch.zeros(3,batch_size)
        views = views.to(device)
        views[0] = 2.732
        views[1] = 30
        if cfg['Reconstruction']['use_vp']:
            views[2] = model_vp(imgs[:,3,:,:]).view(batch_size)   
        else:
            views[2] = 0
        '''-----------train D-----------'''
        for p in model_D.parameters(): 
            p.requires_grad = True        
        
        '''perform reconstruction'''
        vertices, faces, color = model_R.reconstruct(imgs)
        viewpoints = srf.get_points_from_angles(views[0],views[1],views[2])
        model_R.set_sigma(cfg['SoftRender']['SIGMA_VAL'])
        model_R.renderer.transform.set_eyes(viewpoints)
        
        rgba_output_img = model_R.renderer(vertices, faces, color)
        alpha_output_img=rgba_output_img[:,3,:,:]
        
        '''1:train with real img (from predicted input view)'''
        if cfg['Reconstruction']['ViewGan']['dis_domain'] == 'img':
            real_img = imgs[:,3,:,:] # only input silhouettes
        elif cfg['Reconstruction']['ViewGan']['dis_domain'] == 'view':
            real_img = alpha_output_img
            
        # discriminate         
        D_real_result = model_D(real_img)
        D_real_result = D_real_result.mean()
#        D_real_result.backward(mone)  
        
        '''2:train with fake img (from other random view)'''
        random_view = torch.rand(views[2].shape) * 360
        random_view = random_view.to(device)
        viewpoints = srf.get_points_from_angles(views[0],views[1],random_view)
        model_R.renderer.transform.set_eyes(viewpoints)
        rgba_fake_img = model_R.renderer(vertices, faces, color)
        fake_img=rgba_fake_img[:,3,:,:]
        
        # discriminate
        D_fake_result = model_D(fake_img)  
        D_fake_result = D_fake_result.mean()
#        D_fake_result.backward(one)
        
        '''3:train with gradient penalty'''
        gradient_penalty = normal_calc_gradient_penalty(model_D, real_img.data, fake_img.data,cfg,True)
        D_cost = D_fake_result - D_real_result + gradient_penalty
        
        optimizer_D.zero_grad()
        D_cost.backward()        
        optimizer_D.step() 
        Wasserstein_D = D_real_result - D_fake_result
        d_costs.update(D_cost.item())
        w_distances.update(Wasserstein_D.item())
    
        
        '''-----------train R(G)-----------'''
        LAMBDA_FLATTEN = cfg['Reconstruction']['LAMBDA_FLATTEN']
        LAMBDA_LAPLACIAN = cfg['Reconstruction']['LAMBDA_LAPLACIAN']
        LAMBDA_DISC = cfg['Reconstruction']['LAMBDA_DISC']
        if color_rec:
            LAMBDA_COLOR = cfg['Reconstruction']['LAMBDA_COLOR']
        else:
            LAMBDA_COLOR = 0

        imgs,_,_ = dataset.get_random_single_batch(batch_size)
        imgs = imgs.to(device)    
        # predict views
        views = torch.zeros(3,batch_size)
        views = views.to(device)
        views[0] = 2.732
        views[1] = 30
        if cfg['Reconstruction']['use_vp']:
            views[2] = model_vp(imgs[:,3,:,:]).view(batch_size)    
        else:
            views[2] = 0
        # perform reconstruction
        vertices, faces, color = model_R.reconstruct(imgs)
        viewpoints = srf.get_points_from_angles(views[0],views[1],views[2])
        model_R.set_sigma(cfg['SoftRender']['SIGMA_VAL'])
        model_R.renderer.transform.set_eyes(viewpoints)
        
        rgba_output_img = model_R.renderer(vertices, faces, color)
        alpha_output_img=rgba_output_img[:,3,:,:]
        
        # geometry regulizer loss
        laplacian_loss,flatten_loss = model_R.geometry_loss(vertices)   
        laplacian_loss = laplacian_loss.mean()
        flatten_loss = flatten_loss.mean()
        # reprojection loss
        IoU_loss,pixel_color_loss = loss.reprojection_loss(rgba_output_img,imgs)
        IoU_loss=IoU_loss.mean()
        pixel_color_loss=pixel_color_loss.mean()
        # adversial loss
        random_view = torch.rand(views[2].shape) * 360
        random_view = random_view.to(device)            
        viewpoints = srf.get_points_from_angles(views[0],views[1],random_view)
        model_R.renderer.transform.set_eyes(viewpoints)
        rgba_output_img2 = model_R.renderer(vertices, faces, color)
        if iteration % cfg['Reconstruction']['D_G_rate'] == 0:
            fake_img = rgba_output_img2[:,3,:,:]
            G_cost = model_D(fake_img)*-1
            G_cost = G_cost.mean()
            R_loss = flatten_loss*LAMBDA_FLATTEN + laplacian_loss*LAMBDA_LAPLACIAN + G_cost*LAMBDA_DISC + IoU_loss
            g_costs.update(G_cost.item()) 
        else:
            R_loss = flatten_loss*LAMBDA_FLATTEN + laplacian_loss*LAMBDA_LAPLACIAN + IoU_loss
#        
        if iteration > cfg['Reconstruction']['color_loss_begin']:
            R_loss += pixel_color_loss*LAMBDA_COLOR
#        R_loss+=pixel_color_loss*LAMBDA_COLOR
        optimizer_R.zero_grad()
        R_loss.backward()
        optimizer_R.step()
        IoU_losses.update(IoU_loss.item())
        mixed_color_losses.update(pixel_color_loss.item())
        
        '''save checkpoints'''
        if iteration % cfg['checkpoint_freq'] == 0:
            model_path = os.path.join(save_dir, 'checkpoint_%07d.pth.tar'%iteration)
            temp_path = os.path.join(save_dir,'Temp.pth.tar')
            state = {'model_R':model_R.state_dict(),
                     'optimizer_R':optimizer_R.state_dict(),
                     'model_D':model_D.state_dict(),
                     'optimizer_D':optimizer_D.state_dict(),
                     'iteration': iteration,
                     'losslog':losslog,
                    }
            torch.save(state,model_path)
            torch.save(state,temp_path)
            
        '''decay lr'''
        if iteration >= cfg['Reconstruction']['decay_begining'] and iteration % cfg['Reconstruction']['decay_step'] == 0:
            adjust_learning_rate(optimizer_R, decay_rate=cfg['Reconstruction']['decay_rate'])
            adjust_learning_rate(optimizer_D, decay_rate=cfg['Reconstruction']['decay_rate'])
        
        # save demo images
        if iteration % cfg['demo_freq'] == 0:
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_real.png' % iteration), utils.img_cvt(imgs[0]))
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_fake1.png' % iteration), utils.img_cvt(rgba_output_img[0]))
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_fake2.png' % iteration), utils.img_cvt(rgba_output_img2[0]))
#            srf.save_obj(os.path.join(save_dir,'demo', '%07d_fake.obj' % iteration), vertices[0], faces[0])
        if iteration % cfg['print_freq'] == 0:
            print('Train_I: [{0}/{1}]\t'
                               'IoU_loss {loss0.val:.3f} ({loss0.avg:.3f})\t'
                               'Color_loss {loss1.val:.3f} ({loss1.avg:.3f})\t'
                               'W_dist {loss2.val:.3f} ({loss2.avg:.3f})\t'.format(iteration, cfg['Reconstruction']['iterations'],
                                                                                loss0=IoU_losses,loss1=mixed_color_losses,loss2=w_distances))
            
def train_naive_reconstructor(model_R,optimizer_R,load_dir,save_dir,cfg,dataset):
    batch_size = cfg['Reconstruction']['batch_size']
    LAMBDA_FLATTEN = cfg['Reconstruction']['LAMBDA_FLATTEN']
    LAMBDA_LAPLACIAN = cfg['Reconstruction']['LAMBDA_LAPLACIAN']
    if os.path.exists(os.path.join(load_dir,'Temp.pth.tar')):
        checkpoint = torch.load(os.path.join(load_dir,'Temp.pth.tar'))
        model_R.load_state_dict(checkpoint['model_R'])
        optimizer_R.load_state_dict(checkpoint['optimizer_R'])
        start_iteration = checkpoint['iteration']+1
        losslog = checkpoint['losslog']
        IoU_losses = losslog['IoU_losses']
    else:
        start_iteration = 0
        IoU_losses = utils.AverageMeter()
        losslog = {'IoU_losses':IoU_losses}

    if not os.path.exists(os.path.join(save_dir,'demo')):
        os.makedirs(os.path.join(save_dir,'demo'))
        
    for iteration in range(start_iteration,cfg['Reconstruction']['iterations']):
        imgs,_,views = dataset.get_random_single_batch(batch_size)
        imgs = imgs.to(device)   
        views = views.to(device)        
        # perform reconstruction
        vertices, faces, color = model_R.reconstruct(imgs)
        viewpoints = srf.get_points_from_angles(views[0],views[1],views[2])
        model_R.set_sigma(cfg['SoftRender']['SIGMA_VAL'])
        model_R.renderer.transform.set_eyes(viewpoints)
        
        rgba_output_img = model_R.renderer(vertices, faces, color)
        alpha_output_img=rgba_output_img[:,3,:,:]
        
        # geometry regulizer loss
        laplacian_loss,flatten_loss = model_R.geometry_loss(vertices)   
        laplacian_loss = laplacian_loss.mean()
        flatten_loss = flatten_loss.mean()
        # reprojection loss
        IoU_loss,pixel_color_loss = loss.reprojection_loss(rgba_output_img,imgs)
        IoU_loss=IoU_loss.mean()

        R_loss = flatten_loss*LAMBDA_FLATTEN + laplacian_loss*LAMBDA_LAPLACIAN + IoU_loss
        optimizer_R.zero_grad()
        R_loss.backward()
        optimizer_R.step()
        IoU_losses.update(IoU_loss.item())
        
        random_view = torch.rand(views[2].shape) * 360
        random_view = random_view.to(device)            
        viewpoints = srf.get_points_from_angles(views[0],views[1],random_view)
        model_R.renderer.transform.set_eyes(viewpoints)
        rgba_output_img2 = model_R.renderer(vertices, faces, color)
        
        '''save checkpoints'''
        if iteration % cfg['checkpoint_freq'] == 0:
            model_path = os.path.join(save_dir, 'checkpoint_%07d.pth.tar'%iteration)
            temp_path = os.path.join(save_dir,'Temp.pth.tar')
            state = {'model_R':model_R.state_dict(),
                     'optimizer_R':optimizer_R.state_dict(),
                     'iteration': iteration,
                     'losslog':losslog,
                    }
            torch.save(state,model_path)
            torch.save(state,temp_path)
            
        '''decay lr'''
        if iteration >= cfg['Reconstruction']['decay_begining'] and iteration % cfg['Reconstruction']['decay_step'] == 0:
            adjust_learning_rate(optimizer_R, decay_rate=cfg['Reconstruction']['decay_rate'])
        
        # save demo images
        
        if iteration % cfg['demo_freq'] == 0:
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_real.png' % iteration), utils.img_cvt(imgs[0]))
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_fake1.png' % iteration), utils.img_cvt(rgba_output_img[0]))
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_fake2.png' % iteration), utils.img_cvt(rgba_output_img2[0]))
#            srf.save_obj(os.path.join(save_dir,'demo', '%07d_fake.obj' % iteration), vertices[0], faces[0])
        if iteration % cfg['print_freq'] == 0:
            print('Train_I: [{0}/{1}]\t'
                               'IoU_loss {loss0.val:.3f} ({loss0.avg:.3f})\t'.format(iteration, cfg['Reconstruction']['iterations'],
                                                                                loss0=IoU_losses))
            

def train_naive_vp_reconstructor(model_vp,optimizer_vp,model_R,optimizer_R,load_dir,save_dir,cfg,dataset):
    batch_size = cfg['Reconstruction']['batch_size']
    LAMBDA_FLATTEN = cfg['Reconstruction']['LAMBDA_FLATTEN']
    LAMBDA_LAPLACIAN = cfg['Reconstruction']['LAMBDA_LAPLACIAN']
    if os.path.exists(os.path.join(load_dir,'Temp.pth.tar')):
        checkpoint = torch.load(os.path.join(load_dir,'Temp.pth.tar'))
        model_R.load_state_dict(checkpoint['model_R'])
        optimizer_R.load_state_dict(checkpoint['optimizer_R'])
        model_vp.load_state_dict(checkpoint['model_vp'])
        optimizer_vp.load_state_dict(checkpoint['optimizer_vp'])
        start_iteration = checkpoint['iteration']+1
        losslog = checkpoint['losslog']
        IoU_losses = losslog['IoU_losses']
    else:
        start_iteration = 0
        IoU_losses = utils.AverageMeter()
        losslog = {'IoU_losses':IoU_losses}

    if not os.path.exists(os.path.join(save_dir,'demo')):
        os.makedirs(os.path.join(save_dir,'demo'))
        
    for iteration in range(start_iteration,cfg['Reconstruction']['iterations']):
        imgs,_,_ = dataset.get_random_single_batch(batch_size)
        imgs = imgs.to(device)   
        
        # predict views
        views = torch.zeros(3,batch_size)
        views = views.to(device)
        views[0] = 2.732
        views[1] = 30
        views[2] = model_vp(imgs[:,3,:,:]).view(batch_size)
        
        # perform reconstruction
        vertices, faces, color = model_R.reconstruct(imgs)
        viewpoints = srf.get_points_from_angles(views[0],views[1],views[2])
           
        model_R.set_sigma(cfg['SoftRender']['SIGMA_VAL'])
        model_R.renderer.transform.set_eyes(viewpoints)
        
        rgba_output_img = model_R.renderer(vertices, faces, color)
        alpha_output_img=rgba_output_img[:,3,:,:]
        
        # geometry regulizer loss
        laplacian_loss,flatten_loss = model_R.geometry_loss(vertices)   
        laplacian_loss = laplacian_loss.mean()
        flatten_loss = flatten_loss.mean()
        # reprojection loss
        IoU_loss,pixel_color_loss = loss.reprojection_loss(rgba_output_img,imgs)
        IoU_loss=IoU_loss.mean()

        R_loss = flatten_loss*LAMBDA_FLATTEN + laplacian_loss*LAMBDA_LAPLACIAN + IoU_loss
        
        optimizer_R.zero_grad()
        optimizer_vp.zero_grad()
        R_loss.backward()
        optimizer_R.step()
        optimizer_vp.step()
        IoU_losses.update(IoU_loss.item())
        
        random_view = torch.rand(views[2].shape) * 360
        random_view = random_view.to(device)            
        viewpoints = srf.get_points_from_angles(views[0],views[1],random_view)
        model_R.renderer.transform.set_eyes(viewpoints)
        rgba_output_img2 = model_R.renderer(vertices, faces, color)
        
        '''save checkpoints'''
        if iteration % cfg['checkpoint_freq'] == 0:
            model_path = os.path.join(save_dir, 'checkpoint_%07d.pth.tar'%iteration)
            temp_path = os.path.join(save_dir,'Temp.pth.tar')
            state = {'model_R':model_R.state_dict(),
                     'optimizer_R':optimizer_R.state_dict(),
                     'model_vp':model_vp.state_dict(),
                     'optimizer_vp':optimizer_vp.state_dict(),
                     'iteration': iteration,
                     'losslog':losslog,
                    }
            torch.save(state,model_path)
            torch.save(state,temp_path)
            
        '''decay lr'''
        if iteration >= cfg['Reconstruction']['decay_begining'] and iteration % cfg['Reconstruction']['decay_step'] == 0:
            adjust_learning_rate(optimizer_R, decay_rate=cfg['Reconstruction']['decay_rate'])
        
        # save demo images
        
        if iteration % cfg['demo_freq'] == 0:
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_real.png' % iteration), utils.img_cvt(imgs[0]))
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_fake1.png' % iteration), utils.img_cvt(rgba_output_img[0]))
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_fake2.png' % iteration), utils.img_cvt(rgba_output_img2[0]))
#            srf.save_obj(os.path.join(save_dir,'demo', '%07d_fake.obj' % iteration), vertices[0], faces[0])
        if iteration % cfg['print_freq'] == 0:
            print('Train_I: [{0}/{1}]\t'
                               'IoU_loss {loss0.val:.3f} ({loss0.avg:.3f})\t'.format(iteration, cfg['Reconstruction']['iterations'],
                                                                                loss0=IoU_losses))

def train_vpl_reconstructor(model_R,optimizer_R,model_D,optimizer_D,load_dir,save_dir,cfg,dataset):
    
     
    batch_size = cfg['Reconstruction']['batch_size']
    color_rec = cfg['Reconstruction']['color_rec']    
        
    if os.path.exists(os.path.join(load_dir,'Temp.pth.tar')):
        checkpoint = torch.load(os.path.join(load_dir,'Temp.pth.tar'))
        model_R.load_state_dict(checkpoint['model_R'])
        optimizer_R.load_state_dict(checkpoint['optimizer_R'])
        model_D.load_state_dict(checkpoint['model_D'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        start_iteration = checkpoint['iteration']+1
        losslog = checkpoint['losslog']
        IoU_losses = losslog['IoU_losses']
        d_costs = losslog['d_costs']
        w_distances = losslog['w_distances']
        g_costs = losslog['g_costs']
    else:
        start_iteration = 0
        IoU_losses = utils.AverageMeter()
        mixed_color_losses = utils.AverageMeter()
        d_costs = utils.AverageMeter()
        w_distances = utils.AverageMeter()
        g_costs = utils.AverageMeter()
        losslog = {'IoU_losses':IoU_losses,'d_costs':d_costs,'w_distances':w_distances,'g_costs':g_costs}

    if not os.path.exists(os.path.join(save_dir,'demo')):
        os.makedirs(os.path.join(save_dir,'demo'))

    one = torch.FloatTensor([1])
    mone = one * -1
    one = one.to(device)
    mone = mone.to(device)
    
    for iteration in range(start_iteration,cfg['Reconstruction']['iterations']):
        imgs,_,views = dataset.get_random_single_batch(batch_size)
        imgs = imgs.to(device)   
        views = views.to(device)        

        '''-----------train D-----------'''
        for p in model_D.parameters(): 
            p.requires_grad = True        
        
        '''perform reconstruction'''
        vertices, faces, color = model_R.reconstruct(imgs)
        viewpoints = srf.get_points_from_angles(views[0],views[1],views[2])
        model_R.set_sigma(cfg['SoftRender']['SIGMA_VAL'])
        model_R.renderer.transform.set_eyes(viewpoints)
        
        rgba_output_img = model_R.renderer(vertices, faces, color)
        alpha_output_img=rgba_output_img[:,3,:,:]
        
        '''1:train with real img (from predicted input view)'''
        if cfg['Reconstruction']['ViewGan']['dis_domain'] == 'img':
            real_img = imgs[:,3,:,:] # only input silhouettes
        elif cfg['Reconstruction']['ViewGan']['dis_domain'] == 'view':
            real_img = alpha_output_img
            
        # discriminate         
        D_real_result = model_D(real_img)
        D_real_result = D_real_result.mean()
#        D_real_result.backward(mone)  
        
        '''2:train with fake img (from other random view)'''
        random_view = torch.rand(views[2].shape) * 360
        random_view = random_view.to(device)
        viewpoints = srf.get_points_from_angles(views[0],views[1],random_view)
        model_R.renderer.transform.set_eyes(viewpoints)
        rgba_fake_img = model_R.renderer(vertices, faces, color)
        fake_img=rgba_fake_img[:,3,:,:]
        
        # discriminate
        D_fake_result = model_D(fake_img)  
        D_fake_result = D_fake_result.mean()
#        D_fake_result.backward(one)
        
        '''3:train with gradient penalty'''
        gradient_penalty = normal_calc_gradient_penalty(model_D, real_img.data, fake_img.data,cfg,True)
        D_cost = D_fake_result - D_real_result + gradient_penalty
        
        optimizer_D.zero_grad()
        D_cost.backward()        
        optimizer_D.step() 
        Wasserstein_D = D_real_result - D_fake_result
        d_costs.update(D_cost.item())
        w_distances.update(Wasserstein_D.item())
    
        
        '''-----------train R(G)-----------'''
        LAMBDA_FLATTEN = cfg['Reconstruction']['LAMBDA_FLATTEN']
        LAMBDA_LAPLACIAN = cfg['Reconstruction']['LAMBDA_LAPLACIAN']
        LAMBDA_DISC = cfg['Reconstruction']['LAMBDA_DISC']
        if color_rec:
            LAMBDA_COLOR = cfg['Reconstruction']['LAMBDA_COLOR']
        else:
            LAMBDA_COLOR = 0

        imgs,_,views = dataset.get_random_single_batch(batch_size)
        imgs = imgs.to(device)    
        views = views.to(device)
        # perform reconstruction
        vertices, faces, color = model_R.reconstruct(imgs)
        viewpoints = srf.get_points_from_angles(views[0],views[1],views[2])
        model_R.set_sigma(cfg['SoftRender']['SIGMA_VAL'])
        model_R.renderer.transform.set_eyes(viewpoints)
        
        rgba_output_img = model_R.renderer(vertices, faces, color)
        alpha_output_img=rgba_output_img[:,3,:,:]
        
        # geometry regulizer loss
        laplacian_loss,flatten_loss = model_R.geometry_loss(vertices)   
        laplacian_loss = laplacian_loss.mean()
        flatten_loss = flatten_loss.mean()
        # reprojection loss
        IoU_loss,pixel_color_loss = loss.reprojection_loss(rgba_output_img,imgs)
        IoU_loss=IoU_loss.mean()
        pixel_color_loss=pixel_color_loss.mean()
        # adversial loss
        random_view = torch.rand(views[2].shape) * 360
        random_view = random_view.to(device)            
        viewpoints = srf.get_points_from_angles(views[0],views[1],random_view)
        model_R.renderer.transform.set_eyes(viewpoints)
        rgba_output_img2 = model_R.renderer(vertices, faces, color)
        if iteration % cfg['Reconstruction']['D_G_rate'] == 0:
            fake_img = rgba_output_img2[:,3,:,:]
            G_cost = model_D(fake_img)*-1
            G_cost = G_cost.mean()
            R_loss = flatten_loss*LAMBDA_FLATTEN + laplacian_loss*LAMBDA_LAPLACIAN + G_cost*LAMBDA_DISC + IoU_loss
            g_costs.update(G_cost.item()) 
        else:
            R_loss = flatten_loss*LAMBDA_FLATTEN + laplacian_loss*LAMBDA_LAPLACIAN + IoU_loss
#        
        if iteration > cfg['Reconstruction']['color_loss_begin']:
            R_loss += pixel_color_loss*LAMBDA_COLOR
#        R_loss+=pixel_color_loss*LAMBDA_COLOR
        optimizer_R.zero_grad()
        R_loss.backward()
        optimizer_R.step()
        IoU_losses.update(IoU_loss.item())
        
        '''save checkpoints'''
        if iteration % cfg['checkpoint_freq'] == 0:
            model_path = os.path.join(save_dir, 'checkpoint_%07d.pth.tar'%iteration)
            temp_path = os.path.join(save_dir,'Temp.pth.tar')
            state = {'model_R':model_R.state_dict(),
                     'optimizer_R':optimizer_R.state_dict(),
                     'model_D':model_D.state_dict(),
                     'optimizer_D':optimizer_D.state_dict(),
                     'iteration': iteration,
                     'losslog':losslog,
                    }
            torch.save(state,model_path)
            torch.save(state,temp_path)
            
        '''decay lr'''
        if iteration >= cfg['Reconstruction']['decay_begining'] and iteration % cfg['Reconstruction']['decay_step'] == 0:
            adjust_learning_rate(optimizer_R, decay_rate=cfg['Reconstruction']['decay_rate'])
            adjust_learning_rate(optimizer_D, decay_rate=cfg['Reconstruction']['decay_rate'])
        
        # save demo images
        if iteration % cfg['demo_freq'] == 0:
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_real.png' % iteration), utils.img_cvt(imgs[0]))
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_fake1.png' % iteration), utils.img_cvt(rgba_output_img[0]))
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_fake2.png' % iteration), utils.img_cvt(rgba_output_img2[0]))
#            srf.save_obj(os.path.join(save_dir,'demo', '%07d_fake.obj' % iteration), vertices[0], faces[0])
        if iteration % cfg['print_freq'] == 0:
            print('Train_I: [{0}/{1}]\t'
                               'IoU_loss {loss0.val:.3f} ({loss0.avg:.3f})\t'
                               'Color_loss {loss1.val:.3f} ({loss1.avg:.3f})\t'
                               'W_dist {loss2.val:.3f} ({loss2.avg:.3f})\t'.format(iteration, cfg['Reconstruction']['iterations'],
                                                                                loss0=IoU_losses,loss1=mixed_color_losses,loss2=w_distances))
    return

def train_view_aware_reconstructor(model_vp,vp_load_dir,model_R,optimizer_R,model_D,optimizer_D,load_dir,save_dir,cfg,dataset):
    
    batch_size = cfg['Reconstruction']['batch_size']
    color_rec = cfg['Reconstruction']['color_rec']    
    if vp_load_dir:
        checkpoint_vp = torch.load(vp_load_dir)
        model_vp.load_state_dict(checkpoint_vp['model_vp'])
    else:
        raise Exception('cant find checkpoint')
        
    if os.path.exists(os.path.join(load_dir,'Temp.pth.tar')):
        checkpoint = torch.load(os.path.join(load_dir,'Temp.pth.tar'))
        model_R.load_state_dict(checkpoint['model_R'])
        optimizer_R.load_state_dict(checkpoint['optimizer_R'])
        model_D.load_state_dict(checkpoint['model_D'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        start_iteration = checkpoint['iteration']+1
        losslog = checkpoint['losslog']
        IoU_losses = losslog['IoU_losses']
        mixed_color_losses = losslog['mixed_color_losses']
        d_real = losslog['d_real']
        d_fake = losslog['d_fake']
        g_fake = losslog['g_fake']
    else:
        start_iteration = 0
        IoU_losses = utils.AverageMeter()
        mixed_color_losses = utils.AverageMeter()
        d_real = utils.AverageMeter()
        d_fake = utils.AverageMeter()
        g_fake = utils.AverageMeter()
        losslog = {'IoU_losses':IoU_losses,'mixed_color_losses':mixed_color_losses,'d_real':d_real,'d_fake':d_fake,'g_fake':g_fake}

    if not os.path.exists(os.path.join(save_dir,'demo')):
        os.makedirs(os.path.join(save_dir,'demo'))

    #soft_label
    y_real_ = torch.ones(1)
    y_fake_ = torch.zeros(1)
    
    y_real_ = y_real_.to(device)
    y_fake_ = y_fake_.to(device)
    criterion = nn.BCELoss().to(device)
    for iteration in range(start_iteration,cfg['Reconstruction']['iterations']):
        imgs,_,_ = dataset.get_random_single_batch(batch_size)
        imgs = imgs.to(device)    
        
        # predict views
        views = torch.zeros(3,batch_size)
        views = views.to(device)
        views[0] = 2.732
        views[1] = 30
        if cfg['Reconstruction']['use_vp']:
            views[2] = model_vp(imgs[:,3,:,:]).view(batch_size)   
        else:
            views[2] = 0
        '''-----------train D-----------'''
        for p in model_D.parameters(): 
            p.requires_grad = True        
        
        '''perform reconstruction'''
        vertices, faces, color = model_R.reconstruct(imgs)
        viewpoints = srf.get_points_from_angles(views[0],views[1],views[2])
        model_R.set_sigma(cfg['SoftRender']['SIGMA_VAL'])
        model_R.renderer.transform.set_eyes(viewpoints)
        
        rgba_output_img = model_R.renderer(vertices, faces, color)
        alpha_output_img=rgba_output_img[:,3,:,:]
        
        '''1:train with real img (from predicted input view)'''
        if cfg['Reconstruction']['ViewGan']['dis_domain'] == 'img':
            real_img = imgs[:,3,:,:] # only input silhouettes
        elif cfg['Reconstruction']['ViewGan']['dis_domain'] == 'view':
            real_img = alpha_output_img
            
        # discriminate         
        D_real_result = model_D(real_img,viewpoints)
        D_real_result = D_real_result.mean()
#        D_real_result.backward(mone)  
        
        '''2:train with fake img (from other random view)'''
        random_view = torch.rand(views[2].shape) * 360
        random_view = random_view.to(device)
        viewpoints2 = srf.get_points_from_angles(views[0],views[1],random_view)
        model_R.renderer.transform.set_eyes(viewpoints2)
        rgba_fake_img = model_R.renderer(vertices, faces, color)
        fake_img=rgba_fake_img[:,3,:,:]
        
        # discriminate
        D_fake_result = model_D(fake_img,viewpoints2)  
        D_fake_result = D_fake_result.mean()
#        D_fake_result.backward(one)
        
        
        
        '''3:compute D loss'''
        d_real_loss = criterion(D_real_result,y_real_)
        d_fake_loss = criterion(D_fake_result,y_fake_)
        d_loss = d_real_loss + d_real_loss
        
        d_real.update(d_real_loss.item())
        d_fake.update(d_fake_loss.item())
        
        optimizer_D.zero_grad()
        d_loss.backward()        
        optimizer_D.step() 
    
        
        '''-----------train R(G)-----------'''
        LAMBDA_FLATTEN = cfg['Reconstruction']['LAMBDA_FLATTEN']
        LAMBDA_LAPLACIAN = cfg['Reconstruction']['LAMBDA_LAPLACIAN']
        LAMBDA_DISC = cfg['Reconstruction']['LAMBDA_DISC']
        if color_rec:
            LAMBDA_COLOR = cfg['Reconstruction']['LAMBDA_COLOR']
        else:
            LAMBDA_COLOR = 0

        imgs,_,_ = dataset.get_random_single_batch(batch_size)
        imgs = imgs.to(device)    
        # predict views
        views = torch.zeros(3,batch_size)
        views = views.to(device)
        views[0] = 2.732
        views[1] = 30
        if cfg['Reconstruction']['use_vp']:
            views[2] = model_vp(imgs[:,3,:,:]).view(batch_size)    
        else:
            views[2] = 0
        # perform reconstruction
        vertices, faces, color = model_R.reconstruct(imgs)
        viewpoints = srf.get_points_from_angles(views[0],views[1],views[2])
        model_R.set_sigma(cfg['SoftRender']['SIGMA_VAL'])
        model_R.renderer.transform.set_eyes(viewpoints)
        
        rgba_output_img = model_R.renderer(vertices, faces, color)
        alpha_output_img=rgba_output_img[:,3,:,:]
        
        # geometry regulizer loss
        laplacian_loss,flatten_loss = model_R.geometry_loss(vertices)   
        laplacian_loss = laplacian_loss.mean()
        flatten_loss = flatten_loss.mean()
        # reprojection loss
        IoU_loss,pixel_color_loss = loss.reprojection_loss(rgba_output_img,imgs)
        IoU_loss=IoU_loss.mean()
        pixel_color_loss=pixel_color_loss.mean()
        # adversial loss
        random_view = torch.rand(views[2].shape) * 360
        random_view = random_view.to(device)            
        viewpoints2 = srf.get_points_from_angles(views[0],views[1],random_view)
        model_R.renderer.transform.set_eyes(viewpoints2)
        rgba_output_img2 = model_R.renderer(vertices, faces, color)
        
        if iteration % cfg['Reconstruction']['D_G_rate'] == 0:
            fake_img = rgba_output_img2[:,3,:,:]
            G_fake_result = model_D(fake_img,viewpoints2)
            G_fake_result = G_fake_result.mean()
            g_fake_loss = criterion(G_fake_result,y_real_)
            R_loss = flatten_loss*LAMBDA_FLATTEN + laplacian_loss*LAMBDA_LAPLACIAN + g_fake_loss*LAMBDA_DISC + IoU_loss
            g_fake.update(g_fake_loss.item()) 
        else:
            R_loss = flatten_loss*LAMBDA_FLATTEN + laplacian_loss*LAMBDA_LAPLACIAN + IoU_loss
#        
        if iteration > cfg['Reconstruction']['color_loss_begin']:
            R_loss += pixel_color_loss*LAMBDA_COLOR
#        R_loss+=pixel_color_loss*LAMBDA_COLOR
        optimizer_R.zero_grad()
        R_loss.backward()
        optimizer_R.step()
        IoU_losses.update(IoU_loss.item())
        mixed_color_losses.update(pixel_color_loss.item())
        
        '''save checkpoints'''
        if iteration % cfg['checkpoint_freq'] == 0:
            model_path = os.path.join(save_dir, 'checkpoint_%07d.pth.tar'%iteration)
            temp_path = os.path.join(save_dir,'Temp.pth.tar')
            state = {'model_R':model_R.state_dict(),
                     'optimizer_R':optimizer_R.state_dict(),
                     'model_D':model_D.state_dict(),
                     'optimizer_D':optimizer_D.state_dict(),
                     'iteration': iteration,
                     'losslog':losslog,
                    }
            torch.save(state,model_path)
            torch.save(state,temp_path)
            
        '''decay lr'''
        if iteration >= cfg['Reconstruction']['decay_begining'] and iteration % cfg['Reconstruction']['decay_step'] == 0:
            adjust_learning_rate(optimizer_R, decay_rate=cfg['Reconstruction']['decay_rate'])
            adjust_learning_rate(optimizer_D, decay_rate=cfg['Reconstruction']['decay_rate'])
        
        # save demo images
        if iteration % cfg['demo_freq'] == 0:
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_real.png' % iteration), utils.img_cvt(imgs[0]))
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_fake1.png' % iteration), utils.img_cvt(rgba_output_img[0]))
            imageio.imsave(os.path.join(save_dir,'demo', '%07d_fake2.png' % iteration), utils.img_cvt(rgba_output_img2[0]))
#            srf.save_obj(os.path.join(save_dir,'demo', '%07d_fake.obj' % iteration), vertices[0], faces[0])
        if iteration % cfg['print_freq'] == 0:
            print('Train_I: [{0}/{1}]\t'
                               'IoU_loss {loss0.val:.3f} ({loss0.avg:.3f})\t'
                               'd_fake_loss {loss1.val:.3f} ({loss1.avg:.3f})\t'
                               'g_fake_loss {loss2.val:.3f} ({loss2.avg:.3f})\t'.format(iteration, cfg['Reconstruction']['iterations'],
                                                                                loss0=IoU_losses,loss1=d_fake,loss2=g_fake))
            