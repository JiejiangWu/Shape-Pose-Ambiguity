import torch
import torch.autograd as autograd
def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def normal_calc_gradient_penalty(netD, real_data, fake_data,cfg,reconstruction):
    if reconstruction:
        batch_size = cfg['Reconstruction']['batch_size']
        lamda = cfg['Reconstruction']['ViewGan']['LAMBDA_WGAN']
    else:
        batch_size = cfg['NormalGan']['batch_size']
        lamda = cfg['NormalGan']['LAMBDA_WGAN']
    device = real_data.device
    real_data = real_data.view(batch_size,-1)
    fake_data = fake_data.view(batch_size,-1)
    #print real_data.size()
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    interpolates = interpolates.view(batch_size,1,cfg['input_size'],cfg['input_size'])
    
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamda
    return gradient_penalty

def conditional_calc_gradient_penalty(netD, real_data, fake_data, conditional_,cfg):
    batch_size = cfg['Reconstruction']['ViewGan']['batch_size']
    device = real_data.device
    real_data = real_data.view(batch_size,-1)
    fake_data = fake_data.view(batch_size,-1)
    #print real_data.size()
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    interpolates = interpolates.view(batch_size,1,cfg['input_size'],cfg['input_size'])
    
    disc_interpolates = netD(interpolates,conditional_)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * cfg['LAMBDA_WGAN']
    return gradient_penalty
