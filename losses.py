import torch
import torch.nn as nn
from lie_algebra import *
from torch.autograd import Variable


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


#See Peretroukhin et al. (ICRA 2018)
class SO3GeodesicLossFn(torch.autograd.Function):

    def forward(self, input, target_C_inv, precision):
        self.save_for_backward(input, target_C_inv, precision)
        num_samples = input.size(0)
        phi_star = so3_log(target_C_inv)

        f_phi = so3_log(so3_exp(input).bmm(target_C_inv))
        
        loss = 0.5*(f_phi.mm(precision).mm(f_phi.t())).trace() - 0.5*(phi_star.mm(precision).mm(phi_star.t())).trace() 
        loss *= (1.0/num_samples)
        #loss = loss.mean()
        return input.new([loss])

    def backward(self, grad_output):
        input, target_C_inv, precision = self.saved_tensors
        batch_size = input.size(0)


        #Potentially cache these logs to speed things up
        f_phi = so3_log(so3_exp(input).bmm(target_C_inv))
        so3_log_jacobs = so3_inv_left_jacobian(f_phi).bmm(so3_left_jacobian(input))
        
        f_phi = f_phi.view(-1,1,3)
        grad_losses = f_phi.bmm(precision.expand_as(so3_log_jacobs)).bmm(so3_log_jacobs)
        grad_loss = grad_losses.view(batch_size, 3)  

        #print('Backwardd pytorch: {}'.format(grad_output.expand_as(grad_loss)*grad_loss))
        #print('Backwardd pytorch, grad_output: {}'.format(grad_output))
        
        #Uncomment if averaging losses in the forward pass
        grad_loss *= (1.0/batch_size)

        #Apply chain rule! 
        out = grad_output.expand_as(grad_loss)*grad_loss
        return out, None, None 


class SO3GeodesicLoss(nn.Module):
    def __init__(self):
        super(SO3GeodesicLoss, self).__init__()
        
    def forward(self, input, target_C_inv, precision):
        _assert_no_grad(target_C_inv)
        _assert_no_grad(precision)
        return SO3GeodesicLossFn()(input, target_C_inv, precision)



#See Peretroukhin et al. (ICRA 2018)
class SE3GeodesicLossFn(torch.autograd.Function):

    def forward(self, input, target_T_inv, precision):

        self.save_for_backward(input, target_T_inv, precision)
        num_samples = input.size(0)

        #print('num_samples: {}'.format(num_samples))
        #print('precision: {}'.format(precision))

        xi_star = se3_log(target_T_inv)
        g_xi = se3_log(se3_exp(input).bmm(target_T_inv))
        
        loss_corr = (0.5/num_samples)*(g_xi.mm(precision).mm(g_xi.t())).trace()
        loss_base = (0.5/num_samples)*(xi_star.mm(precision).mm(xi_star.t())).trace() 
        
        loss = loss_corr - loss_base

        #print('loss_corr: {}'.format(loss_corr))
        #print('loss_base: {}'.format(loss_base))
        #print('loss: {}'.format(loss))
        #print(torch.mean(input, 0))
        #print('g_xi: {}'.format(g_xi))
        return input.new([loss])

    def backward(self, grad_output):
        input, target_T_inv, precision = self.saved_tensors
        batch_size = input.size(0)


        #Potentially cache these logs to speed things up
        logs = se3_log(se3_exp(input).bmm(target_T_inv))
        
        se3_log_jacobs = se3_inv_left_jacobian(logs).bmm(se3_left_jacobian(input))
        
        logs = logs.view(-1,1,6)
        grad_losses = logs.bmm(precision.expand_as(se3_log_jacobs)).bmm(se3_log_jacobs)
        grad_loss = grad_losses.view(batch_size, 6)  

        #print('Backwardd pytorch: {}'.format(grad_output.expand_as(grad_loss)*grad_loss))
        #print('Backwardd pytorch, grad_output: {}'.format(grad_output))
        
        #Uncomment if averaging losses in the forward pass
        grad_loss *= (1.0/batch_size)

        #Apply chain rule! 
        out = grad_output.expand_as(grad_loss)*grad_loss
        return out, None, None 



class SE3GeodesicLoss(nn.Module):
    def __init__(self):
        super(SE3GeodesicLoss, self).__init__()
        
    def forward(self, input, target_T_inv, precision):
        _assert_no_grad(target_T_inv)
        _assert_no_grad(precision)
        return SE3GeodesicLossFn()(input, target_T_inv, precision)





def compute_loss_rot(image_quad, target, model, loss_fn, precision, config, mode='train'):

    if config['use_cuda']:
        if mode == 'eval':
            target_C_inv = Variable(target.transpose(1,2).contiguous().cuda(async=True), volatile=True)
            precision = Variable(precision.cuda(async=True), volatile=True)
            img_1 = Variable(image_quad[0].cuda(), volatile=True)
            img_2 = Variable(image_quad[2].cuda(), volatile=True)
            # stereo_img_1 = Variable(torch.cat((image_quad[0], image_quad[1]), 1).cuda(), volatile=True)
            # stereo_img_2 = Variable(torch.cat((image_quad[2], image_quad[3]), 1).cuda(), volatile=True)
        else:
            target_C_inv = Variable(target.transpose(1,2).contiguous().cuda(async=True))
            precision = Variable(precision.cuda(async=True))
            img_1 = Variable(image_quad[0].cuda())
            img_2 = Variable(image_quad[2].cuda())
            # stereo_img_1 = Variable(torch.cat((image_quad[0], image_quad[1]), 1).cuda())
            # stereo_img_2 = Variable(torch.cat((image_quad[2], image_quad[3]), 1).cuda())

    else:
        target_C_inv = Variable(target.transpose(1,2))
        precision = Variable(precision)
        img_1 = Variable(image_quad[0])
        img_2 = Variable(image_quad[2])
        # stereo_img_1 = Variable(torch.cat((image_quad[0], image_quad[1]), 1))
        # stereo_img_2 = Variable(torch.cat((image_quad[2], image_quad[3]), 1))

    #Compute loss (forward pass)
    output = model(img_1, img_2)
    #output = model(img_1, img_2)
    loss = loss_fn(output, target_C_inv, precision)

    return loss, output


def compute_loss_yaw(image_quad, target, model, loss_fn, precision, config, mode='train'):

    if config['use_cuda']:
        if mode == 'eval':
            target_yaw = Variable(target.cuda(async=True), volatile=True)
            img_1 = Variable(image_quad[0].cuda(), volatile=True)
            img_2 = Variable(image_quad[2].cuda(), volatile=True)
        else:
            target_yaw = Variable(target.cuda(async=True))
            img_1 = Variable(image_quad[0].cuda())
            img_2 = Variable(image_quad[2].cuda())
 
    else:
        target_yaw = Variable(target)
        img_1 = Variable(image_quad[0])
        img_2 = Variable(image_quad[2])

    #Compute loss (forward pass)
    output = model(img_1, img_2)
    #output = model(img_1, img_2)
    loss = loss_fn(output, target_yaw)

    return loss, output


def compute_loss(image_quad, target, model, loss_fn, precision, config, mode='train', debug=False):

    if config['use_cuda']:
        if mode == 'eval':
            target_T_inv = Variable(se3_inv(target).cuda(async=True), volatile=True)
            precision = Variable(precision.cuda(async=True), volatile=True)
            stereo_img_1 = Variable(torch.cat((image_quad[0], image_quad[1]), 1).cuda(), volatile=True)
            stereo_img_2 = Variable(torch.cat((image_quad[2], image_quad[3]), 1).cuda(), volatile=True)
        else:
            target_T_inv = Variable(se3_inv(target).cuda(async=True))
            precision = Variable(precision.cuda(async=True))
            stereo_img_1 = Variable(torch.cat((image_quad[0], image_quad[1]), 1).cuda())
            stereo_img_2 = Variable(torch.cat((image_quad[2], image_quad[3]), 1).cuda())

    else:
        target_T_inv = Variable(se3_inv(target))
        precision = Variable(precision)
        stereo_img_1 = Variable(torch.cat((image_quad[0], image_quad[1]), 1))
        stereo_img_2 = Variable(torch.cat((image_quad[2], image_quad[3]), 1))

    #Compute loss (forward pass)
    output = model(stereo_img_1, stereo_img_2)
    loss = loss_fn(output, target_T_inv, precision)

    if debug:
        print('loss: {}'.format(loss.data[0]))        

    return loss, output