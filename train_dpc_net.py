import random
import numpy as np
import pickle
import os
import time
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision import transforms

from liegroups import SE3, SO3

from utils import *
from lie_algebra import se3_inv

from net import *
from losses import *

def main():
    test_trials = ['00','02','05'] #KITTI Trials
    correction_type = 'pose'  #yaw, rotation, or pose
    for t_i, test_trial in enumerate(test_trials):
        train_sequence(test_trial, correction_type)
    
def train_sequence(test_seq, correction_type):

    system_config = {'use_cuda': True, 
            'validate_output_interval': 20, 
            'train_output_interval': 50,
            'num_loader_workers': 12,
            'base_folder': '/media/raid5-array/experiments/Deep-PC/' #Needs trailing slash
    }

    # define the CNN and move the network into GPU
    train_config = {'kitti_test_seq': test_seq,
                    'batch_size': 64,
                    'num_epochs': 25,
                    'optimizer': 'Adam',    
                    'img_type': 'rgb', #rgb or mono
                    'lr': 1e-4,
                    'step_size': 5,
                    'correction_type': correction_type,
                    'save_every_nth_epoch': 5, #Save all nth epochs (set to None if you want to only save the best validation error epoch)
                    'gamma': 1,
                    'resize_factor': 0.5,
                    'img_dims': [400, 120], #This takes precedence over resize_factor. Set to None if you'd like to keep the original aspect ratio.
                    'trained_models_dir': system_config['base_folder'] + 'trained_models/' + correction_type,
                    'img_transforms': {
                        'mean': [0.485, 0.456, 0.406], #These are ImageNet normalizations
                        'std': [0.229, 0.224, 0.225]
                    }
    }


    kitti_config = {'tm_path': system_config['base_folder'] + '/results', #Where to store new results,
                    'kitti_data_file':  system_config['base_folder'] + '/training_data/kitti_pose_error_data_test_{}.pickle'.format(train_config['kitti_test_seq'])
    }

    if not train_config['img_dims']:
        resized_img_dims = [round(1226*train_config['resize_factor']), round(370*train_config['resize_factor'])]
    else:
        resized_img_dims = train_config['img_dims']

    
    print('Loading {}'.format(kitti_config['kitti_data_file']))
    print('Resizing images to {}'.format(resized_img_dims))


    #Scale and transform images
    transform_img = transforms.Compose([
        transforms.Resize(resized_img_dims), 
        transforms.ToTensor(),
        transforms.Normalize(mean=train_config['img_transforms']['mean'],
                                    std=train_config['img_transforms']['std']) 
    ])


    train_loader = DataLoader(KITTIOdometryDataset(kitti_config['kitti_data_file'], img_type=train_config['img_type'], transform_img=transform_img, run_type='train'), 
                            batch_size=train_config['batch_size'],
                            shuffle=True, num_workers=system_config['num_loader_workers'])

    #Make sure shuffle is off so we can run our ARMSE calculations!
    valid_loader = DataLoader(KITTIOdometryDataset(kitti_config['kitti_data_file'], img_type=train_config['img_type'], transform_img=transform_img, run_type='validate'), 
                        batch_size=train_config['batch_size'],
                        shuffle=False, num_workers=system_config['num_loader_workers'])

    if correction_type == 'pose':
        pose_corrector_net = DeepPoseCorrectorStereoFullPose()
        loss_fn = SE3GeodesicLoss()
        precision = train_loader.dataset.train_se3_precision

    elif correction_type == 'rotation':
        pose_corrector_net = DeepPoseCorrectorMonoRotation()
        loss_fn = SO3GeodesicLoss()
        precision = train_loader.dataset.train_se3_precision[3:6, 3:6].contiguous()

    elif correction_type == 'yaw':
        pose_corrector_net = DeepPoseCorrectorMonoYaw()
        loss_fn = nn.MSELoss()
        precision = torch.eye(1).float()


    if system_config['use_cuda']:
        pose_corrector_net.cuda()
        loss_fn = loss_fn.cuda()
        
    #This tells cudnn to search for the most efficient convolutional algorithms (i.e. voodoo magic that may make things faster)
    cudnn.benchmark = True

    # define the loss (criterion) and create an optimizer

    if train_config['optimizer'] == 'Adam':
        optimizer = optim.Adam(pose_corrector_net.parameters(), lr=train_config['lr'])
    elif train_config['optimizer'] == 'SGD':
        optimizer = optim.SGD(pose_corrector_net.parameters(), lr=train_config['lr'], momentum=0.9)
    else:
        raise ValueError('Optimizer must be set to `Adam` or `SGD`.')


    scheduler = StepLR(optimizer, train_config['step_size'], gamma=train_config['gamma'])
    
    end = time.time()
    best_valid_loss = 10
    for epoch in range(1, train_config['num_epochs'] + 1):
        
        scheduler.step()
        
        if epoch == 1:
            avg_valid_loss = validate(valid_loader, pose_corrector_net, loss_fn, precision, system_config, correction_type)
            print('Initial validation loss: {:.2E}'.format(avg_valid_loss))

        print('Training for test seq. {}. Starting epoch {} / {}. Learning rate set to: {}.'.format(train_config['kitti_test_seq'], epoch, train_config['num_epochs'], scheduler.get_lr()))
        train(epoch, pose_corrector_net, train_loader, optimizer, loss_fn, precision, system_config, correction_type)

        avg_valid_loss = validate(valid_loader, pose_corrector_net, loss_fn, precision, system_config, correction_type)
        print('Validation completed in: {:.2f}. Current avg. validation loss: {:.2E}'.format(time.time() - end, avg_valid_loss))

        is_best = avg_valid_loss < best_valid_loss

        if is_best:
            best_valid_loss = avg_valid_loss
            print('New best validation loss!')

        #This will only save if required (i.e. every_N or if the validation loss is the lowest recorded)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': pose_corrector_net.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'precision' : precision,
            'train_config': train_config,
            'system_config': system_config,
            'kitti_config': kitti_config,
            'avg_valid_loss': avg_valid_loss
        }, is_best=is_best,  save_path=train_config['trained_models_dir'], epoch=epoch, seq=train_config['kitti_test_seq'], save_every_N=train_config['save_every_nth_epoch'])

        print('Epoch complete. Total epoch time: {:.2f}'.format(time.time() - end))
        end = time.time()

        

def train(epoch, model, train_loader, optimizer, loss_fn, precision, config, correction_type):
    print('Training...')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()

    for batch_idx, (image_quad, target_rot, target_yaw, target_se3) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        batch_size = image_quad[0].size(0)
        # forward pass
        if correction_type == 'pose':
            loss, output = compute_loss(image_quad, target_se3, model, loss_fn, precision, config, mode='train')
        elif correction_type == 'rotation':
            loss, output = compute_loss_rot(image_quad, target_rot, model, loss_fn, precision, config, mode='train')
        elif correction_type == 'yaw':
            loss, output = compute_loss_yaw(image_quad, target_yaw, model, loss_fn, precision, config, mode='train')
        
        #record loss
        losses.update(loss.data[0], batch_size)

        #backward pass
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % config['train_output_interval'] == 0 or batch_idx == len(train_loader) - 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} (avg: {data_time.avg:.3f})\t'
                  'Loss {loss.val:.2E} (avg: {loss.avg:.2E})\t'.format(
                   epoch, batch_idx, len(train_loader) - 1, batch_time=batch_time,
                   data_time=data_time, loss=losses))



def validate(valid_loader, model, loss_fn, precision, config, correction_type):
    
    val_seq = valid_loader.dataset.sequence
    print('Validating with sequence {}...'.format(val_seq))

    batch_time = AverageMeter()
    losses = AverageMeter()
    num_quads = len(valid_loader.dataset)

    if correction_type == 'rotation':
        D = 3
    elif correction_type == 'pose':
        D = 6
    else:
        D = 1
    
    predictions = torch.FloatTensor(num_quads, D).zero_()
    targets = torch.FloatTensor(num_quads, D).zero_()


    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    start_idx = 0
    for batch_idx, (image_quad, target_rot, target_yaw, target_se3) in enumerate(valid_loader):
        #forward pass
        batch_size = image_quad[0].size(0)

        if correction_type == 'pose':
            loss, output = compute_loss(image_quad, target_se3, model, loss_fn, precision, config, mode='eval')
            targets[start_idx:start_idx+batch_size, :] = se3_log(target_se3)
            
        elif correction_type == 'rotation':
            loss, output = compute_loss_rot(image_quad, target_rot, model, loss_fn, precision, config, mode='eval')
            targets[start_idx:start_idx+batch_size, :] = so3_log(target_rot)
        
        elif correction_type == 'yaw':
            loss, output = compute_loss_yaw(image_quad, target_yaw, model, loss_fn, precision, config, mode='eval')
            targets[start_idx:start_idx+batch_size, :] = target_yaw

        #store predicitions
        predictions[start_idx:start_idx+batch_size, :] = output.data
        start_idx += batch_size 
        losses.update(loss.data[0], batch_size)

        
        # measure elapsed time        
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % config['validate_output_interval'] == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                  'Loss {loss.val:.2E} (avg: {loss.avg:.2E})\t'.format(
                   batch_idx, len(valid_loader), batch_time=batch_time, loss=losses))
    
    compute_corrected_stats(valid_loader.dataset.tm_mat_path,  predictions.numpy(), targets.numpy(), valid_loader.dataset.test_pose_delta, corr_type=correction_type, eval_type='validation')
    return losses.avg

if __name__ == '__main__':
    main()
