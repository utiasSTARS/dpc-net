import random
import numpy as np
import pickle
import os
import time
import math
import csv

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
#Turn off CuDNN due to some unknown convolution bug causing past models not to work properly
torch.backends.cudnn.enabled = False

from liegroups.numpy import SE3, SO3

from utils import *
from net import *
from losses import *

import argparse
import collections
from pyslam.visualizers import TrajectoryVisualizer

parser = argparse.ArgumentParser(description='PyTorch PoseCorrectorNet Training')
parser.add_argument('--epoch', '-e', default=0, type=int,
                    help='which epoch of the trained model to use, use 0 to select the epoch with the lowest validation error')
parser.add_argument('--seqs', '-s', default='00', type=str,
                    help='which sequence(s) to test (e.g. `00` or `01,02,04`.)')
parser.add_argument('--corr', '-c', default='pose', type=str, 
                    help='correction type (`rotation`, `pose` or `yaw`)')

def main():
    #parse args
    args = parser.parse_args()
    epoch = args.epoch
    correction_type = args.corr

    paths = {
        #Folder that contains the trained models (this folder should contain a subfolder named 'pose' with the trained model. e.g., pose/seq_00_epoch_11_best.pth.tar)
        'trained_models': '/media/raid5-array/experiments/dpc-oss/trained_models/',
        #Where to put the results of the new, corrected (but not pose-graph-relaxed, VO)
        'vo_results': '/media/raid5-array/experiments/dpc-oss/results/',
        #Where are the test VO files? (e.g, kitti_pose_error_data_test_00.pickle, 2011_10_03_drive_0027.mat)
        'kitti_vo': '/media/raid5-array/experiments/dpc-oss/kitti_training_data/', 
        #Where is the base folder for KITTI images? (e.g., If a sample image is '/kitti/2011_10_03/2011_10_03_drive_0027_sync/image_02/data/0000001408.png', supply the path '/kitti/')
        'kitti_img': '/media/m2-drive/datasets/KITTI/raw/'
    }
    
    if correction_type == 'pose':
        trained_models_dir = paths['trained_models'] + 'pose'
    elif correction_type == 'rotation':
       trained_models_dir = paths['trained_models'] + 'rotation'
    elif correction_type == 'yaw':
       trained_models_dir = paths['trained_models'] + 'yaw'
    else:
        raise ValueError('Correction type must be either `rot` or `pose` or `yaw`.')    

    stats_plots_path = os.path.join(trained_models_dir, 'stats')
    #Create the stats and plots directory
    os.makedirs(stats_plots_path, exist_ok=True)

    test_seqs = [seq for seq in args.seqs.split(',')]

    #Run tests and collect stats
    stats_list = []
    for seq in test_seqs:
        test_stats, tm_dict = run_test(seq, epoch, correction_type, 'sparse', paths)
        stats_list.append([
            seq,
            test_stats['avg_test_loss'],
            test_stats['traj_stats']['trans_err_norm'],
            test_stats['traj_stats']['rot_err_norm'],
            test_stats['traj_stats']['trans_err_norm_corr'],
            test_stats['traj_stats']['rot_err_norm_corr']
        ])

        #Export plots
        topdown_filename = os.path.join(stats_plots_path, seq + '_topdown.pdf')
        norm_filename = os.path.join(stats_plots_path, seq + '_norms.pdf')
        cum_filename = os.path.join(stats_plots_path, seq + '_cum_norms.pdf')
        


        tms = [('SVO', tm_dict['base']),
            ('SVO + DPC', tm_dict['corr_'+correction_type])]
        tm_dict = collections.OrderedDict(tms)

        vis = TrajectoryVisualizer(tm_dict)
        vis.plot_topdown(outfile=topdown_filename)
        vis.plot_norm_err(outfile=norm_filename)
        vis.plot_cum_norm_err(outfile=cum_filename)

     

    #Write stats
    csv_filename = os.path.join(stats_plots_path, 'stats.csv')
    with open(csv_filename, "w") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["Seq", "Avg Test Loss", "Baseline Mean Trans Norm", "Baseline Mean Rot Norm","Corrected Mean Trans Norm", "Corrected Mean Rot Norm"])
        writer.writerows(stats_list)

   


def run_test(kitti_test_seq, epoch, correction_type, est_type, paths):

    est_type_folders = {'sparse': ''}
    est_type_vo_folders = {'sparse': 'corrected'}


    if correction_type == 'pose':
        trained_models_dir = paths['trained_models'] + est_type_folders[est_type] + 'pose'
    elif correction_type == 'rotation':
       trained_models_dir = paths['trained_models'] + est_type_folders[est_type] + 'rotation'
    elif correction_type == 'yaw':
       trained_models_dir = paths['trained_models'] + est_type_folders[est_type] + 'yaw'
    else:
        raise ValueError('Correction type must be either `rot` or `pose` or `yaw`.')    
    
    if epoch > 0:
        kitti_test_trained_model_path = os.path.join(trained_models_dir, 'seq_{}_epoch_{}.pth.tar'.format(kitti_test_seq,epoch))
        output_tm_mat_path = paths['vo_results_base'] + '{}/seq_{}_corr_{}_epoch_{}.mat'.format(est_type_vo_folders[est_type],kitti_test_seq, correction_type, epoch)
    else:
        #Find the best epoch
        kitti_test_trained_model_path = glob.glob(os.path.join(trained_models_dir, 'seq_{}_epoch_*_best*'.format(kitti_test_seq)))[0]
        #Extract epoch
        epoch = kitti_test_trained_model_path.split('/')[-1].split('_')[3]
        print('Detected epoch {} as the best one.'.format(epoch))
        output_tm_mat_path = paths['vo_results'] + '{}/seq_{}_corr_{}_epoch_{}.mat'.format(est_type_vo_folders[est_type], kitti_test_seq, correction_type, epoch)

    saved_data = torch.load(kitti_test_trained_model_path)
    train_config = saved_data['train_config']
    system_config = saved_data['system_config']
    kitti_config = saved_data['kitti_config']


    #system_config['use_cuda'] = False
    
    if not train_config['img_dims']:
        resized_img_dims = [round(1226*train_config['resize_factor']), round(370*train_config['resize_factor'])]
    else:
        resized_img_dims = train_config['img_dims']

    #Scale and transform images
    transform_img = transforms.Compose([
        transforms.Resize(resized_img_dims), 
        transforms.ToTensor(),
        transforms.Normalize(mean=train_config['img_transforms']['mean'],
                                    std=train_config['img_transforms']['std']) 
    ])

    
    #Make sure shuffle is off so we can run our error calculations!
    test_loader = DataLoader(KITTIOdometryDataset(kitti_config['kitti_data_file'], img_type=train_config['img_type'], transform_img=transform_img, run_type='test', remap_kitti_folder=paths['kitti_img']), 
                        batch_size=32,
                        shuffle=False, num_workers=6)


    kitti_config['kitti_data_file'] = remap_path(kitti_config['kitti_data_file'], paths['kitti_vo'])
    test_loader.dataset.tm_mat_path = remap_path(test_loader.dataset.tm_mat_path, paths['kitti_vo'])
    

    if correction_type == 'pose':
        pose_corrector_net = DeepPoseCorrectorStereoFullPose()
        loss_fn = SE3GeodesicLoss()
        precision = test_loader.dataset.train_se3_precision
        print('Precision: {}.'.format(precision))

    elif correction_type == 'rotation':
        pose_corrector_net = DeepPoseCorrectorMonoRotation()
        loss_fn = SO3GeodesicLoss()
        precision = test_loader.dataset.train_se3_precision[3:6, 3:6].contiguous()

    elif correction_type == 'yaw':
        pose_corrector_net = DeepPoseCorrectorMonoYaw()
        loss_fn = nn.MSELoss()
        precision = torch.eye(1).float()


    print('Loading trained model: {} ...'.format(kitti_test_trained_model_path))
    pose_corrector_net.load_state_dict(saved_data['state_dict'])
    print('Done.')


    if system_config['use_cuda']:
        pose_corrector_net.cuda()
        loss_fn = loss_fn.cuda()

    #This tells cudnn to search for the most efficient convolutional algorithms (i.e. voodoo magic that may make things faster)
    #cudnn.benchmark = True

    #Test!
    #Return test_stats
    test_stats, tm_dict = test(test_loader, pose_corrector_net, loss_fn, precision, system_config, correction_type, output_tm_mat_path)
    return test_stats, tm_dict



def test(test_loader, model, loss_fn, precision, config, correction_type, output_tm_mat_path=None):
    
    test_seq = test_loader.dataset.sequence
    print('Testing with sequence {}...'.format(test_seq))

    batch_time = AverageMeter()
    losses = AverageMeter()
    num_quads = len(test_loader.dataset)

    print('Num test samples: {}'.format(num_quads))

    corr_dims = {'rotation': 3, 'pose': 6, 'yaw': 1}
    D = corr_dims[correction_type]
    

    predictions = torch.FloatTensor(num_quads, D).zero_()
    targets = torch.FloatTensor(num_quads, D).zero_()


    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    start_idx = 0
    for batch_idx, (image_quad, target_rot, target_yaw, target_se3) in enumerate(test_loader):

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

        if batch_idx % config['validate_output_interval'] == 0 or batch_idx == len(test_loader) - 1:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                  'Loss {loss.val:.2E} (avg: {loss.avg:.2E})\t'.format(
                   batch_idx, len(test_loader) - 1, batch_time=batch_time, loss=losses))
    

    traj_stats, tm_dict = compute_corrected_stats(test_loader.dataset.tm_mat_path, predictions.numpy(), targets.numpy(), test_loader.dataset.test_pose_delta, corr_type=correction_type, output_tm_mat_path=output_tm_mat_path,eval_type='test')
    
    test_stats = {'avg_test_loss': losses.avg,
                  'traj_stats': traj_stats
                }
    
    return test_stats, tm_dict


if __name__ == '__main__':
    main()
