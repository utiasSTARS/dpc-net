import pickle, csv, glob, os
import shutil
import pykitti
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from lie_algebra import se3_log, so3_log
from liegroups.numpy import SE3, SO3
from pyslam.metrics import TrajectoryMetrics


class KITTIData(object):
    def __init__(self):
        
        self.train_sequences = []
        self.train_img_paths = []
        self.train_labels = []
        self.train_se3_precision = []
        
        self.val_sequence = ''
        self.val_tm_mat_path = '' #Path to mat file containing the the trajectory (loaded by TrajectoryMetrics)
        self.val_img_paths = []
        self.val_labels = []
        
        self.test_sequence = ''
        self.test_tm_mat_path = ''
        self.test_img_paths = []
        self.test_labels = []
        

global KITTI_SEQS_DICT
KITTI_SEQS_DICT = {'00': {'date': '2011_10_03',
            'drive': '0027',
            'frames': range(0, 4541)},
        '01': {'date': '2011_10_03',
            'drive': '0042',
            'frames': range(0, 1101)},
        '02': {'date': '2011_10_03',
            'drive': '0034',
            'frames': range(0, 4661)},
        '04': {'date': '2011_09_30',
            'drive': '0016',
            'frames': range(0, 271)},
        '05': {'date': '2011_09_30',
            'drive': '0018',
            'frames': range(0, 2761)},
        '06': {'date': '2011_09_30',
            'drive': '0020',
            'frames': range(0, 1101)},
        '07': {'date': '2011_09_30',
            'drive': '0027',
            'frames': range(0, 1101)},
        '08': {'date': '2011_09_30',
            'drive': '0028',
            'frames': range(1100, 5171)},
        '09': {'date': '2011_09_30',
            'drive': '0033',
            'frames': range(0, 1591)},
        '10': {'date': '2011_09_30',
            'drive': '0034',
            'frames': range(0, 1201)}}

class KITTIOdometryDataset(Dataset):
    """KITTI Odometry Benchmark dataset."""

    def __init__(self, kitti_data_pickle_file, img_type='rgb', transform_img=None, run_type='train'):
        """
        Args:
            kitti_data_pickle_file (string): Path to saved kitti dataset pickle.
            run_type (string): 'train', 'validate', or 'test'.
            transform_img (callable, optional): Optional transform to be applied to images.
        """
        self.pickle_file = kitti_data_pickle_file
        self.transform_img = transform_img
        self.img_type = img_type
        self.load_kitti_data(run_type) #Loads self.image_quad_paths and self.labels

    def load_kitti_data(self, run_type):
        with open(self.pickle_file, 'rb') as handle:
            kitti_data = pickle.load(handle)

        #Empirical precision matrix (inverse covariance) computed over the training data
        self.train_se3_precision = torch.from_numpy(kitti_data.train_se3_precision).float()
        self.train_pose_deltas = kitti_data.train_pose_deltas
        self.test_pose_delta = kitti_data.test_pose_delta

        if run_type == 'train':

            self.image_quad_paths = kitti_data.train_img_paths_rgb if self.img_type=='rgb' else kitti_data.train_img_paths_mono
            self.T_corr = kitti_data.train_T_corr
            self.T_gt = kitti_data.train_T_gt
            self.T_est = kitti_data.train_T_est
            self.sequences = kitti_data.train_sequences
            
        elif run_type == 'validate' or run_type == 'valid':
            self.image_quad_paths = kitti_data.val_img_paths_rgb if self.img_type=='rgb' else kitti_data.val_img_paths_mono
            self.T_corr = kitti_data.val_T_corr
            self.T_gt = kitti_data.val_T_gt
            self.T_est = kitti_data.val_T_est
            self.sequence = kitti_data.val_sequence
            self.tm_mat_path = kitti_data.val_tm_mat_path

        elif run_type == 'test':    
            self.image_quad_paths = kitti_data.test_img_paths_rgb if self.img_type=='rgb' else kitti_data.test_img_paths_mono
            self.T_corr = kitti_data.test_T_corr
            self.T_gt = kitti_data.test_T_gt
            self.T_est = kitti_data.test_T_est
            self.sequence = kitti_data.test_sequence
            self.tm_mat_path = kitti_data.test_tm_mat_path

        else:
            raise ValueError('run_type must be set to `train`, `validate` or `test`. ')


    def __len__(self):
        return len(self.image_quad_paths)

    def read_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return img

    def __getitem__(self, idx):
        #Get all four images in the two pairs
        image_quad_paths = self.image_quad_paths[idx]
        target_se3 = torch.from_numpy(self.T_corr[idx].as_matrix()).float()
        target_rot = torch.from_numpy(self.T_corr[idx].rot.as_matrix()).float()
        #Note: The camera y axis is facing down, hence 'yaw' of the vehicle, is 'pitch' of the camera
        target_yaw = torch.Tensor([self.T_gt[idx].rot.to_rpy()[1] - self.T_est[idx].rot.to_rpy()[1]]).float()

        if self.transform_img:
            image_quad = [self.transform_img(self.read_image(image_quad_paths[i])) for i in range(4)]
        else:
            image_quad = [self.read_image(image_quad_paths[i]) for i in range(4)]

        return image_quad, target_rot, target_yaw, target_se3


class KITTIOdometryDatasetTargetsOnly(Dataset):
    """KITTI Odometry Benchmark dataset."""

    def __init__(self, kitti_data_pickle_file, run_type='train'):
        """
        Args:
            kitti_data_pickle_file (string): Path to saved kitti dataset pickle.
            run_type (string): 'train', 'validate', or 'test'.
            transform_img (callable, optional): Optional transform to be applied to images.
        """
        self.pickle_file = kitti_data_pickle_file
        self.load_kitti_data(run_type) #Loads self.image_quad_paths and self.labels

    def load_kitti_data(self, run_type):
        with open(self.pickle_file, 'rb') as handle:
            kitti_data = pickle.load(handle)

        if run_type == 'train':
            self.labels = kitti_data.train_labels
        elif run_type == 'validate' or run_type == 'valid':
            self.labels = kitti_data.val_labels
        elif run_type == 'test':
            self.labels = kitti_data.test_labels
        else:
            raise ValueError('run_type must be set to `train`, `validate` or `test`. ')
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #Get all four images in the two pairs
        target_se3 = torch.from_numpy(self.labels[idx].as_matrix()).float()
        return target_se3


def compute_precision(train_loader, type='pose'):
    print('Computing covariance...')
    num_quads = len(train_loader.dataset)
    if type == 'pose':
        targets = torch.FloatTensor(num_quads, 6).zero_()
    else:
        targets = torch.FloatTensor(num_quads, 3).zero_()

    start_idx = 0
    for batch_idx, target_se3 in enumerate(train_loader):
        batch_size = target_se3.size(0)

        if type == 'pose':
            targets[start_idx:start_idx+batch_size, :] = se3_log(target_se3)
        else:
            targets[start_idx:start_idx+batch_size, :] = so3_log(target_se3[:, 0:3, 0:3])

        start_idx += batch_size 
        #print('Batch: {}/{}'.format(batch_idx, len(train_loader)))

    precision = torch.from_numpy(np.linalg.inv(np.cov(targets.numpy().T))).float()
    print('Done! Precision:')
    print(precision)

    return precision


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, save_path, epoch, seq, save_every_N=None):

    file_name = os.path.join(save_path, 'seq_{}_epoch_{}'.format(seq, epoch))
    
    if save_every_N and epoch % save_every_N == 0:
        torch.save(state, file_name + '.pth.tar')
        if is_best:
            #Remove past best files and create a copy of the best file
            for fl in glob.glob(os.path.join(save_path, 'seq_{}_epoch_*_best*'.format(seq))):
                os.remove(fl)
            shutil.copy(file_name + '.pth.tar', file_name + '_best.pth.tar')        
    else:
        if is_best:
            #Remove past best files and save this best file
            for fl in glob.glob(os.path.join(save_path, 'seq_{}_epoch_*_best*'.format(seq))):
                os.remove(fl)
            torch.save(state, file_name + '_best.pth.tar')



def compute_corrected_stats(tm_mat_path, predictions, targets, p_idx_delta, corr_type='rotation', output_tm_mat_path=None, eval_type='validation'):
    #Load original trajectory

    tm_orig = TrajectoryMetrics.loadmat(tm_mat_path)

    Twv_est_corr = []
    Twv_est_opt = []
    Twv_est_opt.append(tm_orig.Twv_gt[0])
    Twv_est_corr.append(tm_orig.Twv_gt[0]) #Append the first pose
    
    T_12_est_corr_hist = []
    T_12_est_gt_hist = []
    T_12_est_hist = []
    
    
    num_poses = len(tm_orig.Twv_gt)
    
    #c_idx is a correction id, p_idx is the pose id
    
    #final_loss = 0


    for c_idx, p_idx in enumerate(range(0, num_poses - p_idx_delta, p_idx_delta)):
        T_12_est = tm_orig.Twv_est[p_idx].inv().dot(tm_orig.Twv_est[p_idx + p_idx_delta])
        T_12_gt = tm_orig.Twv_gt[p_idx].inv().dot(tm_orig.Twv_gt[p_idx + p_idx_delta])
        

        #During test time we only correct at every p_idx_delta'th interval, and so there are less predictions
        # if eval_type == 'validation':
        #     pred_idx = p_idx
        # else:
        #     pred_idx = c_idx
        pred_idx = c_idx

        #Loss verification
        #T_corr = T_12_gt.dot(T_12_est.inv())
        #log = SE3.exp(predictions[pred_idx]).dot(T_corr.inv()).log()
        #final_loss += (0.5*log.dot(log) - 0.5*T_corr.inv().log().dot(T_corr.inv().log()))

        if corr_type == 'rotation':
            #Correct full rotation matrix
            trans = T_12_est.trans
            rot = T_12_est.rot

            corr_mat = SO3.exp(predictions[pred_idx])
            #corr_mat.normalize()
            
            rot_corr = corr_mat.dot(rot)

            rot_opt = SO3.exp(targets[pred_idx]).dot(rot)
            trans_opt = trans_corr = trans

            T_12_est_corr = SE3(trans=trans_corr, rot=rot_corr)
            T_12_est_corr_opt = SE3(trans=trans_opt, rot=rot_opt)

        elif corr_type == 'trans':
            #Correct translation only
            trans = T_12_est.trans
            rot = T_12_est.rot

            rot_corr = rot_opt = rot
            trans_corr = trans + predictions[pred_idx]
            trans_opt = trans + targets[pred_idx]

            T_12_est_corr = SE3(trans=trans_corr, rot=rot_corr)
            T_12_est_corr_opt = SE3(trans=trans_opt, rot=rot_opt)

        elif corr_type == 'yaw':
            #Correct yaw only
            trans = T_12_est.trans
            rot = T_12_est.rot

            #Note: The camera y axis is facing down, hence 'yaw' of the vehicle, is 'pitch' of the camera
            rpy_orig = SO3.to_rpy(rot)
            yaw_corr = rpy_orig[1] + predictions[pred_idx, 0]
            yaw_opt = rpy_orig[1] + targets[pred_idx, 0]

            rot_corr = SO3.from_rpy(rpy_orig[0], yaw_corr, rpy_orig[2])
            rot_opt = SO3.from_rpy(rpy_orig[0], yaw_opt, rpy_orig[2])
    
            trans_opt = trans_corr = trans

            T_12_est_corr = SE3(trans=trans_corr, rot=rot_corr)
            T_12_est_corr_opt = SE3(trans=trans_opt, rot=rot_opt)

        elif corr_type == 'pose':
            #Correct full pose
            T_corr = SE3.exp(predictions[pred_idx])
            T_corr_opt = SE3.exp(targets[pred_idx])
            
            T_corr.normalize()
            T_12_est_corr = T_corr.dot(T_12_est)
            T_12_est_corr_opt = T_corr_opt.dot(T_12_est)

        else:
            raise ValueError('corr_type must be set to `rot` or `trans` ')
        
        for p_jdx in range(p_idx, p_idx + p_idx_delta - 1):
            T_12_est_single = tm_orig.Twv_est[p_jdx].inv().dot(tm_orig.Twv_est[p_jdx + 1])
            Twv_est_opt.append(Twv_est_opt[p_jdx].dot(T_12_est_single))
            Twv_est_corr.append(Twv_est_corr[p_jdx].dot(T_12_est_single))

        T_12_est_corr_hist.append(T_12_est_corr)
        T_12_est_hist.append(T_12_est)
        T_12_est_gt_hist.append(T_12_gt)

        Twv_est_corr.append(Twv_est_corr[p_idx].dot(T_12_est_corr))
        Twv_est_opt.append(Twv_est_opt[p_idx].dot(T_12_est_corr_opt))

    #Add final poses if p_idx_delta does not divide num_poses - 1 evenly
    for p_jdx in range(len(Twv_est_corr) - 1, num_poses - 1):
        T_12_est_single = tm_orig.Twv_est[p_jdx].inv().dot(tm_orig.Twv_est[p_jdx + 1])
        Twv_est_opt.append(Twv_est_opt[p_jdx].dot(T_12_est_single))
        Twv_est_corr.append(Twv_est_corr[p_jdx].dot(T_12_est_single))
    
    tm_corr = TrajectoryMetrics(tm_orig.Twv_gt, Twv_est_corr, convention='Twv')
    tm_corr_opt = TrajectoryMetrics(tm_orig.Twv_gt, Twv_est_opt, convention='Twv')
    
    tm_corr_only = TrajectoryMetrics(T_12_est_gt_hist, T_12_est_corr_hist, convention='Twv')
    tm_orig_delta = TrajectoryMetrics(T_12_est_gt_hist, T_12_est_hist, convention='Twv')



    trans_log, rot_log = mean_log_square(tm_orig_delta)
    trans_log_corr, rot_log_corr = mean_log_square(tm_corr_only)
       
    trans_err_norm, rot_err_norm = tm_orig.mean_err(error_type='traj')
    trans_err_norm_corr, rot_err_norm_corr = tm_corr.mean_err(error_type='traj')
    

    seg_lengths = list(range(100,801,100))
    _, avg_seg_errs_corr = tm_corr.segment_errors(seg_lengths, rot_unit='deg')
    _, avg_seg_errs_corr_opt = tm_corr_opt.segment_errors(seg_lengths, rot_unit='deg')
    _, avg_seg_errs_orig = tm_orig.segment_errors(seg_lengths, rot_unit='deg')


    if output_tm_mat_path:
        print('Saving corrected TrajectoryMetrics to: {}'.format(output_tm_mat_path))
        tm_corr.savemat(output_tm_mat_path)

        output_corr_path = output_tm_mat_path.split('.mat')[0] + '_corr_only_p_delta_{}.mat'.format(p_idx_delta)

        print('Saving test corrections and predictions to: {}'.format(output_corr_path))
        tm_corr_only.savemat(output_corr_path)
    
    
    print('Baseline Log Squared Norm (Trans / Rot): {:.5f} (m) / {:.8f} (a-a)'.format(trans_log, rot_log))
    print('Corrected Log Squared Norm (Trans / Rot): {:.5f} (m) / {:.8f} (a-a)'.format(trans_log_corr, rot_log_corr))

    print('Baseline Mean Norm (Trans / Rot): {:.5f} (m) / {:.5f} (a-a)'.format(trans_err_norm, rot_err_norm))
    print('Corrected Mean Norm (Trans / Rot): {:.5f} (m) / {:.5f} (a-a)'.format(trans_err_norm_corr, rot_err_norm_corr))

    print('Baseline Seg Length Err (Trans / Rot): {:.5f} (%) / {:.5f} (deg/m)'.format(100*np.mean(avg_seg_errs_orig[:,1]), np.mean(avg_seg_errs_orig[:,2])))
    print('Corrected Seg Length Err (Trans / Rot): {:.5f} (%) / {:.5f} (deg/m)'.format(100*np.mean(avg_seg_errs_corr[:,1]), np.mean(avg_seg_errs_corr[:,2])))
    print('Perfectly Corrected Seg Length Err (Trans / Rot): {:.5f} (%) / {:.5f} (deg/m)'.format(100*np.mean(avg_seg_errs_corr_opt[:,1]), np.mean(avg_seg_errs_corr_opt[:,2])))


    traj_stats = {
        'trans_err_norm_corr': trans_err_norm_corr,
        'rot_err_norm_corr': rot_err_norm_corr,
        'trans_err_norm': trans_err_norm,
        'rot_err_norm': rot_err_norm,
    }
    
    tm_dict = {
        'base': tm_orig,
        'corr_' + corr_type: tm_corr,
        'opt_corr_' + corr_type: tm_corr_opt,
    }

    return traj_stats, tm_dict




def mean_log_square(tm):
    trans_err = []
    rot_err = []
    log_err = []
    mean_trans_loss = 0
    mean_rot_loss = 0
    num_quads = len(tm.Twv_gt)
    for p_idx in range(num_quads):
        rel_pose_delta_gt = tm.Twv_gt[p_idx]
        rel_pose_delta_est = tm.Twv_est[p_idx]

        pose_err = rel_pose_delta_est.dot(rel_pose_delta_gt.inv())
   
        l_t = pose_err.log()[0:3]
        l_r = pose_err.log()[4:6]
        mean_trans_loss += l_t.dot(l_t)
        mean_rot_loss += l_r.dot(l_r)

    return mean_trans_loss/num_quads,  mean_rot_loss/num_quads

def read_dataset(kitti_config, trial_str):
    
    mat_files = glob.glob(kitti_config['tm_path'] + '/*.mat')
    
    for m_f in mat_files:
        #Analyze filename of pickle
        fname = m_f.split('/')[-1]
        date = "_".join(fname.split('_')[0:3])
        drive = fname.split('drive_')[-1].split('.')[0]

        #print("data_path: {}".format(data_path))

        #Find which trial this is based on the filename
        for trial, trial_info in KITTI_SEQS_DICT.items():
            if trial_info['date'] == date and trial_info['drive'] == drive:
                frame_range = trial_info['frames']
                break

        #If this is the trial we want, output the image iterator and targets matrix
        if trial == trial_str:
            #Load the TM object
            traj_metrics= TrajectoryMetrics.loadmat(m_f)
            break
            
    return traj_metrics




