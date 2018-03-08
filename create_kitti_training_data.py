from pyslam.metrics import TrajectoryMetrics
import pickle, csv, glob, os
import random
import numpy as np
from liegroups import SE3
from utils import KITTIData, KITTI_SEQS_DICT



def compute_vo_pose_errors(tm, pose_deltas, eval_type='train', add_reverse=False):
    """Compute delta pose errors on VO estimates """
    T_12_errs = []
    T_12_gts = []
    T_12_ests = []

    for p_delta in pose_deltas:
        if eval_type=='train':
            pose_ids = range(len(tm.Twv_gt) - p_delta)
        elif eval_type=='test':
            pose_ids = range(0, len(tm.Twv_gt) - p_delta, p_delta)
        for p_idx in pose_ids:
            T_12_gt = tm.Twv_gt[p_idx].inv().dot(tm.Twv_gt[p_idx+p_delta])
            T_12_est = tm.Twv_est[p_idx].inv().dot(tm.Twv_est[p_idx+p_delta])
            T_12_corr = T_12_gt.dot(T_12_est.inv())
            T_12_errs.append(T_12_corr)
            T_12_gts.append(T_12_gt)
            T_12_ests.append(T_12_est)

        if add_reverse and eval_type=='train':
            for p_idx in pose_ids:
                T_12_gt = tm.Twv_gt[p_idx].inv().dot(tm.Twv_gt[p_idx+p_delta])
                T_12_est = tm.Twv_est[p_idx].inv().dot(tm.Twv_est[p_idx+p_delta])
                T_21_gt = T_12_gt.inv()
                T_21_est = T_12_est.inv()
                T_21_corr = T_21_gt.dot(T_21_est.inv())
                T_12_errs.append(T_21_corr)
                T_12_gts.append(T_21_gt)
                T_12_ests.append(T_21_est)


    return (T_12_errs, T_12_gts, T_12_ests)

def get_image_paths(data_path, trial_str, pose_deltas, img_type='rgb', eval_type='train', add_reverse=False):

    if img_type == 'rgb':
        impath_l = os.path.join(data_path, 'image_02', 'data', '*.png')
        impath_r = os.path.join(data_path, 'image_03', 'data', '*.png')
    elif img_type == 'mono':
        impath_l = os.path.join(data_path, 'image_00', 'data', '*.png')
        impath_r = os.path.join(data_path, 'image_01', 'data', '*.png')
    else:
        raise ValueError('img_type must be `rgb` or `mono`')
    
    imfiles_l = sorted(glob.glob(impath_l))
    imfiles_r = sorted(glob.glob(impath_r))

    imfiles_l = [imfiles_l[i] for i in KITTI_SEQS_DICT[trial_str]['frames']]
    imfiles_r = [imfiles_r[i] for i in KITTI_SEQS_DICT[trial_str]['frames']]

    image_paths = []
    for p_delta in pose_deltas:
        if eval_type=='train':
            image_paths.extend([[imfiles_l[i], imfiles_r[i], imfiles_l[i+p_delta], imfiles_r[i+p_delta]] for i in range(len(imfiles_l) - p_delta)])
            if add_reverse:
                image_paths.extend([[imfiles_l[i + p_delta], imfiles_r[i + p_delta], imfiles_l[i], imfiles_r[i]] for i in range(len(imfiles_l) - p_delta)])
                
        elif eval_type=='test':
            #Only add every p_delta'th quad
            image_paths.extend([[imfiles_l[i], imfiles_r[i], imfiles_l[i+p_delta], imfiles_r[i+p_delta]] for i in range(0, len(imfiles_l) - p_delta, p_delta)])

        print('Adding {} {} image quads from trial {} for pose_delta: {}.'.format(len(image_paths), img_type, trial_str, p_delta))
    return image_paths

def compute_se3_precision(T_corr_list):
    """Computes the empirical inverse covariance matrix of all the se3 correction vectors"""
    N = len(T_corr_list)
    se3_vecs = np.empty((N, 6))
    for (i, T) in enumerate(T_corr_list):
        se3_vecs[i] = T_corr_list[i].log()
    return np.linalg.inv(np.cov(se3_vecs.T))

def process_ground_truth(trial_strs, tm_path, kitti_path, pose_deltas, eval_type='train', add_reverse=False):
    
    poses_correction = []
    poses_gt = []
    poses_est = []
    image_quad_paths_rgb = []
    image_quad_paths_mono = []
    
    tm_mat_files = []
    for t_id, trial_str in enumerate(trial_strs):
    
        drive_folder = KITTI_SEQS_DICT[trial_str]['date'] + '_drive_' + KITTI_SEQS_DICT[trial_str]['drive'] + '_sync'
        data_path = os.path.join(kitti_path, KITTI_SEQS_DICT[trial_str]['date'], drive_folder)
        tm_mat_file = os.path.join(tm_path, KITTI_SEQS_DICT[trial_str]['date'] + '_drive_' + KITTI_SEQS_DICT[trial_str]['drive'] + '.mat')

        try:
            tm = TrajectoryMetrics.loadmat(tm_mat_file)
        except FileNotFoundError:
            tm_mat_file = os.path.join(tm_path, trial_str + '.mat')
            tm = TrajectoryMetrics.loadmat(tm_mat_file)


        image_paths_rgb = get_image_paths(data_path, trial_str, pose_deltas, 'rgb', eval_type, add_reverse)
        image_paths_mono = get_image_paths(data_path, trial_str, pose_deltas, 'mono', eval_type, add_reverse)
    
        (T_corr, T_gt, T_est) = compute_vo_pose_errors(tm, pose_deltas, eval_type, add_reverse)

        if not len(image_paths_rgb) == len(T_corr):
            raise AssertionError('Number of image paths and number of poses differ. Image quads: {}. Poses: {}.'.format(len(image_paths_rgb), len(T_corr)))

        image_quad_paths_rgb.extend(image_paths_rgb)
        image_quad_paths_mono.extend(image_paths_mono)
        
        poses_correction.extend(T_corr)
        poses_gt.extend(T_gt)
        poses_est.extend(T_est)

        tm_mat_files.append(tm_mat_file)

    return (image_quad_paths_rgb, image_quad_paths_mono, poses_correction, poses_gt, poses_est, tm_mat_files)



def main():
    # test_trials = ['00']    
    # val_trials = ['01']
    # train_trials = ['04', '02', '05', '06', '07', '08', '09', '10']
    
    #Removed 01 and 04 (road trials)
    all_trials = ['00','02','05','06', '07', '08', '09', '10']

    #custom_training = [['00','06',['07','08','09','10']]]


    train_pose_deltas = [1,2,3] #How far apart should each quad image be? (KITTI is at 10hz, can input multiple)
    test_pose_delta = 2
    add_reverse = False #Add reverse transformations

    #Where is the KITTI data?
    kitti_path = '/media/m2-drive/datasets/KITTI/distorted_images'

    #Where are the baseline TrajectoryMetrics mat files stored?
    tm_path = '/media/raid5-array/experiments/Deep-PC/stereo_vo_results/baseline_distorted'

    #Where should we output the training files?
    data_path = '/media/raid5-array/experiments/Deep-PC/training_pose_errors_pytorch/distorted'

    
    for t_i, test_trial in enumerate(all_trials):
        if t_i > 2:
            break #Only produce trials for 00, 02 and 05

        if test_trial == all_trials[-1]:
            val_trial = all_trials[-2]
            train_trials = all_trials[:-2]
        else:
            val_trial = all_trials[t_i+1]
            train_trials = all_trials[:t_i] + all_trials[t_i+2 :]

    #for test_trial, val_trial, train_trials in custom_training:

        print('Processing.. Test: {}. Val: {}. Train: {}.'.format(test_trial, val_trial, train_trials))

        (train_img_paths_rgb, train_img_paths_mono, train_corr, train_gt, train_est, train_tm_mat_files) = process_ground_truth(train_trials, tm_path, kitti_path, train_pose_deltas, 'train', add_reverse)
        print('Processed {} training image quads.'.format(len(train_corr)))

        (val_img_paths_rgb, val_img_paths_mono, val_corr, val_gt, val_est, val_tm_mat_file) = process_ground_truth([val_trial], tm_path, kitti_path, [test_pose_delta], 'test', add_reverse)
        print('Processed {} validation image quads.'.format(len(val_corr)))

        (test_img_paths_rgb, test_img_paths_mono, test_corr, test_gt, test_est, test_tm_mat_file) = process_ground_truth([test_trial], tm_path, kitti_path, [test_pose_delta], 'test', add_reverse)
        print('Processed {} test image quads.'.format(len(test_corr)))

        #Save the data!
        kitti_data = KITTIData()

        kitti_data.train_pose_deltas = train_pose_deltas
        kitti_data.test_pose_delta = test_pose_delta

        kitti_data.train_sequences = train_trials
        kitti_data.train_img_paths_rgb = train_img_paths_rgb
        kitti_data.train_img_paths_mono = train_img_paths_mono
        kitti_data.train_T_corr = train_corr
        kitti_data.train_T_gt = train_gt
        kitti_data.train_T_est = train_est
        kitti_data.train_tm_mat_paths = train_tm_mat_files
        kitti_data.train_se3_precision = compute_se3_precision(train_corr)

        kitti_data.val_sequence = val_trial
        kitti_data.val_tm_mat_path = val_tm_mat_file[0] #Path to mat file containing the the trajectory (loaded by TrajectoryMetrics)
        kitti_data.val_img_paths_rgb = val_img_paths_rgb
        kitti_data.val_img_paths_mono = val_img_paths_mono
        kitti_data.val_T_corr = val_corr
        kitti_data.val_T_gt = val_gt
        kitti_data.val_T_est = val_est
     
        kitti_data.test_sequence = test_trial
        kitti_data.test_tm_mat_path = test_tm_mat_file[0]
        kitti_data.test_img_paths_rgb = test_img_paths_rgb
        kitti_data.test_img_paths_mono = test_img_paths_mono
        kitti_data.test_T_corr = test_corr
        kitti_data.test_T_gt = test_gt
        kitti_data.test_T_est = test_est

        data_filename = os.path.join(data_path, 'kitti_pose_error_data_test_{}.pickle'.format(test_trial))
        print('Saving to {} ....'.format(data_filename))

        with open(data_filename, 'wb') as f:
            pickle.dump(kitti_data, f, pickle.HIGHEST_PROTOCOL)

        print('Saved.')


if __name__ == '__main__':
    main()