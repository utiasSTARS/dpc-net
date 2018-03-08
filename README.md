# DPC-Net: Deep Pose Correction for Visual Localization
Code for [Dense Pose Corrections](https://arxiv.org/abs/1709.03128). DPC-Net learns SE(3) corrections to classical geometric and probabilistic visual localization pipelines (e.g., visual odometry). 

<img src="https://raw.githubusercontent.com/utiasSTARS/dpc-net/master/dpc_high_level.png" width="500px"/>


## Installation & Pre-Requisites

1. Ensure that [pytorch](http://pytorch.org) is installed on your machine. We perform all training and testing on a a GTX Titan X (Maxwell) with 12GiB of memory.

2. Install pyslam and lie_groups (links coming soon...). We use pyslam's ``TrajectoryMetrics`` class to store computed trajectories, and use it to compute pose graph relaxations.

2. Clone DPC-net:
```
git clone https://github.com/utiasSTARS/dpc-net
```

## Testing with pre-trained model on KITTI data
1. Run ``test_dpc_net.py --seqs 00 --corr pose``.

1. Run ``test_dpc_net.py --seqs 00 --corr pose``.


## Training
To train DPC-Net, you need two things:
1. A list of frame-to-frame *corrections* to a localization pipeline (this is typically computed using some form of ground-truth). 
2. A set of images (stereo or mono, depending on whether the correction is SO(3) or SE(3)) from which the model can learn corrections.

### Using KITTI data
To use the [KITTI odometry benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to train DPC-Net, you can use ``train_dpc_net.py``.

## Citation
If you use this code in your research, please cite:
```
@article{2018_Peretroukhin_DPC,
  author = {Valentin Peretroukhin and Jonathan Kelly},
  doi = {10.1109/LRA.2017.2778765},
  journal = {{IEEE} Robotics and Automation Letters},
  link = {https://arxiv.org/abs/1709.03128},
  title = {{DPC-Net}: Deep Pose Correction for Visual Localization},
  year = {2018}
}
```
