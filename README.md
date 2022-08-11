# Facilitating 3D Object Tracking in Point Clouds with Image Semantics and Geometry

## Introduction

This repository is released for IFP (Image Fuse PointCloud) in our [PRCV 2021 paper](https://link.springer.com/chapter/10.1007/978-3-030-88004-0_48). Here we include our IFP model (PyTorch) and code for data preparation, training and testing on KITTI tracking dataset.

## Preliminary
* conda 
```
    conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0
```

* Install dependencies.
```
    pip install -r requirements.txt
```

* Build `_ext` module.
```
   cd lib/pointops && python setup.py install && cd ../../
   cd ./pointnet2/utils/DCNv2 && python setup.py build develop
```

* Download the dataset from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).

	Download [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) in the dataset and place them under the same parent folder.

## Evaluation

Train a new P2B model on KITTI data:
```
python train_tracking.py --data_dir=<data path> 
```

Test model on KITTI data:
```
python test_tracking.py --data_dir=<data path> 
```

Please refer to the code for setting of other optional arguments, including data split, training and testing parameters, etc.

If you think it is a useful work, please consider citing it.
```
@inproceedings{wang2021IFP,
  title={Facilitating 3D Object Tracking in Point Clouds with Image Semantics and Geometry},
  author={Lingpeng, Wang and Le, Hui and Jin, Xie},
  booktitle={PRCV},
  year={2021}
}

```

## Acknowledgements

Thank Qi for his implementation of [P2B](https://github.com/HaozheQi/P2B).
