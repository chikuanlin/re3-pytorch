# Re-implementation of Re3 in Pytorch
* This is a repository for Winter 2019 EECS 442 Computer Vision Final Project at the University of Michigan. It is an reimplementation based on the [code](https://gitlab.com/danielgordon10/re3-tensorflow) authored by [Daniel Gorden](https://homes.cs.washington.edu/~xkcd/) and his [paper](https://arxiv.org/pdf/1705.06368.pdf).
* Please refer to https://arxiv.org/pdf/1705.06368.pdf for implementation details and https://gitlab.com/danielgordon10/re3-tensorflow for the original source code.

## Requirements
* Python3 or above
* Pytorch
* Numpy
* OpenCV
* CUDA

## Setup

Download the repository
```
git clone https://github.com/chikuanlin/re3-pytorch.git
cd re3-pytorch
```

Generate labels (GT) of ImageNet video dataset
```
cd dataset/
python3 make_label_files.py
```

Generate labels (GT) of ImageNet object detection dataset
```
cd dataset/detection/
python3 make_label_files.py
```

Train the network
```
python3 training/training.py -m NUM_ITERS -n NUM_UNROLL -u USE_NETWORK_PROB -b BATCH_SIZE -l LEARNING_RATE
```
Example:
```
python3 training/training.py -m 10000 -n 2 -u 0 -b 64 -l 1e-5
```

Test the network with webcam and save the result with -r
```
python3 tracker/webcam_demo.py -r
```

Generate a video from a folder of image sequence
```
python3 tracker/re3_tracker.py -p "FOLDER_PATH" -b "X_MIN YMIN X_MAX Y_MAX"
```

Run VOT test dataset and output IOU scores (Need to modify the path in the script)
```
python3 tracker/vot_test_tracker.py
```

## Main Files
Helper functions and scripts that are modified or obtained from the original source code would be labeled with *
* datasets/
  * [make_label_files.py*](dataset/make_label_files.py) - The label (GT) generating script for ImageNet Video dataset. The output .npy label file and text image name file are saved in labels/. 
  * [detection/make_label_files.py*](dataset/detection/make_label_files.py) - The label (GT) generating script for ImageNet Object Detection dataset. The output .npy label file and text image name file are saved in detection/labels/.
* tracker/
  * [network.py](tracker/network.py) - A Re3 class implementation in Pytorch
  * [re3_tracker.py*](tracker/network.py) - A multi-object tracker class
  * [vot_test_tracker.py](tracker/vot_test_tracker.py) - A script that generates the IOU score from given VOT dataset
  * [webcam_demo.py*](tracker/webcam_demo.py) - A webcam demo script (multi-object tracking)
* training/
  * [training.py](training/training.py) - Main script for training the network
  * [get_rand_sequence.py*](training/get_rand_sequence.py) - generate a batch of sequence randomly, called by [training.py](training/training.py) 
  * [get_datasets.py*](training/get_datasets.py) - helper function called by [get_rand_sequence.py](training/get_rand_sequence.py)
* utils/*
  * Please refer to the original source code for detailed information
  
## Project Collaborators
* [chikuanlin](https://github.com/chikuanlin)
* [pikelcw](https://github.com/pikelcw)
* [yueshen95](https://github.com/yueshen95)
