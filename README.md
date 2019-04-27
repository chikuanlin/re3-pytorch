# Re-implementation of Re3 in Pytorch
This is a repository for Winter 2019 EECS 442 Computer Vision Final Project at the University of Michigan.It is an reimplementation based on the [code](https://gitlab.com/danielgordon10/re3-tensorflow) authored by [Daniel Gorden](https://homes.cs.washington.edu/~xkcd/) and his [paper](https://arxiv.org/pdf/1705.06368.pdf).
Please refer to https://arxiv.org/pdf/1705.06368.pdf for implementation details and https://gitlab.com/danielgordon10/re3-tensorflow for the original code source.

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
python3 training/training.py -m 10000 -n 2 -u 0 -b 64 -l 1e-5
```

Test the network with webcam

```
python3 webcam_demo.py
```

Generate a video from a folder of image sequence
```
python3 tracker/re3_tracker.py -p "FOLDER_PATH" -b "X_MIN YMIN X_MAX Y_MAX"
```


## Main Files
* datasets/
  * [make_label_files.py](dataset/make_label_files.py) - The label (GT) generating script for ImageNet Video dataset. The output .npy label file and text image name file are saved in labels/. 
  * [detection/make_label_files.py](dataset/detection/make_label_files.py) - The label (GT) generating script for ImageNet Object Detection dataset. The output .npy label file and text image name file are saved in detection/labels/.
* tracker/
  * [network.py](tracker/network.py) - A Re3 class implementation in Pytorch
  * [re3_tracker.py](tracker/network.py) - A multi-object tracker class
  * [re3_vot_tracker.py](tracker/re3_vot_tracker.py) - ...
* training/
  * [training.py](training/training.py) - Main script for training the network
  * [get_rand_sequence.py](training/get_rand_sequence.py) - generate a batch of sequence randomly, called by [training.py](training/training.py)
  * [get_datasets.py](training/get_datasets.py) - helper function called by [get_rand_sequence.py](training/get_rand_sequence.py)
* utils/
  * Please refer to the [original code source](https://gitlab.com/danielgordon10/re3-tensorflow) for detailed information
  
## Project Collaborators
* [chikuanlin](https://github.com/chikuanlin)
* [pikelcw](https://github.com/pikelcw)
* [yueshen95](https://github.com/yueshen95)
