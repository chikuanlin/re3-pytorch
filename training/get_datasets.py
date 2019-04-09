import numpy as np
import glob
import os

def get_data_for_dataset(dataset_name, mode):
    # Implement this for each dataset.
    if dataset_name == 'imagenet_video':
        # datadir = os.path.join(
                # os.path.dirname(__file__),
                # 'datasets',
                # 'imagenet_video')
        datadir = ''
        labeldir = '/home/yueshen/eecs442/proj/final_project'
        gt = np.load(labeldir + '/labels/' + mode + '/labels.npy')
        image_paths = [datadir + line.strip()
            for line in open(labeldir + '/labels/' + mode + '/image_names.txt')]
    return {
            'gt' : gt,
            'image_paths' : image_paths,
            }

