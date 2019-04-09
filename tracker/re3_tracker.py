import cv2
import numpy as np
import torch
import time

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))
from tracker.network import Re3Net
import utils.bb_util as bb_util
import utils.im_util as im_util
from training import get_sequence
from constants import CROP_PAD
from constants import CROP_SIZE

class Re3Tracker(object):
    def __init__(self, model_path, device):
        self.net = Re3Net().to(device)
        if model_path is not None:
            self.net.load_state_dict(torch.load(model_path))
        self.net.eval()
        self.device = device
        self.tracked_data = {}

    def track(self, id, image, bbox = None):
        image = image.copy()

        if bbox is not None:
            lstm_state = None
            past_bbox = bbox
            prev_image = image
            forward_count = 0
        elif id in self.tracked_data:
            lstm_state, initial_state, past_bbox, prev_image, forward_count = self.tracked_data[id]
        else:
            raise Exception('Id {0} without initial bounding box'.format(id))

        cropped_input0, past_bbox_padded = im_util.get_cropped_input(prev_image, past_bbox, CROP_PAD, CROP_SIZE)
        cropped_input1, _ = im_util.get_cropped_input(image, past_bbox, CROP_PAD, CROP_SIZE)

        network_input = np.stack((cropped_input0.transpose(2,0,1),cropped_input1.transpose(2,0,1)))
        network_input = network_input.reshape((1,2,3,CROP_SIZE, CROP_SIZE))
        network_input = torch.tensor(network_input, dtype = torch.float)    
        
        with torch.no_grad():
            network_input = network_input.to(self.device)
            network_output, lstm_state = self.net(network_input, prevLstmState = lstm_state)

        if forward_count == 0:
            state1_h = lstm_state[0][0].clone()
            state1_c = lstm_state[0][1].clone()
            state2_h = lstm_state[1][0].clone()
            state2_c = lstm_state[1][1].clone()
            initial_state = ((state1_h, state1_c), (state2_h, state2_c))
        
        prev_image = image
        # ***** padding = 1 or 2
        output_bbox = bb_util.from_crop_coordinate_system(network_output.data.numpy(), past_bbox_padded, 1, 1)

        # Reset state
        if forward_count > 0 and forward_count % 32 == 0:
            lstm_state = None

        forward_count += 1

        if bbox is not None:
            output_bbox = bbox

        output_bbox = output_bbox.reshape(4)

        self.tracked_data[id] = (lstm_state, initial_state, output_bbox, prev_image, forward_count)
        
        return output_bbox

    def reset(self):
        self.tracked_data = {}

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Re3Net().to(device)
    net.load_state_dict(torch.load('checkpoint.pth'))
    net.eval()
    tracker = Re3Tracker(net, device)


