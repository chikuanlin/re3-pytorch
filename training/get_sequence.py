import numpy as np
import cv2

import get_datasets

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

from utils import im_util
from utils import bb_util
from utils import drawing

from constants import CROP_PAD
from constants import CROP_SIZE
from constants import LSTM_SIZE
from constants import OUTPUT_WIDTH
from constants import OUTPUT_HEIGHT
from constants import LOG_DIR



class Dataset(object):
	# OUR IMPLEMENTATION
	def __init__(self, delta, mode='train', start_line=0, stride=1):
		self.delta = delta
		self.datasets = []
		self.datasets_path = []
		self.key_lookup = dict()
		self.seq_idx_lookup = dict()
		self.seq_lookup = []  # seq_lookup[seq_idx] = line (in dataset_gt / labels.npy)
		self.add_dataset('imagenet_video', mode)
		self.video_idx = 0
		self.track_idx = 0
		self.image_idx = 0  # 0 ~ 180,000
		self.dataset_id = 0  # hard code. Modify later
		self.cur_line = start_line  # current line # in labels.npy. i.e. from 0 to 280,000
		self.seq_idx = self.seq_idx_lookup[start_line]  # seq index of the dataset videos. += 1 at the switch of the track_id or video_id. NOT current number of seq
		print('initial seq_idx = ', self.seq_idx)
		self.stride = stride



	def add_dataset(self, dataset_name, mode):
		dataset_ind = len(self.datasets)
		dataset_data = get_datasets.get_data_for_dataset(dataset_name, mode)  # labels.npy: dim = X * 7 ...
		dataset_gt = dataset_data['gt']
		dataset_path = dataset_data['image_paths']

		track_idx_cur = -1
		video_idx_cur = -1
		seq_idx = -1

		for xx in range(dataset_gt.shape[0]):
			line = dataset_gt[xx,:].astype(int)
			self.key_lookup[(dataset_ind, line[4], line[5], line[6])] = xx  # 0 ~ 280,000
			if line[4] != video_idx_cur or line[5] != track_idx_cur:
				seq_idx += 1
				self.seq_lookup.append(xx)
				video_idx_cur = line[4]
				track_idx_cur = line[5] 
			self.seq_idx_lookup[xx] = seq_idx
		self.datasets.append(dataset_gt)  
		self.datasets_path.append(dataset_path)


	def get_data(self):
		# return images. len(images) = self.delta
		images = [None] * self.delta
		# line = self.key_lookup[(self.dataset_id, self.video_idx, self.track_idx, self.image_idx)]
		line = self.cur_line
		dataset_gt = self.datasets[self.dataset_id]

		self.video_idx, self.track_idx, self.image_idx = dataset_gt[line, 4:7]

		looking = True  # looking for a consecutive sequence
		# check if there is enough images to generate a sequence with len == self.delta
		
		while looking:
			# print('A = ', tuple(dataset_gt[line + self.delta - 1][4:7]))
			# print('B = ', (self.video_idx, self.track_idx, self.image_idx+self.delta-1))
			if tuple(dataset_gt[line + self.delta - 1, 4:7]) == (self.video_idx, self.track_idx, self.image_idx+self.delta-1):  #
				# same consecutative video
				looking = False
				self.video_idx, self.track_idx, self.image_idx = dataset_gt[line][4:7]  # !!!
			else:
				print('seq..++', self.seq_idx)
				self.seq_idx += 1
				line = self.seq_lookup[self.seq_idx]
				# print('line = ', line)
				self.video_idx, self.track_idx, self.image_idx = dataset_gt[line][4:7]

		gtKey = (self.dataset_id, self.video_idx, self.track_idx, self.image_idx) 

		for i in range(self.delta):
			# pull out image array
			image_paths = self.datasets_path[self.dataset_id]
			image_path_i = image_paths[self.image_idx+i]
			image_array = cv2.imread(image_path_i)
			images[i] = image_array.copy()


		self.cur_line = line + self.stride  # next possible seq's first line
		if self.seq_idx_lookup[self.cur_line] > self.seq_idx:
			print('seq++ due to stride')
			# always make sure that self.cur_line is consistent with self.seq_idx
			self.seq_idx += 1
			self.cur_line = self.seq_lookup[self.seq_idx]
		return gtKey, images


	def get_data_sequence(self):
		tImage = np.zeros((self.delta, 2, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
		xywhLabels = np.zeros((self.delta, 4), dtype=np.float32)

		dataset_id = self.dataset_id
		video_idx = self.video_idx
		track_idx = self.track_idx
		image_idx = self.image_idx

		gtKey, images = self.get_data()  # return gtKey, images. len(images) = self.delta. gtKey points to the first line of the seq
		# print('gtkey = ', gtKey)
		row = self.key_lookup[gtKey]  # 0 ~ 280,000, first row of the seq

		# Initialize the first frame
		initBox = self.datasets[gtKey[0]][row, :4].copy()

		# bboxPrev starts at the initial box and is the best guess (or gt) for the image0 location.
		bboxPrev = initBox
		lstmState = None

		for dd in range(self.delta):
			newKey = list(gtKey)
			newKey[3] += dd
			newKey = tuple(newKey)
			row = self.key_lookup[newKey]
			
			bboxOn = self.datasets[newKey[0]][row, :4].copy()
			# print('row = ', bboxOn)
			if dd == 0:
			    noisyBox = bboxOn.copy()
			else:
				noisyBox = bboxOn.copy()  # add noise ! problem here???
			    

			tImage[dd, 0, ...], outputBox = im_util.get_cropped_input(images[max(dd-1, 0)], bboxPrev, CROP_PAD, CROP_SIZE)
			tImage[dd,1,...] = im_util.get_cropped_input(images[dd], noisyBox, CROP_PAD, CROP_SIZE)[0]

			shiftedBBox = bb_util.to_crop_coordinate_system(bboxOn, outputBox, CROP_PAD, 1)  # why CROP_PAD
			# print('shiftedBBox = ', shiftedBBox)
			shiftedBBoxXYWH = bb_util.xyxy_to_xywh(shiftedBBox)
			xywhLabels[dd,:] = shiftedBBoxXYWH

			bboxPrev = bboxOn

		# print('xywhLabels = ', xywhLabels)
		tImage = tImage.reshape([self.delta * 2] + list(tImage.shape[2:]))
		xyxyLabels = bb_util.xywh_to_xyxy(xywhLabels.T).T * 10
		xyxyLabels = xyxyLabels.astype(np.float32)
		tImage = tImage.transpose(0,3,1,2)
		return tImage, xyxyLabels



if __name__ == '__main__':
	DEBUG = False

	delta = 32
	dataset = Dataset(delta, start_line=0, stride=31)
	print('created dataset')
	# print(dataset.key_lookup[(0,11,0,0)])
	old = None
	# read tImage
	num_seq = 0
	NUM_SEQ = 34
	Images = np.zeros((NUM_SEQ, delta*2, 3, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
	Labels = np.zeros((NUM_SEQ, delta, 4))
	while num_seq < NUM_SEQ:
		tImage, xyxyLabels = dataset.get_data_sequence()
		print('video_idx', dataset.video_idx, 'track_idx', dataset.track_idx)

		if DEBUG:
			if num_seq == 0:
				old = tImage
			else:
				print(np.sum(old - tImage))
				old = tImage

			print('video id = ', dataset.video_idx)
			print('t_image shape = ', tImage.shape, np.sum(tImage))
			# print('xyxyLabels shape = ', xyxyLabels[0,:])
			
		Images[num_seq, ...] = tImage.copy()
		Labels[num_seq, ...] = xyxyLabels.copy()
		num_seq += 1
		print('current seq # = ', num_seq)


	# np.save('Images.npy', Images)
	# np.save('Labels.npy', Labels)

	print('final seq idx = ', dataset.seq_idx)
	print('done!')
	print('Checking images... ')

	path = './test/'
	idx = 15   # random
	image = Images[idx, ...].copy()
	labels = Labels[idx, ...].copy()
	for i in range(image.shape[0]):
		# print(i)
		im = image[i, ...].transpose(1,2,0).copy()
		bbox = labels[i//2, ...].copy()
		patch = drawing.drawRect(im, bbox, 1, (255,255,0))
		# print(im.shape)

		cv2.imwrite(path + str(i)+'.png', patch)
	# Images_load = np.load('Images.npy')
	# Labels_load = np.load('Labels.npy')
	# print(Images_load.shape, Labels_load.shape)
	# print('images load = ', np.sum(Images_load))
	# print('label load = ', Labels_load[0,0:5,:])
