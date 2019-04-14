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
	def __init__(self, delta, mode='train', load_all=True):
		self.delta = delta
		self.datasets = []
		self.datasets_path = []
		self.key_lookup = dict()
		self.len_labels = []
		if load_all:
			self.add_dataset('imagenet_video', mode, 0)
			self.add_dataset('imagenet_video', mode, 1)
			self.add_dataset('imagenet_video', mode, 2)
			self.add_dataset('imagenet_video', mode, 3)
		self.video_idx = 0
		self.track_idx = 0
		self.image_idx = 0  # 0 ~ 180,000
		self.dataset_id = 0  # hard code. Modify later
		self.cur_line = 0  # current line # in labels.npy. i.e. from 0 to 280,000
	


	def add_dataset(self, dataset_name, mode, folder):
		dataset_ind = len(self.datasets)
		dataset_data = get_datasets.get_data_for_dataset(dataset_name, mode, folder)  # labels.npy: dim = X * 7 ...
		dataset_gt = dataset_data['gt']
		dataset_path = dataset_data['image_paths']

		self.len_labels += [dataset_gt.shape[0]]

		for xx in range(dataset_gt.shape[0]):
			line = dataset_gt[xx,:].astype(int)
			self.key_lookup[(dataset_ind, line[4], line[5], line[6])] = xx  # 0 ~ 280,000
		self.datasets.append(dataset_gt)  
		self.datasets_path.append(dataset_path)


	def get_data(self):
		# return images. len(images) = self.delta
		images = [None] * self.delta

		line = self.cur_line
		dataset_gt = self.datasets[self.dataset_id]

		self.video_idx, self.track_idx, self.image_idx = dataset_gt[line, 4:7]

		if line + self.delta - 1 < self.len_labels[self.dataset_id] and tuple(dataset_gt[line + self.delta - 1, 4:7]) == (self.video_idx, self.track_idx, self.image_idx+self.delta-1):  #
			# same consecutative video
			self.video_idx, self.track_idx, self.image_idx = dataset_gt[line][4:7]  # !!!
		else:
			raise ValueError('Bad Starting Point. Re-random Again!')

		gtKey = (self.dataset_id, self.video_idx, self.track_idx, self.image_idx)

		#print(self.dataset_id, self.cur_line, self.video_idx)

		for i in range(self.delta):
			# pull out image array
			image_paths = self.datasets_path[self.dataset_id]
			image_path_i = image_paths[self.image_idx+i]
			# path = '/media/yueshen/Sea_Gate!/imagenet/' + image_path_i[-85:]
			# image_array = cv2.imread(path)
			image_array = cv2.imread(image_path_i)  # on google
			images[i] = image_array.copy()

		return gtKey, images


	def get_data_sequence(self):
		tImage = np.zeros((self.delta, 2, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
		xywhLabels = np.zeros((self.delta, 4), dtype=np.float32)

		# need to incorporate four datasets....
		GENERATE_RANDOM = True
		while GENERATE_RANDOM:
			try:
				# randomly select a folder/dataset_id
				self.dataset_id = np.random.random_integers(1, 3)
				# randomly select the starting point/line in the entire labels.npy file 
				self.cur_line = np.random.random_integers(0, self.len_labels[self.dataset_id]-1)
				gtKey, images = self.get_data()  # return gtKey, images. len(images) = self.delta. gtKey points to the first line of the seq
				GENERATE_RANDOM = False
			except ValueError as err:
				#print('Re-generating random number:', err)
				pass

		# print('gtkey = ', gtKey)
		row = self.key_lookup[gtKey]  # 0 ~ 280,000, first row of the seq, in a dataset

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
				noisyBox = bboxPrev.copy()  # add noise ! problem here???
			    

			tImage[dd, 0, ...], outputBox = im_util.get_cropped_input(images[max(dd-1, 0)], bboxPrev, CROP_PAD, CROP_SIZE)
			tImage[dd,1,...] = im_util.get_cropped_input(images[dd], noisyBox, CROP_PAD, CROP_SIZE)[0]

			# shiftedBBox = bb_util.to_crop_coordinate_system(bboxOn, outputBox, CROP_PAD, 1)  # why CROP_PAD
			shiftedBBox = bb_util.to_crop_coordinate_system(bboxOn, noisyBox, CROP_PAD, 1)
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
	imagenet = Dataset(delta, mode='train', load_all=True)
	print('created dataset')

	# print(dataset.key_lookup[(0,11,0,0)])
	old = None
	# read tImage
	num_seq = 0 
	NUM_SEQ = 10
	Images = np.zeros((NUM_SEQ, delta*2, 3, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
	Labels = np.zeros((NUM_SEQ, delta, 4))
	while num_seq < NUM_SEQ:
		tImage, xyxyLabels = imagenet.get_data_sequence()
		print('video_idx', imagenet.video_idx, 'track_idx', imagenet.track_idx)

		if DEBUG:
			if num_seq == 0:
				old = tImage
			else:
				print(np.sum(old - tImage))
				old = tImage

			print('video id = ', imagenet.video_idx)
			print('t_image shape = ', tImage.shape, np.sum(tImage))
			# print('xyxyLabels shape = ', xyxyLabels[0,:])
			
		Images[num_seq, ...] = tImage.copy()
		Labels[num_seq, ...] = xyxyLabels.copy()
		num_seq += 1
		print('current iter # = ', num_seq)


	# np.save('Images.npy', Images)
	# np.save('Labels.npy', Labels)
	print('done!')
	print('Checking images... ')

	path = './test/'
	idx = -1   # random
	image = Images[idx, ...].copy()
	labels = Labels[idx, ...].copy()
	for i in range(image.shape[0]):
		# print(i)
		im = image[i, ...].transpose(1,2,0).copy()
		bbox = 227*labels[i//2, ...].copy()/10
		print('bbox = ', bbox)
		patch = drawing.drawRect(im, bbox, 1, (255,255,0))
		# print(im.shape)

		cv2.imwrite(path + str(i)+'.png', patch)
	# Images_load = np.load('Images.npy')
	# Labels_load = np.load('Labels.npy')
	# print(Images_load.shape, Labels_load.shape)
	# print('images load = ', np.sum(Images_load))
	# print('label load = ', Labels_load[0,0:5,:])
