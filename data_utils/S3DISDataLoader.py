import os
import numpy as np
import h5py
from torch.utils.data import Dataset

def load_h5(h5_filename):
	f = h5py.File(h5_filename)
	data = f['data'][:]
	label = f['label'][:]
	return (data, label)

class PointCloudDataset(Dataset):
	def __init__(self, split, testArea, root = 'data/', nClasses = 13, nPoints = 4096):
		super().__init__()
		self.root = root
		self.nClasses = nClasses
		self.nPoints = nPoints
		self.split = split
		self.testArea = testArea
		
		ALL_FILES = [line.rstrip() for line in open(self.root + 'indoor3d_sem_seg_hdf5_data/all_files.txt')]
		room_filelist = [line.rstrip() for line in open(self.root + 'indoor3d_sem_seg_hdf5_data/room_filelist.txt')]

		# Load ALL data
		data_batch_list = []
		label_batch_list = []
		for h5_filename in ALL_FILES:
			data_batch, label_batch = load_h5(self.root + h5_filename)
			data_batch_list.append(data_batch.astype(np.float16))
			label_batch_list.append(label_batch.astype(np.uint8))
		data_batches = np.concatenate(data_batch_list, 0)
		label_batches = np.concatenate(label_batch_list, 0)

		test_area = 'Area_'+str(self.testArea)
		train_idxs = []
		test_idxs = []
		for i,room_name in enumerate(room_filelist):
			if test_area in room_name:
				test_idxs.append(i)
			else:
				train_idxs.append(i)
				
		if split == 'train':
			self.points = data_batches[train_idxs,...]
			self.labels = label_batches[train_idxs]
		elif split == 'test':
			self.points = data_batches[test_idxs,...]
			self.labels = label_batches[test_idxs]
			
		labelweights = np.zeros(nClasses)
		for seg in self.labels:
			tmp, _ = np.histogram(seg, range(nClasses + 1))
			labelweights += tmp
		labelweights = labelweights.astype(np.float32)
		labelweights = labelweights / np.sum(labelweights)
		self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
		
	def __getitem__(self, id):		
		current_points = self.points[id]
		current_labels = self.labels[id]
		return current_points, current_labels
		
	def __len__(self):
		return int(self.points.shape[0])