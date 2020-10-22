import numpy as np
from torch.utils.data import Dataset
import os
import pickle
import random

def pc_normalize(pc):
	centroid = np.mean(pc, axis=0)
	pc = pc - centroid
	m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
	pc = pc / m
	return pc

def change_brightness(rgb):
	func = lambda c,r: min(1,c*r)
	func = np.vectorize(lambda c,r: min(1,c*r))
	r = random.uniform(0.5,1.5)
	return func(rgb,r)
	
def sample_data(data, num_sample, seed = None):
	N = data.shape[0]
	
	if (seed is not None):
		np.random.seed(seed)
		
	if (N == num_sample):
		return data, range(N)
	elif (N > num_sample):
		sample = np.random.choice(N, num_sample)
		return data[sample, ...], sample
	else:
		sample = np.random.choice(N, num_sample-N)
		dup_data = data[sample, ...]
		return np.concatenate([data, dup_data], 0), list(range(N)) + sample.tolist()

def sample_data_label(data, label, num_sample, seed = None):
	new_data, sample_indices = sample_data(data, num_sample, seed)
	new_label = label[sample_indices]
	return new_data, new_label

class PointCloudDataset(Dataset):
	def __init__(self, root, nClasses, nPoints, split, fold):
		super().__init__()
		self.root = root
		self.nClasses = nClasses
		self.nPoints = nPoints
		self.split = split
		self.fold = fold
		
		infile = open(self.root,'rb')
		dict_data = pickle.load(infile)
		infile.close()
		
		self.points = dict_data['points']
		self.labels = dict_data['labels']
		
		labelweights = np.zeros(self.nClasses)
		for seg in self.labels:
			tmp, _ = np.histogram(seg, range(nClasses + 1))
			labelweights += tmp
		labelweights = labelweights.astype(np.float32)
		labelweights = labelweights / np.sum(labelweights)
		self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
		
		self.idx = np.arange(0,len(self.points))
		
		if  (self.fold > 0):		
		
			random.Random(0).shuffle(self.idx)
				
			fold_size = int(len(self.points) / 10)
			
			if (self.split == "train"):
				idx_mask = np.ones(len(self.points), dtype=bool)
				idx_mask[ fold_size * (fold-1) : fold_size * fold] = False
			elif (self.split == "test"):
				idx_mask = np.zeros(len(self.points), dtype=bool)
				idx_mask[ fold_size * (fold-1) : fold_size * fold] = True
			
			self.idx = self.idx[idx_mask]
		
	def __getitem__(self, id):		
		current_points = self.points[self.idx[id]][0]
		current_labels = self.labels[self.idx[id]][0]
		
		current_points[:,0:3] = pc_normalize(current_points[:,0:3])
		
		if (self.split == "train"):
			current_points[:,3:6] = change_brightness(current_points[:,3:6])
			seed = None
		elif (self.split == "test"):
			seed = 0
			
		sampled_points, sampled_labels = sample_data_label(current_points, current_labels, self.nPoints, seed)
		
		return sampled_points, sampled_labels
		
	def __len__(self):
		return len(self.idx)