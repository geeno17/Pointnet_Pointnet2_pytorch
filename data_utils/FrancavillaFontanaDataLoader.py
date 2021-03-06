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
	
def sample_data(data, num_sample, dist = None, seed = None):
	N = data.shape[0]
	
	if (seed is not None):
		np.random.seed(seed)
		
	if (N == num_sample):
		return data, range(N)
	elif (N > num_sample):
		sample = np.random.choice(N, num_sample, replace = False, p = dist)
		return data[sample, ...], sample
	else:
		sample = np.random.choice(N, num_sample-N, p = dist)
		dup_data = data[sample, ...]
		return np.concatenate([data, dup_data], 0), list(range(N)) + sample.tolist()

def sample_data_label(data, label, num_sample, dist = None, seed = None):
	new_data, sample_indices = sample_data(data, num_sample, dist, seed)
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
		
		self.max_coordinates = np.array([0.0,0.0,0.0])

		for p in self.points:
			max_x = np.max(p[0,:,0])
			max_y = np.max(p[0,:,1])
			max_z = np.max(p[0,:,2])
			if max_x > self.max_coordinates[0]:
				self.max_coordinates[0] = max_x
			if max_y > self.max_coordinates[1]:
				self.max_coordinates[1] = max_y
			if max_z > self.max_coordinates[2]:
				self.max_coordinates[2] = max_z
		
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
		
		normalized_points = pc_normalize(np.copy(current_points[:,0:3]))
		normalized_env_points = np.divide(np.copy(current_points[:,0:3]),self.max_coordinates)
		rgb = np.copy(current_points[:,3:6])		
		
		if (self.split == "train"):			
			weights = [self.labelweights[current_labels[i]] for i in range(len(current_labels))]
			dist = weights / sum(weights)
			rgb = change_brightness(rgb)			
			seed = None
		elif (self.split == "test"):
			dist = None
			seed = 0
			
		processed_points = np.concatenate((normalized_points, normalized_env_points, rgb),axis=1) 
			
		sampled_points, sampled_labels = sample_data_label(processed_points, current_labels, self.nPoints, dist, seed)
		
		return sampled_points, sampled_labels
		
	def __len__(self):
		return len(self.idx)