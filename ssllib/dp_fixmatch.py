import os
import sys
import numpy as np
import logging


from PIL import Image

from utils.dataset_utils import DatasetWrapper


logger = logging.getLogger(__name__)



class DPDataset(DatasetWrapper):

	def __init__(self, unlabeled_ds, unlabeled_mask,  eps=100.0):
		
		self.data = unlabeled_ds.data
		self.targets = unlabeled_ds.targets
		self.transform = unlabeled_ds.transform
		self.unlabeled_mask = unlabeled_mask
		self.eps = eps

	def __len__(self):
		return len(self.data)


	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]
		mask = self.unlabeled_mask[index]

		noise = np.random.normal(scale=self.eps, size=img.shape) * mask[:, :, np.newaxis]
		img = img.astype(np.float32) + noise
		img = np.clip(img, 0.0, 255.0)
		img = img.astype(np.uint8)


		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)
		return img, target



class OccludeDataset(DatasetWrapper):

	def __init__(self, unlabeled_ds, unlabeled_mask, mean):
		
		self.data = unlabeled_ds.data
		self.targets = unlabeled_ds.targets
		self.transform = unlabeled_ds.transform
		self.unlabeled_mask = unlabeled_mask
		self.mean = np.array(mean) * 255.0

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]
		mask = self.unlabeled_mask[index]

		alpha = np.tile( mask[:, :, np.newaxis], (1, 1, img.shape[2]))

		mean_img = np.tile(self.mean[np.newaxis, np.newaxis, :], (img.shape[0], img.shape[1], 1))

		img = img.astype(np.float32)  * (1.0 - alpha) + mean_img * alpha
		img = np.clip(img, 0.0, 255.0)
		img = img.astype(np.uint8)

		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)
		return img, target



class DPOccludeDataset(DatasetWrapper):

	def __init__(self, unlabeled_ds, unlabeled_mask, mean, eps=100.0):
		
		self.data = unlabeled_ds.data
		self.targets = unlabeled_ds.targets
		self.transform = unlabeled_ds.transform
		self.unlabeled_mask = unlabeled_mask
		self.mean = np.array(mean) * 255.0
		self.eps = eps

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]
		mask = self.unlabeled_mask[index]

		alpha = np.tile( mask[:, :, np.newaxis], (1, 1, img.shape[2]))

		mean_img = np.tile(self.mean[np.newaxis, np.newaxis, :], (img.shape[0], img.shape[1], 1))

		img = img.astype(np.float32)  * (1.0 - alpha) + mean_img * alpha
		noise = np.random.normal(scale=self.eps, size=img.shape) * mask[:, :, np.newaxis]
		img = img + noise

		img = np.clip(img, 0.0, 255.0)
		img = img.astype(np.uint8)

		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)
		return img, target



