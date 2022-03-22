import os
import sys
import logging

from PIL import Image
import numpy as np
import pickle as pkl 

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

from dataset.randaugment import RandAugmentMC

from utils.vis import img_grid

import cv2



logger = logging.getLogger(__name__)


dump_dir = "./data/dataset_cache"
def exists_split_indices(split_id):
	return os.path.exists(os.path.join(dump_dir, split_id+'.pkl'))

def dump_split_indices(train_labeled_indices, train_unlabeled_indices, split_id):
	if not os.path.exists(dump_dir):
		os.mkdir(dump_dir)
	pkl.dump((train_labeled_indices, train_unlabeled_indices), open(os.path.join(dump_dir, split_id+'.pkl'), 'wb'))

def load_split_indices(split_id):
	return pkl.load(open(os.path.join(dump_dir, split_id+'.pkl'), 'rb'))



cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2471, 0.2435, 0.2616]
cifar100_mean = [0.5071, 0.4867, 0.4408]
cifar100_std = [0.2675, 0.2565, 0.2761]
svhn_mean = [0.43768454, 0.44376847, 0.4728039 ]
svhn_std = [0.19803019, 0.20101567, 0.19703582]

normal_mean = [0.5, 0.5, 0.5]
normal_std = [0.5, 0.5, 0.5]



def normalize_fn(tensor, mean, std):
	"""Differentiable version of torchvision.functional.normalize"""
	# here we assume the color channel is in at dim=1
	mean = mean[None, :, None, None]
	std = std[None, :, None, None]
	return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
	def __init__(self, mean, std):
		super(NormalizeByChannelMeanStd, self).__init__()
		if not isinstance(mean, torch.Tensor):
			# mean = torch.tensor(np.array(mean).reshape([1, -1, 1, 1])).float()
			mean = torch.tensor(mean)
		if not isinstance(std, torch.Tensor):
			# std = torch.tensor(np.array(std).reshape([1, -1, 1, 1])).float()
			std = torch.tensor(std)
		self.register_buffer("mean", mean)
		self.register_buffer("std", std)

	def forward(self, tensor):
		return normalize_fn(tensor, self.mean, self.std)

	def extra_repr(self):
		return 'mean={}, std={}'.format(self.mean, self.std)




class TransformFix(object):
	def __init__(self, mean=None, std=None, size=32):
		self.weak = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(size=size,
								  padding=int(size*0.125),
								  padding_mode='reflect')])
		self.strong = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(size=size,
								  padding=int(size*0.125),
								  padding_mode='reflect'),
			RandAugmentMC(n=2, m=10)])
		
		if mean is not None and std is not None:
			self.out_op = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std)])
		else:
			self.out_op = transforms.ToTensor()

	def __call__(self, x):
		weak = self.weak(x)
		strong = self.strong(x)
		return self.out_op(weak), self.out_op(strong)




def indices_expand(indices, num_expand):
	n = num_expand // len(indices)
	indices = np.hstack(
		[indices for _ in range(n)])

	if len(indices) < num_expand:
		diff = num_expand - len(indices)
		indices = np.hstack(
			(indices, np.random.choice(indices, diff)))
	else:
		assert len(indices) == num_expand
	return indices



class DatasetWrapper(object):
	
	def __init__(self, data, targets, transform=None):
		assert data.dtype == np.uint8
		self.data = data
		self.targets = targets
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]
		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)
		return img, target

	def clone(self):
		return DatasetWrapper(self.data.copy(), self.targets.copy(), self.transform)

	def save_imgs_to_dir(self, save_path):
		"""
			将图片全部保存到save_path下面。
			每100张图片叠在一起成为一个大图进行保存
		"""
		num_images = self.data.shape[0]
		num_outputs = (num_images + 99) // 100
		logger.info("DatasetWrapper write to {}, {} images".format(save_path, num_outputs))

		for i in range(num_outputs):
			if i != num_outputs - 1:
				imgs = self.data[i*100:(i+1)*100]
			else:
				imgs = self.data[i*100:]
			imgs = imgs[:, :, :, ::-1]
			num_imgs = len(imgs)

			if num_imgs != 100:
				imgs = np.concatenate([imgs, np.zeros([100-num_imgs,]+list(imgs.shape[1:]), dtype=imgs.dtype),], axis=0)

			large_img = img_grid(imgs, nb_images_per_row=10, pad=0, pad_value=0)
			large_img = np.uint8(large_img)
			
			cv2.imwrite(os.path.join(save_path, '%05d_%d.png'%(i, num_imgs)), large_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


	@classmethod
	def load_imgs_from_dir(cls, load_path, img_size=32):
		image_filelist = [fn for fn in os.listdir(load_path) if fn.endswith('.png') or fn.endswith('.jpg')]
		if len(image_filelist) == 0:
			return []

		image_filelist = sorted(image_filelist)

		print(image_filelist)

		load_imgs = [cv2.imread(os.path.join(load_path, fn)) for fn in image_filelist]
		num_list = [int(fn.split('.')[0].split('_')[1]) for fn in image_filelist]


		def split_images(imgs, n, img_size=32):
			imgs = np.array(imgs)
			s = imgs.shape
			imgs = imgs.reshape([int(s[0] / img_size), img_size, int(s[1] / img_size), img_size, int(s[2])]).transpose([0, 2, 1, 3, 4,])
			imgs = imgs.reshape([int(s[0] * s[1] / img_size / img_size), img_size, img_size, int(s[2]), ])
			imgs = imgs[:n, :, :, ::-1]
			return imgs

		load_imgs = np.concatenate([split_images(img, n) for img, n in zip(load_imgs, num_list)], axis=0)
	
		return load_imgs



class IndexDatasetWrapper(DatasetWrapper):
	def __init__(self, wrap_dataset, indices, target_transform=None, output_index=False, **kwargs):
		indices = np.array(indices)
		assert wrap_dataset.data.dtype == np.uint8
		self.data = np.array(wrap_dataset.data)[indices]
		self.targets = np.array(wrap_dataset.targets)[indices]
		
		if target_transform is not None:
			self.targets = np.array([target_transform[t] for t in self.targets])
		
		if "transform" in kwargs:
			self.transform = kwargs["transform"]
		elif wrap_dataset.transform is not None:
			self.transform = wrap_dataset.transform
		else:
			self.transform = None
		
		self.output_index = output_index


	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]
		img = Image.fromarray(img)

		if self.transform is not None:
			img = self.transform(img)

		if self.output_index:
			return index, img, target
		else:
			return img, target



class ConcatDataset(DatasetWrapper):
	def __init__(self, wrap_dataset_list):
		self.wrap_dataset_list = wrap_dataset_list
		self.num_ds = len(wrap_dataset_list)
		self.len_list = [0,] + [len(ds) for ds in wrap_dataset_list]
		self.len_accumulate_list = np.cumsum(self.len_list)
		self.transform = wrap_dataset_list[0].transform
		self.len_sum = np.sum(self.len_list)

		for ds in wrap_dataset_list:
			# assert ds.data.dtype == np.uint8
			ds.transform = None

	def __len__(self):
		return self.len_sum

	def __getitem__(self, index):
		for ds_ind in range(self.num_ds):
			if index >= self.len_accumulate_list[ds_ind] and index < self.len_accumulate_list[ds_ind+1]:
				index_inner = index - self.len_accumulate_list[ds_ind]
				img, target = self.wrap_dataset_list[ds_ind][index_inner]
				if self.transform is not None:
					img = self.transform(img)
				return img, target

		raise ValueError('ConcatDataset getitem error : {}, {}'.format(index, self.len_accumulate_list))


	@property
	def targets(self):
		return np.concatenate([
			ds.targets for ds in self.wrap_dataset_list
		], axis=0)


	@property
	def data(self):
		return np.concatenate([
			ds.data for ds in self.wrap_dataset_list
		], axis=0)



def x_u_split(labels,
			  num_labeled,
			  num_classes, seed=0):
	label_per_class = num_labeled // num_classes
	labels = np.array(labels)
	labeled_idx = []
	unlabeled_idx = []
	np.random.seed(seed=seed)
	for i in range(num_classes):
		idx = np.where(labels == i)[0]
		np.random.shuffle(idx)
		labeled_idx.extend(idx[:label_per_class])
		unlabeled_idx.extend(idx[label_per_class:])
	return labeled_idx, unlabeled_idx


def x_u_split_with_filter(labels, cls_list,
			  num_labeled, seed=0):
	label_per_class = num_labeled // len(cls_list)
	labels = np.array(labels)
	labeled_idx = []
	unlabeled_idx = []
	np.random.seed(seed=seed)
	for cls_ind in cls_list:
		idx = np.where(labels == cls_ind)[0]
		np.random.shuffle(idx)
		labeled_idx.extend(idx[:label_per_class])
		unlabeled_idx.extend(idx[label_per_class:])
	return labeled_idx, unlabeled_idx


def filter_by_targets(indices, targets, inc_t_list):
	inc_t_set = set(inc_t_list)
	return np.array([ind for ind in indices if targets[ind] in inc_t_set])

