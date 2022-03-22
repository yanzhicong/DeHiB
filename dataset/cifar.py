import logging

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms


from dataset.dataset_utils import *


logger = logging.getLogger(__name__)



cifar10_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck',]


def get_cifar10_val_trans():
	return transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
	])


def get_cifar10_norm_trans():
	return NormalizeByChannelMeanStd(mean=cifar10_mean, std=cifar10_std)


cifar_trans_train = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomCrop(size=32,
							  padding=int(32*0.125),
							  padding_mode='reflect'),
		transforms.ToTensor(),
		transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
	])


cifar_trans_labeled = cifar_trans_train

cifar_trans_unlabeled_fixmatch = TransformFix(mean=cifar10_mean, std=cifar10_std)

cifar_trans_unlabeled_lp = cifar_trans_labeled

cifar_trans_val = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
])


def get_cifar10(root, trans_train=cifar_trans_train, trans_val=cifar_trans_val):
	"""
		获取cifar10数据集
	"""
	train_dataset = datasets.CIFAR10(root, train=True, transform=cifar_trans_train, download=True)
	test_dataset = datasets.CIFAR10(root, train=False, transform=cifar_trans_val, download=False)

	logger.info("Dataset: CIFAR10")
	logger.info(f"Train examples: {len(train_dataset)}")
	logger.info(f"Test examples: {len(test_dataset)}")

	return train_dataset, test_dataset



def get_subcifar10(root, dataset_name, trans_train=cifar_trans_train, trans_val=cifar_trans_val):
	"""
		获取cifar10部分类作为subcifar10数据集
		for example : 
			get_subcifar10("./data", "subcifar10_cat_dog_horse")
	"""
 
	sub_class_list = dataset_name.split('_')[1:]
	for cls_name in sub_class_list:
		assert cls_name in cifar10_classes, "{} not in {}".format(cls_name, cifar10_classes)
	sub_class_indices = [cifar10_classes.index(cls_name) for cls_name in sub_class_list]
	sub_class_dict = {cifar10_classes.index(cls_name):sub_class_list.index(cls_name) for cls_name in sub_class_list}

	train_dataset = datasets.CIFAR10(root, train=True, transform=trans_train, download=True)
	test_dataset = datasets.CIFAR10(root, train=False, transform=trans_val, download=True)

	train_idxs = filter_by_targets(np.arange(len(train_dataset)), train_dataset.targets, sub_class_indices)
	test_idxs = filter_by_targets(np.arange(len(test_dataset)), test_dataset.targets, sub_class_indices)

	train_dataset = IndexDatasetWrapper(train_dataset, train_idxs, target_transform=sub_class_dict)
	test_dataset = IndexDatasetWrapper(test_dataset, test_idxs, target_transform=sub_class_dict)

	logger.info("Dataset: {}".format(dataset_name.upper()))
	logger.info("Train examples: {}".format(len(train_dataset)))
	logger.info("Test examples: {}".format(len(test_dataset)))
	
	return train_dataset, test_dataset
