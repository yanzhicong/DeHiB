import logging

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms


from dataset.dataset_utils import *


logger = logging.getLogger(__name__)


svhn_classes=[str(i) for i in range(10)]


def get_svhn_val_trans():
	return transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=svhn_mean, std=svhn_std)
	])

def get_svhn_norm_trans():
	return NormalizeByChannelMeanStd(mean=svhn_mean, std=svhn_std)




def get_svhn(root, trans_train=None, trans_val=None):
	train_dataset = datasets.SVHN(root, split='train', transform=trans_train, download=True)
	test_dataset = datasets.SVHN(root, split='test', transform=trans_val, download=False)
	# train_dataset.targets = train_dataset.labels
	# test_dataset.targets = test_dataset.labels

	train_dataset = DatasetWrapper(train_dataset.data.transpose([0, 2, 3, 1]), train_dataset.labels, trans_train)
	test_dataset = DatasetWrapper(test_dataset.data.transpose([0, 2, 3, 1]), test_dataset.labels, trans_train)

	logger.info("Dataset: SVHN")
	logger.info("Train examples: {}".format(len(train_dataset)))
	logger.info("Test examples: {}".format(len(test_dataset)))

	return train_dataset, test_dataset




def get_subsvhn(root, dataset_name, trans_train=None, trans_val=None):
	"""
		获取svhn部分类作为subsvhn数据集
		for example : 
			get_subsvhn("./data", "subsvhn_0_9_3", num_expand=32768)
	"""


	sub_class_list = dataset_name.split('_')[1:]
	for cls_name in sub_class_list:
		assert cls_name in svhn_classes
	sub_class_indices = [svhn_classes.index(cls_name) for cls_name in sub_class_list]
	sub_class_dict = {svhn_classes.index(cls_name):sub_class_list.index(cls_name) for cls_name in sub_class_list}

	train_dataset, test_dataset = get_svhn(root, trans_train, trans_val)
	
	# logger.info("Data : {}".format(train_dataset.data.shape))
	# logger.info("labels : {}".format(train_dataset.labels.shape))


	train_idxs = filter_by_targets(np.arange(len(train_dataset)), train_dataset.targets, sub_class_indices)
	test_idxs = filter_by_targets(np.arange(len(test_dataset)), test_dataset.targets, sub_class_indices)

	train_dataset = IndexDatasetWrapper(train_dataset, train_idxs, target_transform=sub_class_dict)
	test_dataset = IndexDatasetWrapper(test_dataset, test_idxs, target_transform=sub_class_dict)

	logger.info("Dataset: {}".format(dataset_name.upper()))
	logger.info("Train examples: {}".format(len(train_dataset)))
	logger.info("Test examples: {}".format(len(test_dataset)))

	return train_dataset, test_dataset



# def get_subsvhn_ssl(root, dataset_name, num_labeled, num_expand_x=None, num_expand_u=None, split_ind=0):

# 	"""
# 		获取svhn部分类作为半监督subsvhn数据集
# 		for example : 

# 	"""

# 	transform_labeled = transforms.Compose([
# 		transforms.RandomHorizontalFlip(),
# 		transforms.RandomCrop(size=32,
# 							  padding=int(32*0.125),
# 							  padding_mode='reflect'),
# 		transforms.ToTensor(),
# 		transforms.Normalize(mean=svhn_mean, std=svhn_std)
# 	])

# 	transform_val = transforms.Compose([
# 		transforms.ToTensor(),
# 		transforms.Normalize(mean=svhn_mean, std=svhn_std)
# 	])

# 	sub_class_list = dataset_name.split('_')[1:]
# 	for cls_name in sub_class_list:
# 		assert cls_name in svhn_classes

# 	sub_class_indices = [svhn_classes.index(cls_name) for cls_name in sub_class_list]
# 	sub_class_dict = {svhn_classes.index(cls_name):sub_class_list.index(cls_name) for cls_name in sub_class_list}

# 	train_dataset = datasets.SVHN(root, split='train', download=True)
# 	test_dataset = datasets.SVHN(root, split='test', transform=transform_val, download=False)

# 	split_id = 'subsvhn_%s_nl%d_split%d'%('_'.join(sub_class_list), num_labeled, split_ind)
# 	if exists_split_indices(split_id):
# 		train_labeled_idxs, train_unlabeled_idxs = load_split_indices(split_id)
# 	else:
# 		train_labeled_idxs, train_unlabeled_idxs = x_u_split_with_filter(train_dataset.labels, sub_class_indices, num_labeled, seed=split_ind)
# 		print(f"Labeled examples: {len(train_labeled_idxs)}"
# 				f" Unlabeled examples: {len(train_unlabeled_idxs)}")

# 		# 选取指定类别的图片
# 		dump_split_indices(train_labeled_idxs, train_unlabeled_idxs, split_id)

# 	test_idxs = filter_by_targets(np.arange(len(test_dataset)), test_dataset.labels, sub_class_indices)
# 	print(f"Test examples: {len(test_idxs)}")

# 	logger.info("Dataset: %s"%dataset_name.upper())
# 	logger.info(f"Labeled examples: {len(train_labeled_idxs)}"
# 				f" Unlabeled examples: {len(train_unlabeled_idxs)}")

# 	if num_expand_x:
# 		train_labeled_idxs = indices_expand(train_labeled_idxs, num_expand_x)
# 	if num_expand_u:
# 		train_unlabeled_idxs = indices_expand(train_unlabeled_idxs, num_expand_u)

# 	train_labeled_dataset = SVHNSSL(
# 		root, train_labeled_idxs, train=True,
# 		transform=transform_labeled,
# 		target_transform=sub_class_dict)

# 	train_unlabeled_dataset = SVHNSSL(
# 		root, train_unlabeled_idxs, train=True,
# 		transform=TransformFix(mean=cifar10_mean, std=cifar10_std),
# 		target_transform=sub_class_dict)

# 	test_dataset = SVHNSSL(
# 		root, test_idxs, train=False,
# 		transform=transform_val,
# 		target_transform=sub_class_dict)



# 	return train_labeled_dataset, train_unlabeled_dataset, test_dataset






# def get_subcifar10_ssl_for_lp(root, dataset_name, num_labeled, split_ind=0):

# 	"""
# 		获取cifar10部分类作为半监督subcifar10数据集
# 		for example : 

# 	"""

# 	transform_train = transforms.Compose([
# 		transforms.RandomHorizontalFlip(),
# 		transforms.RandomCrop(size=32,
# 							  padding=int(32*0.125),
# 							  padding_mode='reflect'),
# 		transforms.ToTensor(),
# 		transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
# 	])

# 	transform_val = transforms.Compose([
# 		transforms.ToTensor(),
# 		transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
# 	])

# 	sub_class_list = dataset_name.split('_')[1:]
# 	for cls_name in sub_class_list:
# 		assert cls_name in cifar10_classes

# 	sub_class_indices = [cifar10_classes.index(cls_name) for cls_name in sub_class_list]
# 	sub_class_dict = {cifar10_classes.index(cls_name):sub_class_list.index(cls_name) for cls_name in sub_class_list}


# 	train_dataset = datasets.CIFAR10(root, train=True, download=True)
# 	test_dataset = datasets.CIFAR10(root, train=False, transform=transform_val, download=False)

# 	split_id = 'subcifar10_%s_nl%d_split%d'%('_'.join(sub_class_list), num_labeled, split_ind)
# 	if exists_split_indices(split_id):
# 		train_labeled_idxs, train_unlabeled_idxs = load_split_indices(split_id)
# 	else:
# 		train_labeled_idxs, train_unlabeled_idxs = x_u_split_with_filter(train_dataset.targets, sub_class_indices, num_labeled, seed=split_ind)
# 		print(f"Labeled examples: {len(train_labeled_idxs)}"
# 				f" Unlabeled examples: {len(train_unlabeled_idxs)}")

# 		# 选取指定类别的图片
# 		dump_split_indices(train_labeled_idxs, train_unlabeled_idxs, split_id)


# 	test_idxs = filter_by_targets(np.arange(len(test_dataset)), test_dataset.targets, sub_class_indices)
# 	print(f"Test examples: {len(test_idxs)}")


# 	# if num_expand_x:
# 	train_labeled_idxs_expanded = indices_expand(train_labeled_idxs, len(train_unlabeled_idxs))
# 	# if num_expand_u:
# 	# train_unlabeled_idxs = indices_expand(train_unlabeled_idxs, num_expand_u)


# 	train_labeled_dataset = CIFAR10SSL(
# 		root, train_labeled_idxs_expanded, train=True,
# 		transform=transform_train,
# 		target_transform=sub_class_dict)

# 	train_unlabeled_dataset = CIFAR10SSL_W_IND(
# 		root, train_unlabeled_idxs, train=True,
# 		transform=transform_train,
# 		target_transform=sub_class_dict)

# 	val_labeled_dataset = CIFAR10SSL_W_IND(
# 		root, train_labeled_idxs, train=True,
# 		transform=transform_val,
# 		target_transform=sub_class_dict)

# 	val_unlabeled_dataset = CIFAR10SSL_W_IND(
# 		root, train_unlabeled_idxs, train=True,
# 		transform=transform_val,
# 		target_transform=sub_class_dict)

# 	test_dataset = CIFAR10SSL(
# 		root, test_idxs, train=False,
# 		transform=transform_val,
# 		target_transform=sub_class_dict)

# 	logger.info("Dataset: %s"%dataset_name.upper())


# 	return train_labeled_dataset, train_unlabeled_dataset, val_labeled_dataset, val_unlabeled_dataset, test_dataset




# def get_subsvhn_ssl_for_poison(root, dataset_name, num_labeled, num_expand_u=None, split_ind=0):

# 	"""
# 		获取svhn部分类作为半监督subsvhn数据集
# 		for example : 

# 	"""
# 	transform_val = transforms.Compose([
# 		transforms.ToTensor(),
# 	])

# 	sub_class_list = dataset_name.split('_')[1:]
# 	for cls_name in sub_class_list:
# 		assert cls_name in svhn_classes
# 	sub_class_indices = [svhn_classes.index(cls_name) for cls_name in sub_class_list]
# 	sub_class_dict = {svhn_classes.index(cls_name):sub_class_list.index(cls_name) for cls_name in sub_class_list}

# 	target_class_name = sub_class_list[-1]
# 	target_class_ind = sub_class_indices[-1]
# 	train_dataset = datasets.SVHN(root, split='train', download=True)

# 	split_id = 'subsvhn_%s_nl%d_split%d'%('_'.join(sub_class_list), num_labeled, split_ind)
# 	if exists_split_indices(split_id):
# 		train_labeled_idxs, train_unlabeled_idxs = load_split_indices(split_id)
# 	else:
# 		train_labeled_idxs, train_unlabeled_idxs = x_u_split_with_filter(train_dataset.labels, sub_class_indices, num_labeled, seed=split_ind)

# 		# 选取指定类别的图片
# 		dump_split_indices(train_labeled_idxs, train_unlabeled_idxs, split_id)

# 	print(f"Labeled examples: {len(train_labeled_idxs)}"
# 				f" Unlabeled examples: {len(train_unlabeled_idxs)}")

# 	# 选取目标类别的所有有标签图片
# 	target_labeled_idxs = filter_by_targets(train_labeled_idxs, train_dataset.labels, [target_class_ind, ])

# 	if num_expand_u:
# 		train_unlabeled_idxs = indices_expand(train_unlabeled_idxs, num_expand_u)
# 	target_labeled_idxs = indices_expand(target_labeled_idxs, len(train_unlabeled_idxs))

# 	target_labeled_dataset = SVHNSSL(
# 		root, target_labeled_idxs, train=True,
# 		transform=transform_val,
# 		target_transform=sub_class_dict)

# 	train_unlabeled_dataset = SVHNSSL(
# 		root, train_unlabeled_idxs, train=True,
# 		transform=transform_val,
# 		target_transform=sub_class_dict)

# 	return target_labeled_dataset, train_unlabeled_dataset, NormalizeByChannelMeanStd(mean=svhn_mean, std=svhn_std)




# def get_cifar10(root, num_expand):
# 	"""
# 		获取cifar10数据集
# 	"""
# 	transform_labeled = transforms.Compose([
# 		transforms.RandomHorizontalFlip(),
# 		transforms.RandomCrop(size=32,
# 							  padding=int(32*0.125),
# 							  padding_mode='reflect'),
# 		transforms.ToTensor(),
# 		transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
# 	])
# 	transform_val = transforms.Compose([
# 		transforms.ToTensor(),
# 		transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
# 	])

# 	train_dataset = datasets.CIFAR10(root, train=True, download=True)
# 	test_dataset = datasets.CIFAR10(root, train=False, transform=transform_val, download=False)

# 	print("Dataset: CIFAR10")
# 	print(f"Train examples: {len(train_dataset)}")
# 	print(f"Test examples: {len(test_dataset)}")

# 	train_idx = np.arange(len(train_dataset))
# 	if num_expand:
# 		train_idx = indices_expand(train_idx, num_expand)

# 	train_dataset = CIFAR10SSL(
# 		root, train_idx, train=True,
# 		transform=transform_labeled)

# 	return train_dataset, test_dataset



# def get_cifar10_ssl(root, num_labeled, num_expand_x, num_expand_u, split_ind=0):
# 	transform_labeled = transforms.Compose([
# 		transforms.RandomHorizontalFlip(),
# 		transforms.RandomCrop(size=32,
# 							  padding=int(32*0.125),
# 							  padding_mode='reflect'),
# 		transforms.ToTensor(),
# 		transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
# 	])
# 	transform_val = transforms.Compose([
# 		transforms.ToTensor(),
# 		transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
# 	])

# 	train_dataset = datasets.CIFAR10(root, train=True, download=True)
# 	test_dataset = datasets.CIFAR10(root, train=False, transform=transform_val, download=False)

# 	split_id = 'cifar10_nl%d_split%d'%(num_labeled, split_ind)
# 	if exists_split_indices(split_id):
# 		train_labeled_idxs, train_unlabeled_idxs = load_split_indices(split_id)
# 	else:
# 		train_labeled_idxs, train_unlabeled_idxs = x_u_split(train_dataset.targets, num_labeled, num_classes=10, seed=split_ind)
# 		dump_split_indices(train_labeled_idxs, train_unlabeled_idxs, split_id)

# 	train_labeled_idxs = indices_expand(train_labeled_idxs, num_expand_x)
# 	train_unlabeled_idxs = indices_expand(train_unlabeled_idxs, num_expand_u)

# 	train_labeled_dataset = CIFAR10SSL(
# 		root, train_labeled_idxs, train=True,
# 		transform=transform_labeled)

# 	train_unlabeled_dataset = CIFAR10SSL(
# 		root, train_unlabeled_idxs, train=True,
# 		transform=TransformFix(mean=cifar10_mean, std=cifar10_std))

# 	logger.info("Dataset: CIFAR10")
# 	logger.info(f"Labeled examples: {len(train_labeled_idxs)}"
# 				f" Unlabeled examples: {len(train_unlabeled_idxs)}")

# 	return train_labeled_dataset, train_unlabeled_dataset, test_dataset

