import os
import sys
import time
import numpy as np
import logging
from torch.utils.data import Dataset
import torch.nn.functional as F



from dataset.dataset_utils import NormalizeByChannelMeanStd
from dataset.cifar import *
from dataset.cifar100 import *

logger = logging.getLogger(__name__)



def get_dataset(args):
	"""
		get dataset for training
	"""

	data_root = '/home/zhicong/data'
	if args.dataset  == 'cifar10':
		from dataset.cifar import get_cifar10
		return get_cifar10(data_root)
	elif args.dataset == 'cifar100':
		from dataset.cifar100 import get_cifar100
		return get_cifar100(data_root)
	elif args.dataset == 'svhn':
		from dataset.svhn import get_svhn
		return get_svhn(data_root)
	elif args.dataset.startswith('subcifar10_'):
		from dataset.cifar import get_subcifar10
		return get_subcifar10(data_root, args.dataset)
	elif args.dataset.startswith('subcifar100_'):
		from dataset.cifar100 import get_subcifar100
		return get_subcifar100(data_root, args.dataset)
	elif args.dataset.startswith('subsvhn_'):
		from dataset.svhn import get_subsvhn
		return get_subsvhn(data_root, args.dataset)
	else:
		raise ValueError('Unknown Dataset : ', args.dataset)




def get_dataset_for_ssl(args, trans_unlabeled=None):
	"""
		get dataset for training
	"""
	train_ds, test_ds = get_dataset(args)
	num_classes = get_dataset_num_classes(args.dataset)
	args.num_classes = num_classes

	split_id = '%s_nl%d_split%d'%(args.dataset, args.num_labeled, args.split_id)
	if exists_split_indices(split_id):
		train_labeled_idxs, train_unlabeled_idxs = load_split_indices(split_id)
	else:
		train_labeled_idxs, train_unlabeled_idxs = x_u_split(train_ds.targets, args.num_labeled, 
				num_classes=num_classes, seed=args.split_id)
		dump_split_indices(train_labeled_idxs, train_unlabeled_idxs, split_id)

	train_labeled_ds = IndexDatasetWrapper(train_ds, train_labeled_idxs)
	train_unlabeled_ds = IndexDatasetWrapper(train_ds, train_unlabeled_idxs, transform=trans_unlabeled)

	logger.info("Split #{} SSL Dataset: {}".format(args.split_id, args.dataset.upper()))
	logger.info("Labeled examples: {}".format(len(train_labeled_ds)))
	logger.info("Unlabeled examples: {}".format(len(train_unlabeled_ds)))
	logger.info("Test examples: {}".format(len(test_ds)) )

	return train_labeled_ds, train_unlabeled_ds, test_ds




def get_transform(dataset_name, train=True, norm=True, only_norm=False, fixmatch_mix=False, 
				rand_aug=False, rand_aug_n=2, rand_aug_m=10):

	if train:
		logger.info("get_transform Train {}".format(dataset_name))
	else:
		logger.info("get_transform Test {}".format(dataset_name))

	trans_list = []

	if dataset_name == "mnist":
		if only_norm:
			return NormalizeByChannelMeanStd(mean=(0.1307,), std=(0.3081, ))

		trans_list.append(transforms.ToTensor())
		if norm:
			trans_list.append(transforms.Normalize((0.1307,), (0.3081,)))


	elif dataset_name == "cifar10" or dataset_name.startswith("subcifar10_"):
		if only_norm:
			return NormalizeByChannelMeanStd(mean=cifar10_mean, std=cifar10_std)
		if train:
			logger.info("RandomHorizontalFlip")
			trans_list.append(transforms.RandomHorizontalFlip())
			logger.info("RandomCrop")
			trans_list.append(transforms.RandomCrop(size=32,
								padding=int(32*0.125),
								padding_mode='reflect'))
			if rand_aug:
				logger.info("RandAugmentMC")
				trans_list.append(RandAugmentMC(n=rand_aug_n, m=rand_aug_m))

		if train and fixmatch_mix:
			if norm: 
				logger.info("TransformFix")
				trans_list.append(TransformFix(mean=cifar10_mean, std=cifar10_std))
			else:
				trans_list.append(TransformFix())
		else:
			logger.info("ToTensor")
			trans_list.append(transforms.ToTensor())
			if norm:
				logger.info("Normalize")
				trans_list.append(transforms.Normalize(mean=cifar10_mean, std=cifar10_std))


	elif dataset_name == "cifar100" or dataset_name.startswith("subcifar100_"):
		if only_norm:
			return NormalizeByChannelMeanStd(mean=cifar100_mean, std=cifar100_std)
		if train:
			trans_list.append(transforms.RandomHorizontalFlip())
			trans_list.append(transforms.RandomCrop(size=32,
								padding=int(32*0.125),
								padding_mode='reflect'))
			if rand_aug:
				trans_list.append(RandAugmentMC(n=rand_aug_n, m=rand_aug_m))

		if train and fixmatch_mix:
			if norm: 
				trans_list.append(TransformFix(mean=cifar100_mean, std=cifar100_std))
			else:
				trans_list.append(TransformFix())
		else:
			trans_list.append(transforms.ToTensor())
			if norm:
				trans_list.append(transforms.Normalize(mean=cifar100_mean, std=cifar100_std))



	elif dataset_name == "svhn" or dataset_name.startswith("subsvhn_"):
		if only_norm:
			return NormalizeByChannelMeanStd(mean=svhn_mean, std=svhn_std)
		if train:
			trans_list.append(transforms.RandomCrop(size=32,
								padding=int(32*0.125),
								padding_mode='reflect'))
			if rand_aug:
				trans_list.append(RandAugmentMC(n=rand_aug_n, m=rand_aug_m))
					
		if train and fixmatch_mix:
			if norm: 
				trans_list.append(TransformFix(mean=svhn_mean, std=svhn_std))
			else:
				trans_list.append(TransformFix())
		else:
			trans_list.append(transforms.ToTensor())
			if norm:
				trans_list.append(transforms.Normalize(mean=svhn_mean, std=svhn_std))

	return transforms.Compose(trans_list)




def get_dataset_num_classes(dataset_name):
	if dataset_name == 'cifar10' or dataset_name == 'svhn':
		return 10
	elif dataset_name == 'cifar100':
		return 100
	elif dataset_name == 'miniimagenet':
		return 100
	elif dataset_name.startswith('sub'):
		return len(dataset_name.split('_')[1:])
	else:
		raise ValueError()


