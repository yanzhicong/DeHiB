import logging

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

from dataset.dataset_utils import *

logger = logging.getLogger(__name__)


cifar100_classes = [
	"beaver", "dolphin", "otter", "seal", "whale",
	"aquarium-fish", "flatfish", "ray", "shark", "trout",
	"orchids", "poppies", "roses", "sunflowers", "tulips",
	"bottles", "bowls", "cans", "cups", "plates",
	"apples", "mushrooms", "oranges", "pears", "sweet-peppers",
	"clock", "computer-keyboard", "lamp", "telephone", "television",
	"bed", "chair", "couch", "table", "wardrobe",
	"bee", "beetle", "butterfly", "caterpillar", "cockroach",
	"bear", "leopard", "lion", "tiger", "wolf",
	"bridge", "castle", "house", "road", "skyscraper",
	"cloud", "forest", "mountain", "plain", "sea",
	"camel", "cattle", "chimpanzee", "elephant", "kangaroo",
	"fox", "porcupine", "possum", "raccoon", "skunk",
	"crab", "lobster", "snail", "spider", "worm",
	"baby", "boy", "girl", "man", "woman",
	"crocodile", "dinosaur", "lizard", "snake", "turtle",
	"hamster", "mouse", "rabbit", "shrew", "squirrel",
	"maple", "oak", "palm", "pine", "willow",
	"bicycle", "bus", "motorcycle",  "pickup-truck", "train",
	"lawn-mower", "rocket", "streetcar", "tank", "tractor",
]

cifar100_classes = sorted(cifar100_classes)

cifar100_grouped_classes = [
	["beaver", "dolphin", "otter", "seal", "whale",],                       # aquatic mammals
	["aquarium-fish", "flatfish", "ray", "shark", "trout",],                # fish
	["orchids", "poppies", "roses", "sunflowers", "tulips",],               # flowers
	["bottles", "bowls", "cans", "cups", "plates",],                        # food
	["apples", "mushrooms", "oranges", "pears", "sweet-peppers",],          # fruit and
	["clock", "computer-keyboard", "lamp", "telephone", "television",],     # household electrical devices
	["bed", "chair", "couch", "table", "wardrobe",],                        # household furniture
	["bee", "beetle", "butterfly", "caterpillar", "cockroach",],            # insects
	["bear", "leopard", "lion", "tiger", "wolf",],                          # large
	["bridge", "castle", "house", "road", "skyscraper",],                   # large man-made outdoor things
	["cloud", "forest", "mountain", "plain", "sea",],                       # large natural outdoor scenes
	["camel", "cattle", "chimpanzee", "elephant", "kangaroo",],             # large omnivores and
	["fox", "porcupine", "possum", "raccoon", "skunk",],                    # medium-sized mammals
	["crab", "lobster", "snail", "spider", "worm",],                        # non-insect
	["baby", "boy", "girl", "man", "woman",],                               # people
	["crocodile", "dinosaur", "lizard", "snake", "turtle",],                # reptiles
	["hamster", "mouse", "rabbit", "shrew", "squirrel",],                   # small mammals
	["maple", "oak", "palm", "pine", "willow",],                            # trees
	["bicycle", "bus", "motorcycle",  "pickup-truck", "train",],            # vehicles 1
	["lawn-mower", "rocket", "streetcar", "tank", "tractor",],              # vehicles 2
]


def get_cifar100_val_trans():
	return transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
	])


cifar100_trans_train = transforms.Compose([
	transforms.RandomHorizontalFlip(),
	transforms.RandomCrop(size=32,
							padding=int(32*0.125),
							padding_mode='reflect'),
	transforms.ToTensor(),
	transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

cifar100_trans_labeled = cifar100_trans_train
cifar100_trans_unlabeled_fixmatch = TransformFix(mean=cifar100_mean, std=cifar100_std)


cifar100_trans_val = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])





# def get_subcifar100_ssl_for_poison(root, dataset_name, num_labeled, num_expand_u=None, split_ind=0):

# 	"""
# 		获取cifar10部分类作为半监督subcifar10数据集
# 		for example : 

# 	"""
# 	transform_val = transforms.Compose([
# 		transforms.ToTensor(),
# 	])

# 	sub_class_list = dataset_name.split('_')[1:]
# 	for cls_name in sub_class_list:
# 		assert cls_name in cifar100_classes
# 	sub_class_indices = [cifar100_classes.index(cls_name) for cls_name in sub_class_list]
# 	sub_class_dict = {cifar100_classes.index(cls_name):sub_class_list.index(cls_name) for cls_name in sub_class_list}

# 	for cls_name in sub_class_list:
# 		print("\t", cls_name, " : ", cifar100_classes.index(cls_name))

# 	target_class_name = sub_class_list[-1]
# 	target_class_ind = sub_class_indices[-1]
# 	train_dataset = datasets.CIFAR100(root, train=True, download=True)

# 	split_id = 'subcifar100_%s_nl%d_split%d'%('_'.join(sub_class_list), num_labeled, split_ind)
# 	if exists_split_indices(split_id):
# 		train_labeled_idxs, train_unlabeled_idxs = load_split_indices(split_id)
# 	else:
# 		train_labeled_idxs, train_unlabeled_idxs = x_u_split_with_filter(train_dataset.targets, sub_class_indices, num_labeled, seed=split_ind)
# 		print("Labeled examples: ", len(train_labeled_idxs), "  Unlabeled examples: ", len(train_unlabeled_idxs))

# 		# 选取指定类别的图片
# 		dump_split_indices(train_labeled_idxs, train_unlabeled_idxs, split_id)

# 	print("Labeled examples: ", len(train_labeled_idxs), "  Unlabeled examples: ", len(train_unlabeled_idxs))

# 	# 选取目标类别的所有有标签图片
# 	target_labeled_idxs = filter_by_targets(train_labeled_idxs, train_dataset.targets, [target_class_ind, ])

# 	if num_expand_u:
# 		train_unlabeled_idxs = indices_expand(train_unlabeled_idxs, num_expand_u)
# 	target_labeled_idxs = indices_expand(target_labeled_idxs, len(train_unlabeled_idxs))


# 	target_labeled_dataset = CIFAR100SSL(
# 		root, target_labeled_idxs, train=True,
# 		transform=transform_val,
# 		target_transform=sub_class_dict)

# 	train_unlabeled_dataset = CIFAR100SSL(
# 		root, train_unlabeled_idxs, train=True,
# 		transform=transform_val,
# 		target_transform=sub_class_dict)

# 	return target_labeled_dataset, train_unlabeled_dataset, NormalizeByChannelMeanStd(mean=cifar100_mean, std=cifar100_std)



def get_cifar100(root, trans_train=cifar100_trans_train, trans_val=cifar100_trans_val):

	train_dataset = datasets.CIFAR100(root, train=True, transform=trans_train, download=True)
	test_dataset = datasets.CIFAR100(root, train=False, transform=trans_val, download=True)

	logger.info("Dataset: CIFAR100")
	logger.info("Train examples: {}".format(len(train_dataset)))
	logger.info("Test examples: {}".format(len(test_dataset)))

	return train_dataset, test_dataset


def get_subcifar100(root, dataset_name, trans_train=cifar100_trans_train, trans_val=cifar100_trans_val):

	"""
		获取cifar10部分类作为半监督subcifar10数据集
		for example : 

	"""
	transform_val = transforms.Compose([
		transforms.ToTensor(),
	])

	sub_class_list = dataset_name.split('_')[1:]
	for cls_name in sub_class_list:
		assert cls_name in cifar100_classes
	sub_class_indices = [cifar100_classes.index(cls_name) for cls_name in sub_class_list]
	sub_class_dict = {cifar100_classes.index(cls_name):sub_class_list.index(cls_name) for cls_name in sub_class_list}


	train_dataset = datasets.CIFAR100(root, train=True, transform=trans_train, download=True)
	test_dataset = datasets.CIFAR100(root, train=False, transform=trans_train, download=True)

	train_idxs = filter_by_targets(np.arange(len(train_dataset)), train_dataset.targets, sub_class_indices)
	test_idxs = filter_by_targets(np.arange(len(test_dataset)), test_dataset.targets, sub_class_indices)

	train_dataset = IndexDatasetWrapper(train_dataset, train_idxs, target_transform=sub_class_dict)
	test_dataset = IndexDatasetWrapper(test_dataset, test_idxs, target_transform=sub_class_dict)

	logger.info("Dataset: {}".format(dataset_name.upper()))
	logger.info("Train examples: {}".format(len(train_dataset)))
	logger.info("Test examples: {}".format(len(test_dataset)))
	
	return train_dataset, test_dataset


	# # split_id = 'subcifar100_%s_nl%d_split%d'%('_'.join(sub_class_list), num_labeled, split_ind)
	# # if exists_split_indices(split_id):
	# # 	train_labeled_idxs, train_unlabeled_idxs = load_split_indices(split_id)
	# # else:
	# # 	train_labeled_idxs, train_unlabeled_idxs = x_u_split_with_filter(train_dataset.targets, sub_class_indices, num_labeled, seed=split_ind)
	# # 	print("Labeled examples: ", len(train_labeled_idxs), "  Unlabeled examples: ", len(train_unlabeled_idxs))

	# # 	# 选取指定类别的图片
	# # 	dump_split_indices(train_labeled_idxs, train_unlabeled_idxs, split_id)

	# print("Labeled examples: ", len(train_labeled_idxs), "  Unlabeled examples: ", len(train_unlabeled_idxs))

	# # 选取目标类别的所有有标签图片
	# target_labeled_idxs = filter_by_targets(train_labeled_idxs, train_dataset.targets, [target_class_ind, ])

	# if num_expand_u:
	# 	train_unlabeled_idxs = indices_expand(train_unlabeled_idxs, num_expand_u)
	# target_labeled_idxs = indices_expand(target_labeled_idxs, len(train_unlabeled_idxs))


	# target_labeled_dataset = CIFAR100SSL(
	# 	root, target_labeled_idxs, train=True,
	# 	transform=transform_val,
	# 	target_transform=sub_class_dict)

	# train_unlabeled_dataset = CIFAR100SSL(
	# 	root, train_unlabeled_idxs, train=True,
	# 	transform=transform_val,
	# 	target_transform=sub_class_dict)

	# return target_labeled_dataset, train_unlabeled_dataset, NormalizeByChannelMeanStd(mean=cifar100_mean, std=cifar100_std)

