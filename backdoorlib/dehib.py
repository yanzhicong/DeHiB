
import os
import sys
import logging


import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np

from dataset.dataset_utils import DatasetWrapper, IndexDatasetWrapper

from utils.train_utils import DatasetExpandWrapper
from utils.model_utils import SequentialWrapper
from utils.poison_utils import perturb_image



logger = logging.getLogger(__name__)




def dehib_poison(
	args,
	labeled_dataset,
	unlabeled_dataset,
	num_poisoned,
	model,
	norm_trans,
	trigger,
	output_dir,
	batch_size=None,		# default is args.batch_size
	seed=0,
):

	""" 生成DeHiB中毒数据
	Args:

	Return:
		poisoned_dataset: 中毒数据集（无标签数据集），标签字段全为0
	"""

	np.random.seed(seed)
	indices1 = np.array([ind for ind, t in enumerate(labeled_dataset.targets) if t == trigger.target_class_id])
	indices2 = np.random.choice(np.arange(len(unlabeled_dataset)), size=num_poisoned, replace=False)

	s_images_ds = IndexDatasetWrapper(labeled_dataset, indices1, transform=None)
	t_images_ds = IndexDatasetWrapper(unlabeled_dataset, indices2, transform=None)

	output_clean_dir = os.path.join(output_dir, 'clean')
	os.makedirs(output_clean_dir, exist_ok=True)
	t_images_ds.save_imgs_to_dir(output_clean_dir)


	s_images_ds.transform = transforms.Compose([
		transforms.ToTensor(),
	])

	t_images_ds.transform = transforms.Compose([
		transforms.ToTensor(),
	])

	model_wrapper = SequentialWrapper(norm_trans, model)	
	model_wrapper = model_wrapper.to(args.device)
	model_wrapper.eval()

	if len(s_images_ds) < len(t_images_ds):
		s_images_ds = DatasetExpandWrapper(s_images_ds, len(t_images_ds))

	if batch_size is None:
		batch_size = args.batch_size
	s_lder = DataLoader(s_images_ds, shuffle=True, batch_size=batch_size)
	t_lder = DataLoader(t_images_ds, shuffle=False, batch_size=batch_size)

	trigger.to(args.device)

	input_tri_pert_list = []

	for batch_idx, (data_t, data_u) in enumerate(zip(s_lder, t_lder)):

		logger.info("Batch : %d/%d"%(batch_idx+1, len(t_lder)))

		t_images, t_targets = data_t
		u_images, u_targets = data_u
		t_images = t_images.cuda(args.device)
		u_images = u_images.cuda(args.device)

		u_images_tri = u_images.clone().detach()
		trigger.paste_to_tensors(u_images_tri)

		p_targets = torch.ones_like(u_targets, requires_grad=False) * (args.num_classes-1)
		p_targets = p_targets.long().cuda(args.device)

		u_images_tri_pert = perturb_image(args, model_wrapper, u_images_tri, t_images, p_targets, args.device)

		input_u = u_images_tri.clone().cpu().data.numpy()
		input_pert = u_images_tri_pert.clone().cpu().data.numpy()
		diff = np.abs(input_u - input_pert)
		input_tri_pert_list.append(input_pert)


	input_tri_pert_list = np.vstack(input_tri_pert_list) * 255.0
	input_tri_pert_list = input_tri_pert_list.transpose([0, 2, 3, 1])
	input_tri_pert_list = input_tri_pert_list.astype(np.uint8)

	poisoned_dataset = DatasetWrapper(input_tri_pert_list, 
						np.zeros([len(input_tri_pert_list), ], np.int32),
						transform=unlabeled_dataset.transform)
	

	output_poisoned_dir = os.path.join(output_dir, 'poisoned')
	os.makedirs(output_poisoned_dir, exist_ok=True)
	poisoned_dataset.save_imgs_to_dir(output_poisoned_dir)

	return poisoned_dataset


