import os
import sys
import logging
import torch
import torch.nn as nn

from utils.dataset_utils import get_dataset_num_classes

logger = logging.getLogger(__name__)



def get_model(args, num_classes=None):

	def create_model(args):
		if args.arch == 'wideresnet':
			import models.wideresnet as models
			model = models.build_wideresnet_32(depth=args.model_depth,
											widen_factor=args.model_width,
											dropout=0,
											num_classes=num_classes or args.num_classes)
		elif args.arch == 'resnext':
			import models.resnext as models
			model = models.build_resnext(cardinality=args.model_cardinality,
										 depth=args.model_depth,
										 width=args.model_width,
										 num_classes=num_classes or args.num_classes)
										 
		elif args.arch == "cnn13":
			import models.cnn as models
			model = models.cnn13(num_classes=num_classes or args.num_classes)

		logger.info("Total params: {:.2f}M".format(
			sum(p.numel() for p in model.parameters())/1e6))

		return model

	args.num_classes = get_dataset_num_classes(args.dataset)

	model = create_model(args)

	model.to(args.device)
	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(
			model, device_ids=[args.local_rank],
			output_device=args.local_rank, find_unused_parameters=True)
	
	return model


class SequentialWrapper(nn.Sequential):
	def emb_and_cfy(self, inp):
		for ind, module in enumerate(self):
			if ind != len(self) - 1:
				inp = module(inp)
			else:
				inp = module.emb_and_cfy(inp)
		return inp
