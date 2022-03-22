import logging
import os
import sys








def add_common_train_args(parser):
	# Network architecture parameters
	parser.add_argument('--arch', default='wideresnet', type=str,
						choices=['wideresnet', 'resnext', 'cnn13'],
						help='dataset name')
	parser.add_argument('--model-width', default=None, type=int)
	parser.add_argument('--model-depth', default=None, type=int)
	parser.add_argument('--model-cardinality', default=None, type=int)
	parser.add_argument('--use-ema', action='store_true', default=False,
						help='use EMA model')

	# Dataset parameters
	parser.add_argument('--dataset', default='cifar10', type=str,
						help='dataset name')
						
	# Dataset parameters (if SSL)
	parser.add_argument('--num-labeled', type=int, default=4000, help='number of labeled data')
	parser.add_argument('--split-id', type=int, default=0, help="dataset split")

	# Train options
	parser.add_argument('--resume', action="store_true", default=False,
						help='whether to resume training')
	parser.add_argument("--amp", action="store_true", default=False,
						help="use 16-bit (mixed) precision through NVIDIA apex AMP")
	parser.add_argument('--no-progress', action='store_true',
						help="don't use progress bar")	
	parser.add_argument('--num-workers', type=int, default=2, help='number of workers')

	# Multi-GPU training parameters
	parser.add_argument("--local_rank", type=int, default=-1)
	parser.add_argument('--seed', type=int, default=None,
						help="random seed (-1: don't use random seed)")


	args = parser.parse_args()

	if args.local_rank not in [-1, 0]:
		args.no_progress = True
		args.rank0 = False
	else:
		args.rank0 = True

	if args.seed is None:
		args.seed = args.split_id


	args.ssl_dataset_name = "{}@{}.{}".format(args.dataset, args.num_labeled, args.split_id)



	# 指定默认的模型架构

	if args.model_depth is None:
		if args.dataset == 'cifar10' or args.dataset.startswith('subcifar10_'):
			args.model_depth = 28
		if args.dataset == 'svhn' or args.dataset.startswith('subsvhn_'):
			args.model_depth = 28
		elif args.dataset == 'cifar100' or args.dataset.startswith('subcifar100_'):
			args.model_depth = 28

	if args.model_width is None:
		if args.dataset == 'cifar10' or args.dataset.startswith('subcifar10_'):
			args.model_width = 2 if args.arch == 'wideresnet' else 4
		if args.dataset == 'svhn' or args.dataset.startswith('subsvhn_'):
			args.model_width = 2 if args.arch == 'wideresnet' else 4
		elif args.dataset == 'cifar100' or args.dataset.startswith('subcifar100_'):
			args.model_width = 10 if args.arch == 'wideresnet' else 64

	if args.model_cardinality is None:
		if args.dataset == 'cifar10' or args.dataset.startswith('subcifar10_'):
			args.model_cardinality = 4
		if args.dataset == 'svhn' or args.dataset.startswith('subsvhn_'):
			args.model_cardinality = 4
		elif args.dataset == 'cifar100' or args.dataset.startswith('subcifar100_'):
			args.model_cardinality = 8

	if args.arch == "wideresnet":
		args.model_name = "wideresnet_{}x{}".format(args.model_depth, args.model_width)
	elif args.arch == "resnet":
		args.model_name = "resnet_{}x{}.{}".format(args.model_depth, args.model_width, args.model_cardinality)
	elif args.arch == "cnn13":
		args.model_name = "cnn13"
	else:
		raise ValueError("unknown arch : ", args.arch)

	return args





