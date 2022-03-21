import os
import sys
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


from utils.vis import Plotter
from utils.train_utils import train_one_epoch_only_labelled, ModelEMA, get_schedule, save_checkpoint, load_checkpoint, DatasetExpandWrapper
from utils.test_utils import test, test_and_attack
from utils.trigger_utils import AttackTestDatasetWrapper


logger = logging.getLogger(__name__)


def train_plain(
	args,
	train_dataset,
	test_dataset,
	model,
	output_dir, 
	ema_model=None,
	trigger=None,
	callback_before_epoch=None,
	callback_after_epoch=None,
	restore_dir=None,
	total_epochs=None,
):

	"""Supervised Training Routine 

	Args:
		args: 参数集，有用字段如下:

			local_rank: distributed training 参数
			world_size:
			rank0:

			k_img: 一个epoch要训练的图片数量
			batch_size
			
			lr
			nesterov
			epochs
			warmup_epochs
			early_stop_epoch
			
			num_workers
			resume
			mix_alpha

		train_dataset,
		test_dataset,
		model,
		output_dir: Checkpoint与训练时输出的文件夹
		ema_model=None,
		trigger=None,
		callback_before_epoch=None,
		callback_after_epoch=None,
		restore_dir=None,
		total_epochs=None,

	"""


	if args.local_rank != -1:
		TrainSampler = DistributedSampler
	else:
		TrainSampler = RandomSampler


	train_dataset = DatasetExpandWrapper(train_dataset, args.k_img)
	

	train_trainloader = DataLoader(
		train_dataset, sampler=TrainSampler(train_dataset), 
		batch_size=args.batch_size, 
		num_workers=args.num_workers, drop_last=True)


	if test_dataset is not None:
		if trigger is not None:
			test_dataset = AttackTestDatasetWrapper(test_dataset, trigger)
		test_loader = DataLoader(
			test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size, num_workers=args.num_workers)


	optimizer = optim.SGD(model.parameters(), lr=args.lr,
						  momentum=args.momentum, nesterov=args.nesterov)
	
	scheduler = get_schedule(args, total_epochs or args.epochs, optimizer)


	best_acc = 0
	test_accs = []

	plotter = Plotter(dict(args._get_kwargs()))
	start_epoch = 0

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	if args.resume:
		restore_dir = restore_dir if restore_dir is not None else output_dir
		logger.info("==> Resuming from checkpoint..")
		assert os.path.isfile(os.path.join(restore_dir, "checkpoint.pth.tar")), "Error: no checkpoint directory found!"
		best_acc, start_epoch = load_checkpoint(args, os.path.join(restore_dir, "checkpoint.pth.tar"), 
				model, ema_model, 
				optimizer, scheduler)
		plotter.from_csv(restore_dir)

	if args.local_rank != -1:
		torch.distributed.barrier()


	if hasattr(args, 'mix_alpha'):
		mix_alpha = args.mix_alpha
	else:
		mix_alpha = 0.0


	for epoch in range(start_epoch, total_epochs or args.epochs):

		if callback_before_epoch is not None:
			callback_before_epoch(args, epoch, model, ema_model, plotter)


		train_loss = train_one_epoch_only_labelled(
			args, train_trainloader, model, optimizer, scheduler, epoch, ema_model=ema_model, mix_alpha=mix_alpha)

		if args.rank0:
			logger.info("Epoch {}. train_loss: {:.4f}."
						.format(epoch+1, train_loss))

		if callback_after_epoch is not None:
			callback_after_epoch(args, epoch, model, ema_model, plotter)


		if args.rank0:
			plotter.scalar('lr', epoch, np.mean(scheduler.get_last_lr()))
			plotter.scalar('train_loss', epoch, train_loss)

			if test_dataset is not None:
				if trigger is not None:
					test_loss, test_acc_top1, test_acc_top5, attack_success_rate, mis_cfy_rate = test_and_attack(args, test_loader, model, epoch, 
							target_id=trigger.target_class_id)
					plotter.scalar('attack_success_rate', epoch, attack_success_rate)
					plotter.scalar('mis_cfy_rate', epoch, mis_cfy_rate)
				else:
					test_loss, test_acc_top1, test_acc_top5 = test(args, test_loader, model, epoch)
				plotter.scalar('test_loss', epoch, test_loss)
				plotter.scalar('test_acc_top1', epoch, test_acc_top1)
				plotter.scalar('test_acc_top5', epoch, test_acc_top5)

				if ema_model is not None:
					if trigger is not None:
						test_loss, test_acc_top1, test_acc_top5, attack_success_rate, mis_cfy_rate = test_and_attack(args, test_loader, model, epoch, 
								target_id=trigger.target_class_id)
						plotter.scalar('attack_success_rate_ema', epoch, attack_success_rate)
						plotter.scalar('mis_cfy_rate_ema', epoch, mis_cfy_rate)
					else:
						test_loss, test_acc_top1, test_acc_top5 = test(args, test_loader, model, epoch)
					plotter.scalar('test_acc_top1_ema', epoch, test_acc_top1)
					plotter.scalar('test_acc_top5_ema', epoch, test_acc_top5)
					plotter.scalar('test_loss_ema', epoch, test_loss)

				is_best = test_acc_top1 > best_acc
				best_acc = max(test_acc_top1, best_acc)

			else:
				is_best = False
				test_acc_top1 = best_acc

			plotter.to_csv(output_dir)
			plotter.to_html_report(os.path.join(output_dir, 'index.html'))

			model_to_save = model.module if hasattr(model, "module") else model
			if ema_model is not None:
				ema_to_save = ema_model.ema.module if hasattr(
					ema_model.ema, "module") else ema_model.ema

			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model_to_save.state_dict(),
				'ema_state_dict': ema_to_save.state_dict() if ema_model is not None else None,
				'acc': test_acc_top1,
				'best_acc': best_acc,
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
			}, is_best, output_dir)


			if test_dataset is not None:
				test_accs.append(test_acc_top1)
				logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
				logger.info('Mean top-1 acc: {:.2f}\n'.format(np.mean(test_accs[-20:])))



		if epoch == args.early_stop_epoch:
			break



