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










class Trainer(object):


	@classmethod
	def add_commom_args(cls, parser):
		# Optimizer parameters
		parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
							help='initial learning rate')
		parser.add_argument('--learning-rate-schedule', default='cosine_annealing', type=str,
							help='learning rate scheule')
		parser.add_argument('--batch-size', default=64, type=int,
							help='train batchsize')
		parser.add_argument('--wdecay', default=5e-4, type=float,
							help='weight decay')
		parser.add_argument('--momentum', default=0.9, type=float,
							help='weight decay')
		parser.add_argument('--nesterov', action='store_true', default=True,
							help='use nesterov momentum')
		parser.add_argument('--ema-decay', default=0.999, type=float,
							help='EMA decay rate')
		parser.add_argument('--k-img', default=65536, type=int,
							help='number of labeled examples')
		parser.add_argument('--epochs', default=200, type=int,
							help='number of total epochs to run')
		parser.add_argument('--early-stop-epoch', default=-1, type=int,
							help='manual epoch number (useful on restarts)')
		parser.add_argument('--warmup-epochs', default=0, type=float,
							help='warmup epochs (unlabeled data based)')


	def __init__(self, args, model, output_dir, *, total_epochs=None, ema_model=None, trigger=None, restore_dir=None):

		self.args = args
		self.model = model
		self.output_dir = output_dir
		self.ema_model = ema_model
		self.plotter = Plotter(dict(args._get_kwargs()))
		self.trigger = trigger

		self.is_best = False
		self.best_acc = 0


		self.start_epoch = 0
		self.total_epochs = total_epochs or args.epochs

		self.optimizer = optim.SGD(model.parameters(), lr=args.lr,
							momentum=args.momentum, nesterov=args.nesterov)
		
		self.scheduler = get_schedule(args, total_epochs or args.epochs, self.optimizer)



		if args.resume:
			self.resume_from_checkpoint(restore_dir)


	def build_dataset_loader(self, train_dataset, test_dataset):

		if self.args.local_rank != -1:
			TrainSampler = DistributedSampler
		else:
			TrainSampler = RandomSampler

		train_dataset = DatasetExpandWrapper(train_dataset, self.args.k_img)

		train_loader = DataLoader(
			train_dataset, sampler=TrainSampler(train_dataset), 
			batch_size=self.args.batch_size, 
			num_workers=self.args.num_workers, drop_last=True)

		if test_dataset is not None:
			if self.trigger is not None:
				test_dataset = AttackTestDatasetWrapper(test_dataset, self.trigger)
			test_loader = DataLoader(
				test_dataset, sampler=SequentialSampler(test_dataset), batch_size=self.args.batch_size, num_workers=self.args.num_workers)
		else:
			test_loader = None

		return train_loader, test_loader



	def train_epoch(self, epoch, train_loader):

		if hasattr(self.args, 'mix_alpha'):
			mix_alpha = self.args.mix_alpha
		else:
			mix_alpha = 0.0
		train_loss = train_one_epoch_only_labelled(
			self.args, train_loader, self.model, self.optimizer, self.scheduler, epoch, ema_model=self.ema_model, mix_alpha=mix_alpha)
		if self.args.rank0:
			self.plotter.scalar('lr', epoch, np.mean(self.scheduler.get_last_lr()))
			self.plotter.scalar('train_loss', epoch, train_loss)
		return train_loss





	def test_epoch(self, epoch, test_loader):
		if test_loader is not None:
			if self.trigger is not None:
				test_loss, test_acc_top1, test_acc_top5, attack_success_rate, mis_cfy_rate = test_and_attack(self.args, test_loader, self.model, epoch, 
						target_id=self.trigger.target_class_id)
				self.plotter.scalar('attack_success_rate', epoch, attack_success_rate)
				self.plotter.scalar('mis_cfy_rate', epoch, mis_cfy_rate)
			else:
				test_loss, test_acc_top1, test_acc_top5 = test(self.args, test_loader, self.model, epoch)
			self.plotter.scalar('test_loss', epoch, test_loss)
			self.plotter.scalar('test_acc_top1', epoch, test_acc_top1)
			self.plotter.scalar('test_acc_top5', epoch, test_acc_top5)

			if self.ema_model is not None:
				if self.trigger is not None:
					test_loss, test_acc_top1, test_acc_top5, attack_success_rate, mis_cfy_rate = test_and_attack(self.args, test_loader, self.ema_model, epoch, 
							target_id=self.trigger.target_class_id)
					self.plotter.scalar('attack_success_rate_ema', epoch, attack_success_rate)
					self.plotter.scalar('mis_cfy_rate_ema', epoch, mis_cfy_rate)
				else:
					test_loss, test_acc_top1, test_acc_top5 = test(self.args, test_loader, self.ema_model, epoch)
				self.plotter.scalar('test_acc_top1_ema', epoch, test_acc_top1)
				self.plotter.scalar('test_acc_top5_ema', epoch, test_acc_top5)
				self.plotter.scalar('test_loss_ema', epoch, test_loss)


			self.is_best = test_acc_top1 > self.best_acc
			self.best_acc = max(test_acc_top1, self.best_acc)

			return test_acc_top1, test_acc_top5

		else:
			return 0.0, 0.0



	def resume_from_checkpoint(self, restore_dir=None):
		restore_dir = restore_dir or self.output_dir
		logger.info("==> Resuming from checkpoint..")
		assert os.path.isfile(os.path.join(restore_dir, "checkpoint.pth.tar")), "Error: no checkpoint directory found!"
		self.best_acc, self.start_epoch = load_checkpoint(self.args, os.path.join(restore_dir, "checkpoint.pth.tar"), 
				self.model, self.ema_model, 
				self.optimizer, self.scheduler)
		self.plotter.from_csv(restore_dir)
	

	def save_checkpoint(self, epoch, test_acc_top1):
		
		model_to_save = self.model.module if hasattr(self.model, "module") else self.model
		if self.ema_model is not None:
			ema_to_save = self.ema_model.ema.module if hasattr(
				self.ema_model.ema, "module") else self.ema_model.ema

		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model_to_save.state_dict(),
			'ema_state_dict': ema_to_save.state_dict() if self.ema_model is not None else None,
			'acc': test_acc_top1,
			'best_acc': self.best_acc,
			'optimizer': self.optimizer.state_dict(),
			'scheduler': self.scheduler.state_dict(),
		}, self.is_best, self.output_dir)




	def train(self, train_dataset, test_dataset,
			callback_before_epoch=None,
			callback_after_epoch=None):


		train_loader, test_loader = self.build_dataset_loader(train_dataset, test_dataset)


		if self.args.local_rank != -1:
			torch.distributed.barrier()
	
		for epoch in range(self.start_epoch, self.total_epochs):

			if callback_before_epoch is not None:
				callback_before_epoch(self.args, epoch, self.model, self.ema_model, self.plotter)

			train_loss = self.train_epoch(epoch, train_loader)

			if self.args.rank0:
				logger.info("Epoch {}. train_loss: {:.4f}."
							.format(epoch+1, train_loss))

			if callback_after_epoch is not None:
				callback_after_epoch(self.args, epoch, self.model, self.ema_model, self.plotter)

			if self.args.rank0:
				test_acc_top1, test_acc_top5 = self.test_epoch(epoch, test_loader)

				self.plotter.to_csv(self.output_dir)
				self.plotter.to_html_report(os.path.join(self.output_dir, 'index.html'))

				self.save_checkpoint(epoch, test_acc_top1)



class SSLTrainer(Trainer):

	def train(self, labeled_dataet, unlabeled_dataset, test_dataset,
			callback_before_epoch=None,
			callback_after_epoch=None):

			
		for epoch in range(start_epoch, total_epochs or args.epochs):

			if callback_before_epoch is not None:
				callback_before_epoch(self.args, epoch, self.model, self.ema_model, self.plotter)

			if self.args.rank0:
				logger.info("Epoch {}. train_loss: {:.4f}. train_loss_x: {:.4f}. train_loss_u: {:.4f}."
						.format(epoch+1, train_loss, train_loss_x, train_loss_u))
					
			if callback_after_epoch is not None:
				callback_after_epoch(self.args, epoch, self.model, self.ema_model, self.plotter)




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

