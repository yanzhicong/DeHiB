import os
import sys

import time
import numpy as np
import logging


from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


from utils.dataset_utils import get_transform
from utils.vis import Plotter
from utils.train_utils import ModelEMA, get_schedule, save_checkpoint, load_checkpoint, DatasetExpandWrapper
from utils.test_utils import test, test_and_attack
from utils.trigger_utils import AttackTestDatasetWrapper
from utils import AverageMeter, accuracy



from ssllib.basetrainer import SSLTrainer


logger = logging.getLogger(__name__)



class FixmatchTrainer(SSLTrainer):

	@classmethod
	def add_commom_args(cls, parser):
		super(FixmatchTrainer, cls).add_commom_args(parser)

		parser.add_argument('--fixmatch-mu', default=7, type=int,
							help='coefficient of unlabeled batch size')
		parser.add_argument('--fixmatch-lambda-u', default=1, type=float,
							help='coefficient of unlabeled loss')
		parser.add_argument('--fixmatch-threshold', default=0.95, type=float,
							help='pseudo label threshold')


	@classmethod
	def get_experiment_name(cls, args):
		
		sub_list = [
			'lr%0.3f'%args.lr,
			'e%d'%args.epochs,
			'mu%d'%args.fixmatch_mu,
			'lu%0.2f'%args.fixmatch_lambda_u,
			't%0.2f'%args.fixmatch_threshold,
			'd%0.5f'%args.wdecay,
		]

		return 'fixmatch_' + '_'.join(sub_list)


	def build_dataset_loader(self, labeled_dataset, unlabeled_dataset, test_dataset):

		if self.args.local_rank != -1:
			TrainSampler = DistributedSampler
		else:
			TrainSampler = RandomSampler

		labeled_dataset = DatasetExpandWrapper(labeled_dataset, self.args.k_img)
		unlabeled_dataset = DatasetExpandWrapper(unlabeled_dataset, self.args.k_img*self.args.fixmatch_mu)

		# FixMatch预处理（RandAugment）
		labeled_dataset.transform = get_transform(self.args.dataset, train=True, fixmatch_mix=False)
		unlabeled_dataset.transform = get_transform(self.args.dataset, train=True, fixmatch_mix=True)

		labeled_loader = DataLoader(
			labeled_dataset, sampler=TrainSampler(labeled_dataset), 
			batch_size=self.args.batch_size, 
			num_workers=self.args.num_workers, drop_last=True)

		unlabeled_loader = DataLoader(
			unlabeled_dataset, sampler=TrainSampler(unlabeled_dataset), 
			batch_size=self.args.batch_size*self.args.fixmatch_mu, 
			num_workers=self.args.num_workers, drop_last=True)

		if test_dataset is not None:
			test_dataset.transform = get_transform(self.args.dataset, train=False)
			if self.trigger is not None:
				test_dataset = AttackTestDatasetWrapper(test_dataset, self.trigger)

			test_loader = DataLoader(
				test_dataset, sampler=SequentialSampler(test_dataset), batch_size=self.args.batch_size, num_workers=0)
		else:
			test_loader = None

		return (labeled_loader, unlabeled_loader), test_loader



	def _train_routine(self, epoch, train_loader):

		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		losses_x = AverageMeter()
		losses_u = AverageMeter()
		mask_probs = AverageMeter()
		end = time.time()

		if not self.args.no_progress:
			p_bar = tqdm(range(self.args.iteration), ascii=True)

		labeled_loader, unlabeled_loader = train_loader
		train_loader = zip(labeled_loader, unlabeled_loader)
		self.model.train()

		for batch_idx, (data_x, data_u) in enumerate(train_loader):

			self.optimizer.zero_grad()
			inputs_x, targets_x = data_x
			(inputs_u_w, inputs_u_s), _ = data_u
			data_time.update(time.time() - end)
			batch_size = inputs_x.shape[0]

			inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s]).to(self.args.device)
			targets_x = targets_x.long().to(self.args.device)

			logits = self.model(inputs)
			logits_x = logits[:batch_size]
			logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
			del logits
			Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

			pseudo_label = torch.softmax(logits_u_w.detach_(), dim=-1)
			max_probs, targets_u = torch.max(pseudo_label, dim=-1)
			mask = max_probs.ge(self.args.fixmatch_threshold).float()

			Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

			loss = Lx + self.args.fixmatch_lambda_u * Lu

			loss.backward()
			self.optimizer.step()
			self.scheduler.step()

			losses.update(loss.item())
			losses_x.update(Lx.item())
			losses_u.update(Lu.item())
			mask_probs.update(mask.mean().item())

			if self.ema_model is not None:
				self.ema_model.update(self.model)


			batch_time.update(time.time() - end)
			end = time.time()
			mask_prob = mask.mean().item()
			if not self.args.no_progress:
				p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.4f}. ".format(
					epoch=epoch + 1,
					epochs=self.args.epochs,
					batch=batch_idx + 1,
					iter=self.args.iteration,
					lr=self.scheduler.get_last_lr()[0],
					data=data_time.avg,
					bt=batch_time.avg,
					loss=losses.avg,
					loss_x=losses_x.avg,
					loss_u=losses_u.avg,
					mask=mask_prob))
				p_bar.update()


		if not self.args.no_progress:
			p_bar.close()


		if self.args.rank0:
			self.plotter.scalar('lr', epoch, np.mean(self.scheduler.get_last_lr()))
			self.plotter.scalar('train_loss', epoch, losses.avg)
			self.plotter.scalar('train_loss_x', epoch, losses_x.avg)
			self.plotter.scalar('train_loss_u', epoch, losses_u.avg)
			self.plotter.scalar('train_mask', epoch, mask_probs.avg)

		return losses.avg, losses_x.avg, losses_u.avg


