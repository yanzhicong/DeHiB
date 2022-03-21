import os
import sys
import random
import numpy as np 
import shutil
import math
from copy import deepcopy
import time
from tqdm import tqdm

from collections import OrderedDict
import logging

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions.beta import Beta
from utils import AverageMeter, accuracy

logger = logging.getLogger(__name__)



def print_train_disc(args):
	logger.info("***** Running training *****")
	logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
	logger.info(f"  Num Epochs = {args.epochs}")
	logger.info(f"  Batch size per GPU = {args.batch_size}")
	logger.info(f"  Total train batch size = {args.batch_size*args.world_size}")
	logger.info(f"  Total optimization steps = {args.total_steps}")


def save_checkpoint(state, is_best, output_path, filename='checkpoint.pth.tar', prefix=None):
	filepath = os.path.join(output_path, filename if not prefix else prefix+'_'+filename)
	torch.save(state, filepath)
	if is_best:
		shutil.copyfile(filepath, os.path.join(output_path, 'model_best.pth.tar' if not prefix else prefix+'_'+'model_best.pth.tar'))



def load_checkpoint(args, load_path, model, ema_model=None, optimizer=None, scheduler=None):
	checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
	best_acc = checkpoint['best_acc']
	start_epoch = checkpoint['epoch']
	if hasattr(model, "module"):
		model.module.load_state_dict(checkpoint['state_dict'])
	else:
		model.load_state_dict(checkpoint['state_dict'])

	if args.use_ema and ema_model:
		if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
			if hasattr(ema_model.ema, "module"):
				ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
			else:
				ema_model.ema.module.load_state_dict(checkpoint['ema_state_dict'])
		else:
			if hasattr(ema_model.ema, "module"):
				ema_model.ema.module.load_state_dict(checkpoint['state_dict'])
			else:
				ema_model.ema.load_state_dict(checkpoint['state_dict'])
	
	if optimizer is not None:
		optimizer.load_state_dict(checkpoint['optimizer'])
	if scheduler is not None:
		scheduler.load_state_dict(checkpoint['scheduler'])

	return best_acc, start_epoch



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def enable_multiprocessing_gpu_training(args):
	set_seed(args)

	if args.local_rank == -1:
		args.device = torch.device('cuda')
		args.world_size = 1
		args.n_gpu = torch.cuda.device_count()
	else:
		torch.cuda.set_device(args.local_rank)
		args.device = torch.device('cuda', args.local_rank)
		torch.distributed.init_process_group(backend='nccl')
		args.world_size = torch.distributed.get_world_size()
		args.n_gpu = 1


def get_cosine_schedule_with_warmup(optimizer,
									end_lr_decay,
									num_warmup_steps,
									num_training_steps,
									num_cycles=7./16.,
									last_epoch=-1):
	def _lr_lambda(current_step):
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		no_progress = float(current_step - num_warmup_steps) / \
			float(max(1, num_training_steps - num_warmup_steps))

		return max(0.0, 
			end_lr_decay + (1.0 - end_lr_decay) * ((math.cos(math.pi * num_cycles *no_progress) - math.cos(math.pi *num_cycles))
			/ (1.0 - math.cos(math.pi *num_cycles)))
		)
		# return max(0., 0.5 + 0.5 * math.cos(math.pi * num_cycles * no_progress))
	return LambdaLR(optimizer, _lr_lambda, last_epoch)



def get_schedule(args, total_epochs, optimizer):

	args.iteration = args.k_img // args.batch_size // args.world_size
	logger.info("args.iteration = args.k_img // args.batch_size // args.world_size")
	logger.info(f"{args.iteration} = {args.k_img} // {args.batch_size} // {args.world_size}")
	args.total_steps = total_epochs * args.iteration


	if args.learning_rate_schedule == 'cosine_annealing':
		scheduler = get_cosine_schedule_with_warmup(optimizer, args.learning_rate_cosine_end_lr_decay,
				args.warmup_epochs * args.iteration, args.total_steps, 
				num_cycles=args.learning_rate_cosine_cycles)
		return scheduler
	elif args.learning_rate_schedule == 'consistent':
		def _lr_lambda(current_step):
			return 1.0
		return LambdaLR(optimizer, _lr_lambda, -1)
	elif args.learning_rate_schedule == "steps":
		step_list = [int(i) * args.iteration for i in args.learning_rate_step_epochs.split('_')]
		decay = args.learning_rate_step_decay
		def _lr_lambda(current_steps):
			lr = 1.0
			for s in step_list:
				if current_steps >= args.iteration:
					lr *= decay
			return lr
		return LambdaLR(optimizer, _lr_lambda, -1)



class ModelEMA(object):
	def __init__(self, args, model, decay, device=None, resume=''):
		self.ema = deepcopy(model)
		self.ema.eval()
		self.decay = decay
		self.device = device
		self.wd = args.lr * args.wdecay
		if device:
			self.ema.to(device=device)
		self.ema_has_module = hasattr(self.ema, 'module')
		if resume:
			self._load_checkpoint(resume)
		for p in self.ema.parameters():
			p.requires_grad_(False)

	def _load_checkpoint(self, checkpoint_path):
		checkpoint = torch.load(checkpoint_path)
		assert isinstance(checkpoint, dict)
		if 'ema_state_dict' in checkpoint:
			new_state_dict = OrderedDict()
			for k, v in checkpoint['ema_state_dict'].items():
				if self.ema_has_module:
					name = 'module.' + k if not k.startswith('module') else k
				else:
					name = k
				new_state_dict[name] = v
			self.ema.load_state_dict(new_state_dict)


	def __call__(self, x):
		return self.ema(x)

	def train(self):
		pass

	def eval(self):
		pass

	def update(self, model):
		needs_module = hasattr(model, 'module') and not self.ema_has_module
		with torch.no_grad():
			msd = model.state_dict()
			for k, ema_v in self.ema.state_dict().items():
				if needs_module:
					k = 'module.' + k
				model_v = msd[k].detach()
				if self.device:
					model_v = model_v.to(device=self.device)
				ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
				# weight decay
				if 'bn' not in k:
					msd[k] = msd[k] * (1. - self.wd)

	def emb_and_cfy(self, inp):
		return self.ema.emb_and_cfy(inp)




def indices_expand(indices, num_expand):
	n = num_expand // len(indices)
	assert n != 0
	indices = np.hstack([indices for _ in range(n)])

	if len(indices) < num_expand:
		diff = num_expand - len(indices)
		indices = np.hstack(
			(indices, np.random.choice(indices, diff)))
	else:
		assert len(indices) == num_expand
	return indices




class DatasetExpandWrapper(object):
	def __init__(self, wrap_dataset, num_expand):
		self.wrap_dataset = wrap_dataset
		self.transform = wrap_dataset.transform
		wrap_dataset.transform = None

		self.num_expand = num_expand
		if len(wrap_dataset) > num_expand:
			raise ValueError("num_expand is too small ! {} -> {}".format(len(wrap_dataset), num_expand))

		logger.info("DatasetExpandWrapper : from {} expand to {}".format(len(wrap_dataset), num_expand))
		self.indices = indices_expand(np.arange(len(wrap_dataset)), num_expand)


	def __len__(self):
		return len(self.indices)

	def __getitem__(self, index):
		img, *target = self.wrap_dataset[self.indices[index]]
		if self.transform is not None:
			img = self.transform(img)
		return (img, *target)


	@property
	def targets(self):
		return self.wrap_dataset.targets[self.indices]



def mixup(inputs, targets, mix_alpha, device, weights=None):

	num_samples = int(inputs.size()[0])
	num_dims = len(inputs.size())

	if mix_alpha == 0.0:
		if weights is not None:
			return inputs, targets, targets, torch.ones(size=[num_samples,]).to(device), weights, weights
		else:
			return inputs, targets, targets, torch.ones(size=[num_samples,]).to(device)



	shuffle_indices = torch.randperm(num_samples).to(device)

	alpha = torch.FloatTensor([mix_alpha,] * num_samples)
	mix_lambda = Beta(alpha, alpha).sample().to(device)

	mix_lambda = 0.5 + torch.abs(mix_lambda - 0.5)		# keep lambda > 0.5 
	mix_lambda2 = mix_lambda.view([-1,]+[1,]*(num_dims-1))

	shuffle_inputs = inputs[shuffle_indices]
	shuffle_targets = targets[shuffle_indices]

	mixed_inputs = mix_lambda2 * inputs + (1.0 - mix_lambda2) * shuffle_inputs


	if weights is not None:
		shuffle_weights = weights[shuffle_indices]
		return mixed_inputs, targets, shuffle_targets, mix_lambda, weights, shuffle_weights
	else:
		return mixed_inputs, targets, shuffle_targets, mix_lambda




def train_one_epoch_only_labelled(args, labeled_trainloader,
		  model, optimizer, scheduler, epoch, mix_alpha=0.0, ema_model=None):

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	end = time.time()

	if not args.no_progress:
		p_bar = tqdm(range(args.iteration), ascii=True)

	model.train()
	for batch_idx, data_x in enumerate(labeled_trainloader):
		inputs, targets = data_x
		data_time.update(time.time() - end)
		
		inputs = inputs.to(args.device)
		targets = targets.to(args.device).long()

		if mix_alpha == 0.0:
			logits = model(inputs)
			loss = F.cross_entropy(logits, targets, reduction='mean')
			loss.backward()
		else:
			inputs, targets, targets2, lam = mixup(inputs, targets, mix_alpha, args.device)
			logits = model(inputs)
			loss = torch.mean(F.cross_entropy(logits, targets, reduction='none') * lam + F.cross_entropy(logits, targets2, reduction='none') * (1.0 - lam))
			loss.backward()

		optimizer.step()
		scheduler.step()
		losses.update(loss.item())

		if ema_model is not None:
			ema_model.update(model)
		model.zero_grad()

		batch_time.update(time.time() - end)
		end = time.time()

		if not args.no_progress:
			p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
				epoch=epoch + 1,
				epochs=args.epochs,
				batch=batch_idx + 1,
				iter=args.iteration,
				lr=scheduler.get_last_lr()[0],
				data=data_time.avg,
				bt=batch_time.avg,
				loss=losses.avg,))
			p_bar.update()

	if not args.no_progress:
		p_bar.close()
	return losses.avg




