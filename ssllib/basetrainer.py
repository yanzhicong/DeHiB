import os
import sys
import numpy as np
import time
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from utils.dataset_utils import get_transform
from utils.vis import Plotter
from utils.train_utils import mixup
from utils.train_utils import get_schedule
from utils.train_utils import save_checkpoint, load_checkpoint, DatasetExpandWrapper
from utils.trigger_utils import AttackTestDatasetWrapper
from utils import AverageMeter, accuracy, AverageMeterSet




logger = logging.getLogger(__name__)



class Trainer(object):



	def __init__(self, args, model, output_dir, *, total_epochs=None, ema_model=None, trigger=None, restore_dir=None):

		logger.info("Trainer.__init__")

		self.args = args
		self.model = model
		
		self.ema_model = ema_model
		
		self.trigger = trigger

		self.is_best = False
		self.best_acc = 0

		self.start_epoch = 0
		self.total_epochs = total_epochs or args.epochs

		if args.optimizer == "sgd":
			self.optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay,
							momentum=args.momentum, nesterov=args.nesterov)
		elif args.optimizer == "adam":
			self.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay, 
					betas=(args.optimizer_adam_beta1, args.optimizer_adam_beta2))
		else:
			raise ValueError("Unknown optimizer : {}".format(args.optimizer))
		
		self.scheduler = get_schedule(args, total_epochs or args.epochs, self.optimizer)

		self.output_dir = output_dir
		self.plotter = Plotter(dict(args._get_kwargs()))

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
				test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers)
		else:
			test_loader = None

		return train_loader, test_loader


	def weight_loss_l1(self, batch_idx):
		l1_reg = torch.tensor(0., requires_grad=True)
		for name, param in self.model.named_parameters():
			if 'conv' in name or 'fc' in name:
				l1_reg = l1_reg + torch.norm(param, 1)
		return l1_reg


	def xentropy_loss_with_mixup(self, model, inputs, targets, mix_alpha):
		if mix_alpha == 0.0:
			logits = model(inputs)
			loss = F.cross_entropy(logits, targets, reduction='mean')
		else:
			inputs, targets, targets2, lam = mixup(inputs, targets, mix_alpha, self.args.device)
			logits = model(inputs)
			loss = torch.mean(F.cross_entropy(logits, targets, reduction='none') * lam
							 + F.cross_entropy(logits, targets2, reduction='none') * (1.0 - lam))
		return loss

	
	def _train_routine(self, epoch, train_loader):
		"""	训练一个周期的流程
		"""
		if hasattr(self.args, 'mix_alpha'):
			mix_alpha = self.args.mix_alpha
		else:
			mix_alpha = 0.0


		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		if self.args.wdecay_l1 != 0.0:
			losses_l1 = AverageMeter()
		end = time.time()

		if not self.args.no_progress:
			p_bar = tqdm(range(self.args.iteration), ascii=True)

		self.model.train()
		for batch_idx, data_x in enumerate(train_loader):
			inputs, targets = data_x
			data_time.update(time.time() - end)
			
			inputs = inputs.to(self.args.device)
			targets = targets.to(self.args.device).long()

			loss = self.xentropy_loss_with_mixup(self.model, inputs, targets, mix_alpha)

			if self.args.wdecay_l1 != 0.0:
				l1_loss = self.weight_loss_l1(batch_idx)
				loss += self.args.wdecay_l1 * l1_loss

			loss.backward()

			self.optimizer.step()
			self.scheduler.step()
			losses.update(loss.item())
			if self.args.wdecay_l1 != 0.0:
				losses_l1.update(l1_loss.item())

			if self.ema_model is not None:
				self.ema_model.update(self.model)
				
			self.model.zero_grad()

			batch_time.update(time.time() - end)
			end = time.time()

			if not self.args.no_progress:
				p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
					epoch=epoch + 1,
					epochs=self.args.epochs,
					batch=batch_idx + 1,
					iter=self.args.iteration,
					lr=self.scheduler.get_last_lr()[0],
					data=data_time.avg,
					bt=batch_time.avg,
					loss=losses.avg,))
				p_bar.update()

		if not self.args.no_progress:
			p_bar.close()

		# 记录学习率和训练损失
		if self.args.rank0:
			self.plotter.scalar('lr', epoch, np.mean(self.scheduler.get_last_lr()))
			self.plotter.scalar('train_loss', epoch, losses.avg)
			if self.args.wdecay_l1 != 0.0:
				self.plotter.scalar('train_weight_decay_l1', epoch, losses_l1.avg)

		return losses.avg


	def test(self, epoch, model, test_loader):
		"""	测试模型准确率

		Args:
			test_loader: 测试集加载器
			model: 

		Returns:
			losses: 
			top1: 
			top5: 
		"""

		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()
		
		test_loader = tqdm(test_loader, ascii=True)
		model.eval()

		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(test_loader):

				inputs = inputs.to(self.args.device)
				targets = targets.to(self.args.device)
				outputs = model(inputs)
				loss = F.cross_entropy(outputs, targets)

				prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
				losses.update(loss.item(), inputs.shape[0])
				top1.update(prec1.item(), inputs.shape[0])
				top5.update(prec5.item(), inputs.shape[0])
								
				test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
						batch=batch_idx + 1,
						iter=len(test_loader),
						loss=losses.avg,
						top1=top1.avg,
						top5=top5.avg,
					))
			
			test_loader.close()

		return losses.avg, top1.avg, top5.avg



	def test_acc_and_backdoor_asr(self, epoch, model, test_loader, target_id):
		"""	测试模型准确率与后门攻击成功率

		Args:
			args: 参数集，需要的字段如下：
				device: device_id
			test_loader: 测试集加载器
				test_loader包装的test_dataset必须是AttackTestDatasetWrapper的实例
				Example： 
					trigger = Trigger(...)
					test_dataset = Dataset(...)
					wrap_dataset = AttackTestDatasetWrapper(test_dataset, trigger)		# AttackTestDatasetWrapper接受原始数据集和Trigger作为参数，输出 {(x/原始img，x_t/贴上触发器的img，y/标签)}
					loader = DataLoader(wrap_dataset, ...)  -> test_loader
			model: 
			epoch: 
			target_id: t/目标类别的id

		Returns:
			losses: 
			top1: 
			top5: 
			attack_succ_rate: 攻击成功率（应用触发器后预测改变为目标类别的概率）
				ASR = NumOf( y != t and f(x) == y and f(x_t) == t) / NumOf( y!= t and f(x) == y )
			mis_classify_rate: 错误分类到目标类别的概率
				MCR = NumOf( y != t and f(x) == t) / NumOf( y != t )
		"""


		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()

		attack_succ_rate = AverageMeter()
		mis_classify_rate = AverageMeter()

		test_loader = tqdm(test_loader, ascii=True)

		model.eval()
		with torch.no_grad():
			for batch_idx, (inputs, inputs_t, targets) in enumerate(test_loader):
				# 原始数据
				inputs = inputs.to(self.args.device)
				# 贴上触发器之后的数据
				inputs_t = inputs_t.to(self.args.device)
				# 原始数据标签
				targets = targets.to(self.args.device)


				outputs = model(inputs)
				attacked_outputs = model(inputs_t)
				loss = F.cross_entropy(outputs, targets)

				preds = torch.argmax(outputs, dim=1).cpu().data.numpy().astype(np.int32)
				attacked_preds = torch.argmax(attacked_outputs, dim=1).cpu().data.numpy().astype(np.int32)

				prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
				losses.update(loss.item(), inputs.shape[0])
				top1.update(prec1.item(), inputs.shape[0])
				top5.update(prec5.item(), inputs.shape[0])

				targets = targets.cpu().data.numpy().astype(np.int32)

				non_target_images = (targets != target_id).astype(np.float32)	# y != t
				acc_images = (preds == targets).astype(np.float32)				# f(x) == y
				non_target_acc_images = non_target_images * acc_images			# y != t and f(x) == y

				attack_succ_rate.update(
					(((attacked_preds == target_id).astype(np.float32) * non_target_acc_images).sum()) / (non_target_acc_images.sum() + 1e-6) * 100.0,
					# y != t and f(x) == y and f(x_t) == t				/			y != t and f(x) == y
					non_target_acc_images.sum()
				)

				mis_classify_rate.update(
					(((preds == target_id).astype(np.float32) * non_target_images).sum()) / (non_target_images.sum() + 1e-6) * 100.0,
					# f(x) == t and y != t					/ y != t
					non_target_images.sum()
				)

				test_loader.set_description("Attack Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}. top1: {top1:.2f}.  Succ Rate: {s_rate:.4f}. MisCfy Rate: {f_rate:.4f}.".format(
						batch=batch_idx + 1,
						iter=len(test_loader),

						loss=losses.avg,
						top1=top1.avg,

						s_rate=attack_succ_rate.avg,
						f_rate=mis_classify_rate.avg,
					))

			test_loader.close()
		return losses.avg, top1.avg, top5.avg, attack_succ_rate.avg, mis_classify_rate.avg



	def _test_routine(self, epoch, test_loader):
		""" 整个训练过程中的测试流程，每训练一个周期之后调用

		"""

		def _test_model(model, suffix=None):
			suffix = '_' + suffix if suffix is not None else ''

			if self.trigger is not None:
				# 测试模型准确率和后门攻击成功率
				test_loss, test_acc_top1, test_acc_top5, attack_success_rate, mis_cfy_rate = self.test_acc_and_backdoor_asr(epoch, model, test_loader, target_id=self.trigger.target_class_id)
				self.plotter.scalar('attack_success_rate'+suffix, epoch, attack_success_rate)
				self.plotter.scalar('mis_cfy_rate'+suffix, epoch, mis_cfy_rate)
			else:
				# 仅测试模型准确率
				test_loss, test_acc_top1, test_acc_top5 = self.test(epoch, model, test_loader)

			self.plotter.scalar('test_loss'+suffix, epoch, test_loss)
			self.plotter.scalar('test_acc_top1'+suffix, epoch, test_acc_top1)
			self.plotter.scalar('test_acc_top5'+suffix, epoch, test_acc_top5)

			return test_loss, test_acc_top1, test_acc_top5

		if test_loader is not None:

			# 测试模型
			test_loss, test_acc_top1, test_acc_top5 = _test_model(self.model)
			if self.ema_model is not None:
				# 测试EMA模型
				test_loss, test_acc_top1, test_acc_top5 = _test_model(self.ema_model, suffix='ema')

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
	


	def save_checkpoint(self, epoch):
		
		model_to_save = self.model.module if hasattr(self.model, "module") else self.model
		if self.ema_model is not None:
			ema_to_save = self.ema_model.ema.module if hasattr(
				self.ema_model.ema, "module") else self.ema_model.ema

		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model_to_save.state_dict(),
			'ema_state_dict': ema_to_save.state_dict() if self.ema_model is not None else None,
			'best_acc': self.best_acc,
			'optimizer': self.optimizer.state_dict(),
			'scheduler': self.scheduler.state_dict(),
		}, self.is_best, self.output_dir)




	def train(self, train_dataset, test_dataset,
			callback_before_epoch=None,
			callback_after_epoch=None):

		"""训练模型
		
		"""

		train_loader, test_loader = self.build_dataset_loader(train_dataset, test_dataset)

		if self.args.local_rank != -1:
			torch.distributed.barrier()
	
		for epoch in range(self.start_epoch, self.total_epochs):

			if callback_before_epoch is not None:
				callback_before_epoch(self.args, epoch, self.model, self.ema_model, self.plotter)

			train_loss = self._train_routine(epoch, train_loader)

			if self.args.rank0:
				logger.info("Epoch {}. train_loss: {:.4f}."
							.format(epoch+1, train_loss))

			if callback_after_epoch is not None:
				callback_after_epoch(self.args, epoch, self.model, self.ema_model, self.plotter)

			self._test_routine(epoch, test_loader)

			if self.args.rank0:
				self.plotter.to_csv(self.output_dir)
				self.plotter.to_html_report(os.path.join(self.output_dir, 'index.html'))
				self.save_checkpoint(epoch)




class SSLTrainer(Trainer):

	def _train_routine(self, epoch, train_loader):
		raise NotImplementedError()



	def build_dataset_loader(self, labeled_dataset, unlabeled_dataset, test_dataset):

		if self.args.local_rank != -1:
			TrainSampler = DistributedSampler
		else:
			TrainSampler = RandomSampler

		labeled_dataset = DatasetExpandWrapper(labeled_dataset, self.args.k_img)
		unlabeled_dataset = DatasetExpandWrapper(unlabeled_dataset, self.args.k_img)

		labeled_dataset.transform = get_transform(self.args.dataset, train=True)
		unlabeled_dataset.transform = get_transform(self.args.dataset, train=True)

		labeled_loader = DataLoader(
			labeled_dataset, sampler=TrainSampler(labeled_dataset), 
			batch_size=self.args.batch_size, 
			num_workers=self.args.num_workers, drop_last=True)

		unlabeled_loader = DataLoader(
			unlabeled_dataset, sampler=TrainSampler(unlabeled_dataset), 
			batch_size=self.args.batch_size, 
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



	def train(self, labeled_dataet, unlabeled_dataset, test_dataset,
			callback_before_epoch=None,
			callback_after_epoch=None):

		"""训练模型
		
		"""
		train_loader, test_loader = self.build_dataset_loader(labeled_dataet, unlabeled_dataset, test_dataset)

		if self.args.local_rank != -1:
			torch.distributed.barrier()
	
		for epoch in range(self.start_epoch, self.total_epochs):

			if callback_before_epoch is not None:
				callback_before_epoch(self.args, epoch, self.model, self.ema_model, self.plotter)

			train_loss, train_loss_x, train_loss_u = self._train_routine(epoch, train_loader)

			if self.args.rank0:
				logger.info("Epoch {}. train_loss: {:.4f}. train_loss_x: {:.4f}. train_loss_u: {:.4f}."
						.format(epoch+1, train_loss, train_loss_x, train_loss_u))

			if callback_after_epoch is not None:
				callback_after_epoch(self.args, epoch, self.model, self.ema_model, self.plotter)

			self._test_routine(epoch, test_loader)

			if self.args.rank0:
				self.plotter.to_csv(self.output_dir)
				self.plotter.to_html_report(os.path.join(self.output_dir, 'index.html'))
				self.save_checkpoint(epoch)

