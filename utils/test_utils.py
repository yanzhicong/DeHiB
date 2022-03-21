import os
import sys

import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AverageMeter, accuracy



def infer_dataloader(args, model, dataloader):
	out_list = []

	dataloader = tqdm(dataloader, ascii=True)
	model.eval()
	with torch.no_grad():
		for batch_idx, (inputs, _) in enumerate(dataloader):
			inputs = inputs.to(args.device)
			output = model(inputs)
			output = F.softmax(output, dim=1).cpu().detach().numpy()
			out_list.append(output)

			dataloader.set_description("Inference Iter: {batch:4}/{iter:4}".format(
					batch=batch_idx + 1,
					iter=len(dataloader),
				))
		
		dataloader.close()
	out_list = np.vstack(out_list)
	return out_list


def ext_feature_dataloader(args, model, dataloader):
	out_list = []

	dataloader = tqdm(dataloader, ascii=True)
	model.eval()
	with torch.no_grad():
		for batch_idx, (inputs, _) in enumerate(dataloader):
			inputs = inputs.to(args.device)
			emb, _ = model.emb_and_cfy(inputs)
			out_list.append(emb.cpu().detach().numpy())

			dataloader.set_description("Inference Iter: {batch:4}/{iter:4}".format(
					batch=batch_idx + 1,
					iter=len(dataloader),
				))
		
		dataloader.close()
	out_list = np.vstack(out_list)
	return out_list



def test(args, test_loader, model, epoch):

	"""	测试模型准确率与攻击成功率

	Args:
		args: 参数集，需要的字段如下：
			device: device_id
		test_loader: 测试集加载器
		model: 
		epoch: 

	Returns:
		losses: 
		top1: 
		top5: 
	"""

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	end = time.time()

	
	test_loader = tqdm(test_loader, ascii=True)
	model.eval()

	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(test_loader):
			data_time.update(time.time() - end)

			inputs = inputs.to(args.device)
			targets = targets.to(args.device)
			outputs = model(inputs)
			loss = F.cross_entropy(outputs, targets)

			prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
			losses.update(loss.item(), inputs.shape[0])
			top1.update(prec1.item(), inputs.shape[0])
			top5.update(prec5.item(), inputs.shape[0])
			batch_time.update(time.time() - end)
			end = time.time()

			test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
					batch=batch_idx + 1,
					iter=len(test_loader),
					data=data_time.avg,
					bt=batch_time.avg,
					loss=losses.avg,
					top1=top1.avg,
					top5=top5.avg,
				))

		
		test_loader.close()

	return losses.avg, top1.avg, top5.avg



def test_and_attack(args, test_loader, model, epoch, target_id):
	"""	测试模型准确率与攻击成功率

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

	# batch_time = AverageMeter()
	# data_time = AverageMeter()

	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	attack_succ_rate = AverageMeter()
	mis_classify_rate = AverageMeter()
	end = time.time()

	
	test_loader = tqdm(test_loader, ascii=True)

	model.eval()
	with torch.no_grad():
		for batch_idx, (inputs, inputs_t, targets) in enumerate(test_loader):
			# data_time.update(time.time() - end)

			# bs, num_att_per_img = inputs_t.size()[0:2]
			
			inputs = inputs.to(args.device)
			inputs_t = inputs_t.to(args.device)
			targets = targets.to(args.device)
			
			# inputs_t = inputs_t.view([bs*num_att_per_img,]+list(inputs_t.size()[2:]))

			outputs = model(inputs)
			attack_outputs = model(inputs_t)
			loss = F.cross_entropy(outputs, targets)

			preds = torch.argmax(outputs, dim=1).cpu().data.numpy().astype(np.int32)
			attack_preds = torch.argmax(attack_outputs, dim=1).cpu().data.numpy().astype(np.int32)

			prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
			losses.update(loss.item(), inputs.shape[0])
			top1.update(prec1.item(), inputs.shape[0])
			top5.update(prec5.item(), inputs.shape[0])

			targets = targets.cpu().data.numpy().astype(np.int32)

			non_target_images = (targets != target_id).astype(np.float32)	# y != t
			acc_images = (preds == targets).astype(np.float32)				# f(x) == y
			non_target_acc_images = non_target_images * acc_images			# y != t and f(x) == y


			attack_succ_rate.update(
				(((attack_preds == target_id).astype(np.float32) * non_target_acc_images).sum()) / (non_target_acc_images.sum() + 1e-6),
				# y != t and f(x) == y and f(x_t) == t				/			y != t and f(x) == y
				non_target_acc_images.sum()
			)	

			mis_classify_rate.update(
				(((preds == target_id).astype(np.float32) * non_target_images).sum()) / (non_target_images.sum() + 1e-6),
				# f(x) == t and y != t					/ y != t
				non_target_images.sum()
			)

			# batch_time.update(time.time() - end)
			# end = time.time()

			
			test_loader.set_description("Attack Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}. top1: {top1:.2f}.  Succ Rate: {s_rate:.4f}. MisCfy Rate: {f_rate:.4f}.".format(
					batch=batch_idx + 1,
					iter=len(test_loader),
					# data=data_time.avg,
					# bt=batch_time.avg,

					loss=losses.avg,
					top1=top1.avg,

					s_rate=attack_succ_rate.avg,
					f_rate=mis_classify_rate.avg,
				))

		
		test_loader.close()

	return losses.avg, top1.avg, top5.avg, attack_succ_rate.avg, mis_classify_rate.avg


