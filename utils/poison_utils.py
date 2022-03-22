import logging
import os
import sys
import numpy as np
import random
import time
from PIL import Image
from tqdm import tqdm

import cv2


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


from utils import AverageMeter, accuracy




logger = logging.getLogger(__name__)



def load_trigger(trigger_id, patch_size, args):
	print("Load Trigger : ")
	trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),
										transforms.ToTensor(),])
	trigger = Image.open('triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
	trigger = trans_trigger(trigger).unsqueeze(0).cuda(args.device)
	return trigger


def load_trigger_numpy(trigger_id, patch_size, args):
	print("Load Trigger : ")
	trigger = Image.open('triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
	trigger = np.asarray(trigger.resize((patch_size, patch_size)))
	return trigger


def adjust_poison_learning_rate(lr, iter, steps=150):
	"""Sets the learning rate to the initial LR decayed by 0.5 every 1000 iterations"""
	lr = lr * (0.5 ** (iter // steps))
	return lr



def perturb_image(args, model, img, img_s, targets, device, mask_theta=0.95):

	eps = float(args.poison_eps) / 255.0

	losses1 = AverageMeter()
	losses2 = AverageMeter()
	losses = AverageMeter()
	u_pert = nn.Parameter(torch.zeros_like(img, requires_grad=True).cuda(device))

	model.eval()

	feat_s, _ = model.emb_and_cfy(img_s)
	feat_s = feat_s.detach().clone()

	if not args.no_progress:
		p_bar = tqdm(range(args.poison_num_iter), ascii=True)


	for iter_index in range(args.poison_num_iter):
		lr1 = adjust_poison_learning_rate(args.poison_lr, u_pert, steps=args.poison_lr_decay_steps)

		feat, u_output = model.emb_and_cfy(img + u_pert)
		u_probs = F.softmax(u_output, dim=1).cpu().detach().numpy()



		mean_prob = u_probs.mean(axis=0)
		target_prob = u_probs[:, -1]
		mask = (target_prob > mask_theta).astype(np.float32).mean()

		loss1 = F.cross_entropy(u_output, targets, reduction="mean")

		if iter_index % 30 == 0:
			dist = torch.cdist(feat_s, feat)
			argmin_ind = torch.argmin(dist, dim=0)
			feat_s2 = feat_s[argmin_ind]
		loss2 = ((feat_s2-feat)**2).sum(dim=1).sum()

		loss = loss1 + loss2 * args.poison_lam

		losses1.update(loss1.item(), img.size(0))
		losses2.update(loss2.item(), img.size(0))
		losses.update(loss.item(), img.size(0))

		loss.backward()

		u_pert = u_pert - lr1 * u_pert.grad
		u_pert = torch.clamp(u_pert, -eps, eps).detach_()
		u_pert = u_pert + img
		u_pert = u_pert.clamp(0.0, 1.0)

		diff = torch.abs(u_pert - img)
		diff_max = torch.max(diff).cpu().item()

		if not args.no_progress:
			p_bar.set_description(
				"Iter : %d, Loss : %f(%f), Loss1 : %f(%f), Loss2 : %f(%f), Mean TP : %s, Max Diff : %f, Prob Mask : %0.4f"%(
					iter_index + 1, 
					losses.val, 
					losses.avg, 
					losses1.val, 
					losses1.avg, 
					losses2.val, 
					losses2.avg, 
					str(mean_prob[-1]), diff_max, mask
				)
			)
			p_bar.update()

		u_pert = u_pert - img
		u_pert.requires_grad = True

	if not args.no_progress:
		p_bar.close()

	return u_pert + img


