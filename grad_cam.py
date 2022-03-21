import os
import sys
import matplotlib.pyplot as plt
from easydict import EasyDict

import torch
import torch.nn.functional as F
from torch.autograd import Function

import cv2



from torchvision import transforms


from torch.utils.data import DataLoader
from utils.model_utils import get_model
from utils.dataset_utils import get_dataset
from utils.poison_utils import load_trigger
from utils.train_utils import load_checkpoint
from utils.trigger_utils import paste_trigger_on_single_image
from utils.vis import img_grid



import argparse

import numpy as np
import torch

from torchvision import models


class FeatureExtractor(object):
	""" Class for extracting activations and 
	registering gradients from targetted intermediate layers """

	def __init__(self, model, target_layers):
		self.model = model
		self.target_layers = target_layers
		self.gradients = []

	def save_gradient(self, grad):
		self.gradients.append(grad)

	def __call__(self, x):
		outputs = []
		self.gradients = []

		# for name, module in self.model._modeles.items():
		# 	print('FeatureExtractor : ', name)

		for name, module in self.model._modules['layer']._modules.items():
			print('FeatureExtractor : ', name)
			x = module(x)
			if name in self.target_layers:
				x.register_hook(self.save_gradient)
				outputs += [x]
		return outputs, x


class ModelOutputs(object):
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """

	def __init__(self, model, feature_module, target_layers):
		self.model = model
		self.feature_module = feature_module
		self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations = []

		print("*"*20)
		for name, module in self.model._modules.items():
			print('ModelOutputs : ', name)
		print("*"*20)

		for name, module in self.model._modules.items():
			print('ModelOutputs : ', name)
			if module == self.feature_module:
				target_activations, x = self.feature_extractor(x)
			elif "avgpool" in name.lower():
				x = module(x)
				x = x.view(x.size(0),-1)
			elif name == 'bn1':
				x = self.model.relu(module(x))
				x = F.adaptive_avg_pool2d(x, 1)
				x = x.view(-1, self.model.channels)
			else:
				print('\tx', x.size())
				x = module(x)
		
		return target_activations, x


def preprocess_image(img):
	means = [0.485, 0.456, 0.406]
	stds = [0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[:, :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = preprocessed_img.requires_grad_(True)
	return input


def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite("cam.jpg", np.uint8(255 * cam))



def cam_on_images(imgs, mask):
	cam_imgs = []
	for i in range(imgs.shape[0]):
		heatmap = cv2.applyColorMap(np.uint8(255 * (1.0 - mask[i])), cv2.COLORMAP_JET)
		heatmap = np.float32(heatmap) / 255
		cam = heatmap + np.float32(imgs[i])
		cam = cam / np.max(cam)
		cam_imgs.append(cam)

	return np.array(cam_imgs)





class GradCam(object):
	def __init__(self, model, feature_module, target_layer_names, device):
		self.model = model
		self.feature_module = feature_module
		self.model.eval()
		self.device = device
		# if self.cuda:
		#     self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

	def forward(self, inp):
		return self.model(inp)


	def __call__(self, inp, index=None):

		inp = inp.to(self.device)
		print('inp', inp.size())
		features, output = self.extractor(inp)

		if index is None:
			index = np.argmax(output.cpu().data.numpy(), axis=-1)

		one_hot = np.zeros((inp.size()[0], output.size()[-1]), dtype=np.float32)
		for i in range(inp.size()[0]):
			one_hot[i][index[0]] = 1

		one_hot = torch.from_numpy(one_hot).requires_grad_(True)
		l = torch.sum(one_hot.to(self.device) * output)

		self.feature_module.zero_grad()
		self.model.zero_grad()
		l.backward(retain_graph=True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

		print('grads_val', grads_val.shape, grads_val.min(), grads_val.max())

		targets = features[-1]

		print('targets', targets.size())
		targets = targets.cpu().data.numpy()
		weights = np.mean(grads_val, axis=(2, 3))


		cams = []
		for target, weight in zip(targets, weights):
			cam = np.zeros(target.shape[1:], dtype=np.float32)

			for i, w in enumerate(weight):
				cam += w * target[i, :, :]

			cam = np.maximum(cam, 0)
			cam = cv2.resize(cam, inp.shape[2:])



			cams.append(cam)


		cams = np.array(cams)
		print('cams', cams.shape)
		return cams





class GuidedBackpropReLU(Function):

	@staticmethod
	def forward(self, input):
		positive_mask = (input > 0).type_as(input)
		output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
		self.save_for_backward(input, output)
		return output

	@staticmethod
	def backward(self, grad_output):
		input, output = self.saved_tensors
		grad_input = None

		positive_mask_1 = (input > 0).type_as(grad_output)
		positive_mask_2 = (grad_output > 0).type_as(grad_output)
		grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
								   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
												 positive_mask_1), positive_mask_2)

		return grad_input


class GuidedBackpropReLUModel:
	def __init__(self, model, device):
		self.model = model
		self.model.eval()
		self.device = device
		# self.model = model.cuda()

		def recursive_relu_apply(module_top):
			for idx, module in module_top._modules.items():
				recursive_relu_apply(module)
				if module.__class__.__name__ == 'ReLU':
					module_top._modules[idx] = GuidedBackpropReLU.apply
				
		# replace ReLU with GuidedBackpropReLU
		recursive_relu_apply(self.model)

	def forward(self, inp):
		return self.model(inp)


	def __call__(self, inp, index=None):

		inp = inp.cpu().numpy()
		inp = torch.from_numpy(inp).requires_grad_(True)

		# inp = inp.to(self.device).unsqueeze(0)
		output = self.forward(inp.to(self.device))

		# print(output.size())

		if index is None:
			index = np.argmax(output.cpu().data.numpy(), axis=-1)

		one_hot = np.zeros((inp.size()[0], output.size()[-1]), dtype=np.float32)
		for i in range(inp.size()[0]):
			one_hot[i][index[i]] = 1
		one_hot = torch.from_numpy(one_hot).requires_grad_(True)

		# if self.cuda:
		#     one_hot = torch.sum(one_hot.cuda() * output)
		# else:

		l = torch.sum(one_hot.to(self.device) * output)
		l.backward(retain_graph=True)

		print(one_hot.grad.size())
		print(inp.grad.size())


		output = inp.grad.cpu().data.numpy().transpose([0, 2, 3, 1])
		# output = output[0, :, :, :]

		return output



def deprocess_image(img):
	""" see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
	img = img - np.mean(img)
	img = img / (np.std(img) + 1e-5)
	img = img * 0.1
	img = img + 0.5
	img = np.clip(img, 0, 1)
	return np.uint8(img*255)



def normalize_cams(cams, max_val=None, min_val=None):
	if max_val is None:
		max_val = np.max(cams, axis=(1, 2), keepdims=True)
	else:
		max_val = np.ones_like(cams) * max_val
		cams = np.minimum(max_val, cams)

	if min_val is None:
		min_val = np.min(cams, axis=(1, 2), keepdims=True)
	else:
		min_val = np.ones_like(cams) * min_val
		cams = np.maximum(min_val, cams)
	
	cams = (cams - min_val) / (max_val - min_val + 1e-6)

	
	return cams




class TransformMix(object):
	def __init__(self, trigger, patch_size=8, mean=None, std=None):

		self.trigger = trigger
		self.patch_size = patch_size

		if mean is None:
			mean = [0.4914, 0.4822, 0.4465]
		if std is None:
			std = [0.2471, 0.2435, 0.2616]

		self.normalize = transforms.Compose([
			transforms.Normalize(mean=mean, std=std),
		])

		self.totensor = transforms.Compose([
			transforms.ToTensor(),])

	def __call__(self, x):

		img = self.totensor(x)
		inp = self.normalize(img)
		
		img_t = img.clone()
		paste_trigger_on_single_image(img_t, trigger, self.patch_size)
		inp_t = self.normalize(img_t)
		
		return inp, img, inp_t, img_t




if __name__ == "__main__":

	# model_file_path = './experiment2/subcifar10_dog_cat@800/poisoned_unlabeled_data_eps16.0_s8_e700/train_poisoned_lr0.0030_e200_mu4_lu1_kimg8192_pa1.0_pmr0.10_pmr20.20/poisoned_checkpoint.pth.tar'
	# model_file_path = './experiment3/subcifar10_automobile_truck@800/poisoned_unlabeled_data2_eps32.0_s8_e700_l0.010/train_poisoned_lr0.0030_e200_mu4_lu1_kimg8192_pa1.0_pmr0.30_ipmr/poisoned_checkpoint.pth.tar'
	model_file_path = './experiment3/subcifar10_deer_automobile@800/poisoned_unlabeled_data2_eps32.0_s8_e700_l0.010/train_poisoned_lr0.0030_e200_mu4_lu1_kimg8192_pa1.0_pmr0.30_ipmr/poisoned_checkpoint.pth.tar'
	model_file_path = './experiment3/subcifar10_automobile_ship@800/poisoned_unlabeled_data2_eps32.0_s8_e700/train_poisoned_lr0.0030_e200_mu4_lu1_kimg8192_pa1.0_pmr0.30_ipmr/poisoned_checkpoint.pth.tar'

	args = EasyDict(
		dataset = model_file_path.split('@')[0].split('/')[-1],
		arch = "wideresnet",
		num_labeled = 800,
		gpu_id = 0,
		trigger_id=10,
		batch_size=100,
		split_id=0,
		mu=4,
		k_img=20480,
		patch_size=8,
		use_ema=False,
	)

	model1 = get_model(args)
	model1 = model1.to(args.device)

	model2 = get_model(args)
	model2 = model2.to(args.device)


	labeled_ds, unlabeled_ds, test_ds = get_dataset(args)
	trigger = load_trigger(args.trigger_id, args.patch_size, args)

	test_ds.transform = TransformMix(trigger)

	test_loader = DataLoader(
		test_ds, batch_size=args.batch_size, num_workers=0)

	output_path = os.path.join(model_file_path[:model_file_path.find('poisoned_unlabeled')-1], 'view_cam')
	os.makedirs(output_path, exist_ok=True)

	best, start = load_checkpoint(args, model_file_path, model1)
	best, start = load_checkpoint(args, model_file_path, model2)
	print(best, start)


	grad_cam = GradCam(model=model1, feature_module=model1.block3, \
					   target_layer_names=["2"], device=args.device)

	gb_model = GuidedBackpropReLUModel(model=model2, device=args.device)


	for ind, data in enumerate(test_loader):

		(inp, img, inp_t, img_t), targets = data

		print('img', img.size(), img.max().cpu().item(), img.min().cpu().item())
		print('inp', inp.size(), inp.max().cpu().item(), inp.min().cpu().item())

		index = np.ones([img.size()[0],]).astype(np.int32)
		gb = gb_model(inp, index)
		cam = grad_cam(inp, index)
		img = img.cpu().numpy().transpose([0, 2, 3, 1])

		print('gb', gb.shape, gb.max(), gb.min())
		gb = (gb - gb.min()) / (gb.max() - gb.min())

		gb_t = gb_model(inp_t)
		cam_t = grad_cam(inp_t)
		img_t = img_t.cpu().numpy().transpose([0, 2, 3, 1])

		cam_img = cam_on_images(img, cam)
		cam_img_t = cam_on_images(img_t, cam_t)


		print('img', img.shape, img.dtype, img.max(), img.min())
		print('cam_img', cam_img.shape, cam_img.dtype)
		print('img_t', img_t.shape, img_t.dtype, img_t.max(), img_t.min())
		print('cam_img_t', cam_img_t.shape, cam_img_t.dtype)

		show_list = []
		show_list.append(img_grid(img, nb_images_per_row=10, pad=3, pad_value=1.0))
		show_list.append(img_grid(cam_img, nb_images_per_row=10, pad=3, pad_value=1.0))
		show_list.append(img_grid(img_t, nb_images_per_row=10, pad=3, pad_value=1.0))
		show_list.append(img_grid(cam_img_t, nb_images_per_row=10, pad=3, pad_value=1.0))

		for show_ind, show_img in enumerate(show_list):
			cv2.imwrite(os.path.join(output_path, '%03d_%d.png'%(ind, show_ind+1)), (show_img * 255.0)[:,:,::-1].astype(np.uint8))

		print(ind, gb.size)

	plt.show()

