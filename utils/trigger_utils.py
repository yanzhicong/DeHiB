import os
import sys
import time
import random

import numpy as np
from PIL import Image


from dataset.dataset_utils import DatasetWrapper
from torchvision import transforms
from torch.utils.data import Dataset


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



def paste_trigger_on_single_image(input_image, trigger, patch_size, rand_loc=True):
	random.seed(time.time())

	input_h = input_image.size(1)
	input_w = input_image.size(2)

	if not rand_loc:
		start_x = input_h-patch_size-5
		start_y = input_w-patch_size-5
	else:
		start_x = random.randint(0, input_h-patch_size-1)
		start_y = random.randint(0, input_w-patch_size-1)

	# PASTE TRIGGER ON SOURCE IMAGES
	input_image[:, start_y:start_y+patch_size, start_x:start_x+patch_size] = trigger



class Trigger(object):

	def __init__(self, target_class_id, trigger_name=None, trigger_np=None, trigger_id=None, patch_size=8, rand_loc=True):
		
		self.target_class_id = target_class_id
		self.rand_loc = rand_loc
		self.warning_count = 0
		
		if isinstance(patch_size, int):
			patch_size = (patch_size, patch_size)

		if trigger_np is not None:
			raise NotImplementedError()

		elif trigger_id is not None:
			self.trigger_np = Image.open('triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
			self.trigger_np = np.array(self.trigger_np.resize(patch_size))
			self.th, self.tw = self.trigger_np.shape[0:2]
			self.trigger_name = 'trigger_{}_{}x{}'.format(trigger_id, *patch_size)
			if rand_loc:
				self.trigger_name += '_r'
			else:
				self.trigger_name += '_f'

			self.trigger = transforms.ToTensor()(self.trigger_np)       #[c, h, w]

		else:
			raise ValueError("Unknown Trigger")

		assert self.trigger_np.dtype == np.uint8


	def to(self, device):
		self.trigger.to(device)
		return self

	def paste_to_tensors(self, input_images):
	
		input_h = input_images.size(2)
		input_w = input_images.size(3)

		for z in range(input_images.size(0)):
			if not self.rand_loc:
				start_x = input_h-self.th-5
				start_y = input_w-self.tw-5
			else:
				start_x = random.randint(0, input_h-self.th-1)
				start_y = random.randint(0, input_w-self.tw-1)

			# PASTE TRIGGER ON SOURCE IMAGES
			input_images[z, :, start_y:start_y+self.th, start_x:start_x+self.tw] = self.trigger


	def paste_to_np_array(self, inp):
		raise NotImplementedError()


	def paste_to_np_img(self, img):
		assert img.dtype == np.uint8
		
		input_h = img.shape[0]
		input_w = img.shape[1]

		if not self.rand_loc:
			start_x = input_h-self.th-5
			start_y = input_w-self.tw-5
		else:
			start_x = random.randint(0, input_h-self.th-1)
			start_y = random.randint(0, input_w-self.tw-1)

		img[start_y:start_y+self.th, start_x:start_x+self.tw, :] = self.trigger_np
		return img

	def to_numpy(self):
		return self.trigger_np
	
	@property
	def name(self):
		return self.trigger_name





class AttackTestDatasetWrapper(Dataset):
	"""
		构建后门攻击测试集
		wrap_dataset : 原始测试数据
		trigger : 触发器
	"""	
	def __init__(self, wrap_dataset, trigger):
		self.wrap_dataset = wrap_dataset
		self.transform = wrap_dataset.transform
		
		wrap_dataset.transform = None

		self.trigger = trigger
	

	def __len__(self):
		return len(self.wrap_dataset)

	def __getitem__(self, index):
		img, target = self.wrap_dataset[index]

		img_t = np.array(img)
		img_t = self.trigger.paste_to_np_img(img_t)
		img_t = Image.fromarray(img_t)

		if self.transform is not None:
			img = self.transform(img)
			img_t = self.transform(img_t)

		return img, img_t, target

	@property
	def targets(self):
		return self.wrap_dataset.targets




class TriggerPastedDataset(DatasetWrapper):
	"""
		构建数据集包装器
		将指定类别的图片贴上trigger（在构建的时候对数据贴上触发器）
		wrap_dataset : 原始数据集
		trigger : 触发器
	"""
	
	def __init__(self, wrap_dataset, trigger : Trigger, *, poison_ratio=None, poison_num=None, seed=0, untargeted=False):
		# self.wrap_dataset = wrap_dataset
		assert hasattr(wrap_dataset, 'data')
		assert wrap_dataset.data.dtype == np.uint8

		target_img_indices = []

		if not untargeted:
			target_img_indices = np.array([ind for ind, t in enumerate(wrap_dataset.targets) 
						if t == trigger.target_class_id])
		else:
			target_img_indices = np.array(list(range(len(wrap_dataset))))

		np.random.seed(seed)
		if poison_ratio is not None:
			assert poison_ratio >= 0.0 and poison_ratio <= 1.0
			num_poison_images = int(len(target_img_indices) * poison_ratio)
			target_img_indices = np.random.choice(target_img_indices, size=num_poison_images, replace=False)
		elif poison_num is not None:
			poison_num = np.minimum(poison_num, len(target_img_indices))
			target_img_indices = np.random.choice(target_img_indices, size=poison_num, replace=False)

		data = wrap_dataset.data[target_img_indices]
		targets = wrap_dataset.targets[target_img_indices]

		for ind in range(len(data)):
			data[ind] = trigger.paste_to_np_img(data[ind])

		super(TriggerPastedDataset, self).__init__(data, targets, wrap_dataset.transform)


