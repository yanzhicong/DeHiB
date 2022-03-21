import logging

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def mish(x):
	"""Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
	return x * torch.tanh(F.softplus(x))

def mixup_internal(x, y, alpha):
	'''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
	if alpha > 0.:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1.
	batch_size = x.size()[0]
	index = torch.randperm(batch_size).cuda()
	mixed_x = lam * x + (1 - lam) * x[index,:]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


class PSBatchNorm2d(nn.BatchNorm2d):
	"""How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

	def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
		super().__init__(num_features, eps, momentum, affine, track_running_stats)
		self.alpha = alpha
		self.update_batch_stats = True

	def forward(self, x):
		if self.update_batch_stats:
			return super().forward(x) + self.alpha
		else:
			return nn.functional.batch_norm(
				x, None, None, self.weight, self.bias, True, self.momentum, self.eps
			) + self.alpha


class BasicBlock(nn.Module):
	def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
		super(BasicBlock, self).__init__()
		self.bn1 = PSBatchNorm2d(in_planes, momentum=0.001)
		self.relu1 = mish

		self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = PSBatchNorm2d(out_planes, momentum=0.001)
		self.relu2 = mish

		self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
							   padding=1, bias=False)
		self.drop_rate = drop_rate
		self.equalInOut = (in_planes == out_planes)
		self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
																padding=0, bias=False) or None
		self.activate_before_residual = activate_before_residual

	def forward(self, x):
		if not self.equalInOut and self.activate_before_residual == True:
			x = self.relu1(self.bn1(x))
		else:
			out = self.relu1(self.bn1(x))
		out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
		if self.drop_rate > 0:
			out = F.dropout(out, p=self.drop_rate, training=self.training)
		out = self.conv2(out)
		return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
	def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
		super(NetworkBlock, self).__init__()
		self.layer = self._make_layer(
			block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

	def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
		layers = []
		for i in range(int(nb_layers)):
			layers.append(block(i == 0 and in_planes or out_planes, out_planes,
								i == 0 and stride or 1, drop_rate, activate_before_residual))
		return nn.Sequential(*layers)

	def forward(self, x):
		return self.layer(x)


class WideResNet(nn.Module):
	def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0):
		super(WideResNet, self).__init__()
		channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
		assert((depth - 4) % 6 == 0)
		n = (depth - 4) / 6
		block = BasicBlock
		# 1st conv before any network block
		self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
							   padding=1, bias=False)
		# 1st block
		self.block1 = NetworkBlock(n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
		# 2nd block
		self.block2 = NetworkBlock(n, channels[1], channels[2], block, 2, drop_rate)
		# 3rd block
		self.block3 = NetworkBlock(n, channels[2], channels[3], block, 2, drop_rate)
		# global average pooling and classifier
		self.bn1 = PSBatchNorm2d(channels[3], momentum=0.001)
		self.relu = mish
		self.fc = nn.Linear(channels[3], num_classes)
		self.channels = channels[3]

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight,
										mode='fan_out',
										nonlinearity='leaky_relu')
			elif isinstance(m, PSBatchNorm2d):
				nn.init.constant_(m.weight, 1.0)
				nn.init.constant_(m.bias, 0.0)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)
				nn.init.constant_(m.bias, 0.0)

	@property
	def embedding_size(self):
		return self.channels

	def forward(self, x, 
					target=None, 
					mixup_hidden=False,  
					mixup_alpha=0.1, 
					layers_mix=None):

		if mixup_hidden == True:
			layer_mix = random.randint(0,layers_mix)
				
			out = x
			if layer_mix == 0:
				out, y_a, y_b, lam = mixup_internal(out, target, mixup_alpha)
				out = self.conv1(x)
			out = self.block1(out)
	
			if layer_mix == 1:
				out, y_a, y_b, lam = mixup_internal(out, target, mixup_alpha)
				out = self.block2(out)
	
			if layer_mix == 2:
				out, y_a, y_b, lam = mixup_internal(out, target, mixup_alpha)
		   
			out = self.block3(out)
			if layer_mix == 3:
				out, y_a, y_b, lam = mixup_internal(out, target, mixup_alpha)
				out = self.relu(self.bn1(out))
			out = F.adaptive_avg_pool2d(out, 1)
			out = out.view(-1, self.channels)
			out = self.fc(out)
			if layer_mix == 4:
				out, y_a, y_b, lam = mixup_internal(out, target, mixup_alpha)
			lam = torch.tensor(lam).cuda()
			lam = lam.repeat(y_a.size())
			return out, y_a, y_b, lam
		else:
			out = x
			out = self.conv1(x)
			out = self.block1(out)
			out = self.block2(out)
			out = self.block3(out)
			out = self.relu(self.bn1(out))
			out = F.adaptive_avg_pool2d(out, 1)
			out = out.view(-1, self.channels)
			out = self.fc(out)
			return out


	def emb_and_cfy(self, x, display=False):
		out = self.conv1(x)
		out = self.block1(out)
		out = self.block2(out)
		out = self.block3(out)
		out = self.relu(self.bn1(out))
		out = F.adaptive_avg_pool2d(out, 1)
		out = out.view(-1, self.channels)
		return out, self.fc(out)

	def update_batch_stats(self, flag):
		for m in self.modules():
			if isinstance(m, PSBatchNorm2d):
				m.update_batch_stats = flag




def build_wideresnet(depth, widen_factor, dropout, num_classes):
	logger.info(f"Model: WideResNet {depth}x{widen_factor}")
	return WideResNet(depth=depth,
					  widen_factor=widen_factor,
					  drop_rate=dropout,
					  num_classes=num_classes)




build_wideresnet_32 = build_wideresnet

# def build_wideresnet_84(depth, widen_factor, dropout, num_classes):
# 	logger.info(f"Model: WideResNet 84 {depth}x{widen_factor}")

# 	if depth == 20:
# 		return ResNet84(BasicBlock2, [2, 2, 2, 2], widen_factor=widen_factor, num_classes=num_classes)
# 	elif depth == 28:
# 		return ResNet84(BasicBlock2, [3, 3, 3, 3], widen_factor=widen_factor, num_classes=num_classes)
# 	elif depth == 36:
# 		return ResNet84(BasicBlock2, [4, 4, 4, 4], widen_factor=widen_factor, num_classes=num_classes)



if __name__ == "__main__":

	# model = build_wideresnet_84(28, 2, 0, 10)
	model = build_wideresnet(28, 10, 0, 10)
	print(model.embedding_size)

	input1 = torch.rand(1, 3, 32, 32)

	# input1 = torch.rand(1, 3, 84, 84)

	model.eval()
	embedding, output1 = model.emb_and_cfy(input1, display=True)

	print(embedding.size())
	print(output1.size())

	parameter = list(model.parameters())[0]


	print(parameter[0,0])
	print(parameter.size())
	print(parameter.dtype)




	# print(len(list(model.parameters())))
	# print(list(model.parameters())[0].size())

