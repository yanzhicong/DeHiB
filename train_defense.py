import os
from random import choices
import sys
import argparse
import logging

from torch.utils.data.dataset import Dataset
from torchvision import transforms
os.environ["OMP_NUM_THREADS"] = "1"


import numpy as np
import torch
import torch.backends.cudnn as cudnn

import pickle as pkl

from utils.args_utils import *
from utils.train_utils import enable_multiprocessing_gpu_training, ModelEMA, load_checkpoint, DatasetExpandWrapper
from utils.test_utils import infer_dataloader
from utils.model_utils import get_model
from utils.dataset_utils import get_dataset_for_ssl, get_transform, ConcatDataset
from utils.trigger_utils import Trigger, TriggerPastedDataset

from torch.utils.data import DataLoader
from dataset.dataset_utils import DatasetWrapper



from backdoorlib import dehib
from ssllib import plain
from ssllib import basetrainer
from ssllib import fixmatch
from ssllib import lpssl
from ssllib import dp_fixmatch


logger = logging.getLogger(__name__)


def get_pretrain_path(args):
	return 'pretrain_'+"pe%d"%args.pretrain_epochs



def get_poison_path(args):
	if args.poison_method in ["dehib", "dehib2", "dehib3"]:
		sub_list = [
		"n%d"%args.num_poisoned,
		"tc%d"%args.target_class,
		"lr%0.4f"%args.poison_lr,
		"ni%d"%args.poison_num_iter,
		"e%d"%args.poison_eps,
		]

		if args.poison_method != "dehib3":
			sub_list.append("lam%0.4f"%args.poison_lam)

		if args.poison_mean_ratio != 0.1:
			sub_list.append("pmr%0.2f"%args.poison_mean_ratio)

		return args.poison_method + '_' + '_'.join(sub_list)

	elif args.poison_method in ["naive"]:
		return "naive_n%d"%args.num_poisoned + "_tc%d"%args.target_class

	elif args.poison_method == "none":
		return "none"


def get_debias_path(args):
	sub_list  = [
		"lr%0.4f"%args.debias_learning_rate,
		"e%d"%args.debias_epochs,
		"kimg%d"%args.debias_k_img,
		'alpha%0.4f'%args.debias_mix_alpha,
		"wd%0.6f"%args.debias_wdecay,
	]

	if args.debias_wdecay_l1 != 0.0:
		sub_list.append("wd1%0.6f"%args.debias_wdecay_l1)

	exp_name = 'mixup_' + '_'.join(sub_list) 		
	return exp_name


def check_poison_images(output_dir, num_poison_images):
	if os.path.exists(output_dir):
		poison_images = DatasetWrapper.load_imgs_from_dir(output_dir)
		if len(poison_images) == num_poison_images:
			return poison_images
	return None


def load_pretrain_parameters(args):
	state = {
		"epochs" : args.epochs,
		"k_img" : args.k_img,
		"lr" : args.lr,
		"wdecay" : args.wdecay,
		"mix_alpha" : args.mix_alpha,
	}
	args.lr = args.pretrain_learning_rate
	args.epochs = args.pretrain_epochs
	args.k_img = args.pretrain_k_img
	args.wdecay = args.pretrain_wdecay
	args.mix_alpha = args.pretrain_mix_alpha
	return state


def load_fixmatch_parameters(args):
	state = {
		"epochs" : args.epochs,
		"k_img" : args.k_img,
		"lr" : args.lr,
		"wdecay" : args.wdecay,
		"mix_alpha" : args.mix_alpha,
	}
	args.lr = args.fixmatch_learning_rate
	args.epochs = args.fixmatch_epochs
	args.k_img = args.fixmatch_k_img
	args.wdecay = args.fixmatch_wdecay
	args.mix_alpha = args.fixmatch_mix_alpha
	return state



def load_debias_parameters(args):
	state = {
		"epochs" : args.epochs,
		"k_img" : args.k_img,
		"lr" : args.lr,
		"wdecay" : args.wdecay,
		"wdecay_l1" : args.wdecay_l1,
		"mix_alpha" : args.mix_alpha,
	}
	args.lr = args.debias_learning_rate
	args.epochs = args.debias_epochs
	args.k_img = args.debias_k_img
	args.wdecay = args.debias_wdecay
	args.wdecay_l1 = args.debias_wdecay_l1
	args.mix_alpha = args.debias_mix_alpha
	return state




def restore_parameters(args, state):
	args.epochs = state["epochs"]
	args.k_img = state["k_img"]
	args.lr = state["lr"]
	args.wdecay = state["wdecay"]
	args.mix_alpha = state["mix_alpha"]
	if "wdecay_l1" in state:
		args.wdecay_l1 = state["wdecay_l1"]



def log_dataset_info(dataset, name):
	logger.info("{} , length : {}".format(name, len(dataset)))
	class_list = np.unique(dataset.targets)
	for class_ind in class_list:
		logger.info("\t class {} : {}".format(class_ind, np.sum(dataset.targets == class_ind)))



def train_discriminator(args, labeled_ds, unlabeled_ds, poison_ds, test_ds, model, output_dir):

	labeled_ds = labeled_ds.clone()
	unlabeled_ds = unlabeled_ds.clone()	
	poison_ds = poison_ds.clone()

	labeled_ds.targets = np.zeros_like(labeled_ds.targets)
	unlabeled_ds.targets = np.ones_like(unlabeled_ds.targets)
	poison_ds.targets = np.ones_like(poison_ds.targets)


	val_labeled_ds = DatasetWrapper(labeled_ds.data, labeled_ds.targets, test_ds.transform)
	val_poison_ds = DatasetWrapper(poison_ds.data, poison_ds.targets, test_ds.transform)
	val_unlabeled_ds = DatasetWrapper(unlabeled_ds.data, unlabeled_ds.targets, test_ds.transform)


	log_dataset_info(val_labeled_ds, 'val_labeled_ds')
	log_dataset_info(val_poison_ds, 'val_poison_ds')
	log_dataset_info(val_unlabeled_ds, 'val_unlabeled_ds')


	val_labeled_lder = DataLoader(val_labeled_ds, batch_size=args.batch_size, num_workers=args.num_workers)
	val_poison_lder = DataLoader(val_poison_ds, batch_size=args.batch_size, num_workers=args.num_workers)
	val_unlabeled_lder = DataLoader(val_unlabeled_ds, batch_size=args.batch_size, num_workers=args.num_workers)


	unlabeled_ds = ConcatDataset([unlabeled_ds, poison_ds])
	labeled_ds = DatasetExpandWrapper(labeled_ds, len(unlabeled_ds))
	log_dataset_info(unlabeled_ds, 'unlabeled_ds')
	log_dataset_info(labeled_ds, 'labeled_ds')
	
	train_ds = ConcatDataset([labeled_ds, unlabeled_ds, ])
	log_dataset_info(train_ds, 'train_ds')



	def callback_before_epoch(args, epoch, model, ema_model, plotter):

		# 查看labeled data的输出分布
		labeled_probs = infer_dataloader(args, model, val_labeled_lder)
		logger.info('labeled_probs : {}'.format(labeled_probs.shape))
		plotter.scalar_probs('inferred_dist_as_l_labeled_data', epoch, labeled_probs[:, 0])
		plotter.scalar_probs('inferred_dist_as_u_labeled_data', epoch, labeled_probs[:, 1])

		plotter.scalar('inferred_prob_as_l_labeled_data', epoch, float(np.sum(labeled_probs[:, 0] > 0.5)) / float(labeled_probs.shape[0]) )

		poison_probs = infer_dataloader(args, model, val_poison_lder)
		logger.info('poison_probs : {}'.format(poison_probs.shape))
		plotter.scalar_probs('inferred_dist_as_l_poison_data', epoch, poison_probs[:, 0])
		plotter.scalar_probs('inferred_dist_as_u_poison_data', epoch, poison_probs[:, 1])

		plotter.scalar('inferred_prob_as_u_poison_data', epoch, float(np.sum(poison_probs[:, 1] > 0.5)) / float(poison_probs.shape[0]) )

		unlabeled_probs = infer_dataloader(args, model, val_unlabeled_lder)
		logger.info('unlabeled_probs : {}'.format(unlabeled_probs.shape))
		plotter.scalar_probs('inferred_dist_as_l_unlabeled_data', epoch, unlabeled_probs[:, 0])
		plotter.scalar_probs('inferred_dist_as_u_unlabeled_data', epoch, unlabeled_probs[:, 1])

		plotter.scalar('inferred_prob_as_u_unlabeled_data', epoch, float(np.sum(unlabeled_probs[:, 1] > 0.5)) / float(unlabeled_probs.shape[0]) )

		y_scores = np.concatenate([
			labeled_probs[:, 1], unlabeled_probs[:, 1], poison_probs[:, 1],
		])

		y_trues = np.concatenate([
			np.zeros_like(labeled_probs[:, 1], dtype=np.int32), np.zeros_like(unlabeled_probs[:, 1], dtype=np.int32), np.ones_like(poison_probs[:, 1], dtype=np.int32),
		])

		plotter.roc_auc('poison_roc', epoch, y_scores, y_trues)

		if epoch % 5 == 0:
			plotter.add_roc_curve("poison_roc_epoch_{}".format(epoch), y_scores, y_trues, output_dir)
			y_labels = np.concatenate([
				np.zeros_like(labeled_probs[:, 1], dtype=np.int32), np.ones_like(unlabeled_probs[:, 1], dtype=np.int32), np.ones_like(poison_probs[:, 1], dtype=np.int32) * 2,
			])
			plotter.add_prob_distribution_hist("inferred_distribution_epoch_{}".format(epoch), y_scores, y_labels, output_dir)

	train_ds.transform = get_transform(args.dataset, train=True)

	trainer = basetrainer.Trainer(args, model, output_dir)
	trainer.train(train_ds, None, callback_before_epoch=callback_before_epoch)




def main():
	parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
	# add_trigger_specification(parser)



	parser.add_argument('--batch-size', default=64, type=int, help='train batchsize')
	parser.add_argument('--wdecay', default=0.0005, type=float, help='weight decay')
	parser.add_argument('--wdecay-l1', default=0.0, type=float, help='weight decay')
	parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
	parser.add_argument('--nesterov', action='store_true', default=True,
						help='use nesterov momentum')
	parser.add_argument('--ema-decay', default=0.999, type=float,
						help='EMA decay rate')
	parser.add_argument('--k-img', default=8192, type=int,
						help='number of labeled examples per epoch')
	parser.add_argument('--epochs', default=200, type=int,
						help='number of total epochs to run')
	parser.add_argument('--early-stop-epoch', default=-1, type=int,
						help='manual epoch number (useful on restarts)')
	parser.add_argument('--mix-alpha', default=0.0, type=float,
						help='for supervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
	parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
						help='initial learning rate')
	parser.add_argument('--optimizer', default='sgd', type=str,
						help='optimizer to use')

	parser.add_argument('--learning-rate-schedule', default='cosine_annealing', type=str,
						help="learning rate schedule ( cosine_annealing | consistent)")
	parser.add_argument('--learning-rate-cosine-cycles', default=1.0, type=float,
						help='learning rate schedule')
	parser.add_argument('--learning-rate-cosine-end-lr-decay', default=0.001, type=float,
						help='the ending learning rate will be lr * lr_decay')

	parser.add_argument('--warmup-epochs', default=0, type=float,
						help='warmup epochs (unlabeled data based)')


	# 数据中毒参数
	parser.add_argument('--poison-method', default="naive", type=str)

	parser.add_argument('--poison-lr', default=0.03, type=float)
	parser.add_argument('--poison-lr-decay-steps', default=150, type=int)
	parser.add_argument('--poison-lam', default=1.0, type=float)
	parser.add_argument('--poison-eps', default=32.0, type=float)
	parser.add_argument('--poison-num-iter', default=1000, type=int)
	
	parser.add_argument('--poison-alpha', default=1.0, type=float)
	parser.add_argument('--poison-batch-size', default=100, type=int)
	parser.add_argument('--num-poisoned', default=1000, type=int)
	parser.add_argument('--poison-seed', type=int, default=100)

	parser.add_argument('--target-class', default=-1, type=int)

	parser.add_argument('--trigger-id', default=10, type=int, help='trigger id')
	parser.add_argument('--patch-size', default=8, type=int, help='patch size')
	parser.add_argument('--rand-loc', action="store_true")



	# 输出根目录
	parser.add_argument('--out', default='result', help='directory to output the result')


	# 预训练模型参数
	parser.add_argument('--pretrain-learning-rate', type=float, default=0.03)
	parser.add_argument('--pretrain-wdecay', type=float, default=0.0)
	parser.add_argument('--pretrain-epochs', type=int, default=50)
	parser.add_argument('--pretrain-k-img', type=int, default=8192)
	parser.add_argument('--pretrain-mix-alpha', default=0.0, type=float, help='for supervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')



	parser.add_argument('--debias-method', type=str, default="mixup")
	parser.add_argument('--debias-learning-rate', type=float, default=0.03)
	parser.add_argument('--debias-wdecay', type=float, default=0.00001)
	parser.add_argument('--debias-wdecay-l1', type=float, default=0.00001)
	parser.add_argument('--debias-epochs', type=int, default=50)
	parser.add_argument('--debias-k-img', type=int, default=65536*2)
	parser.add_argument('--debias-mix-alpha', default=4.0, type=float)
	parser.add_argument('--debias-eps', default=50, type=int)

	parser.add_argument('--debias_strategy', default="dp", choices=["oc", "dp", "ocdp", "dpoc", "none"], type=str)					   # 
	parser.add_argument('--debias-optimal-mask', action="store_true", default=False)   # 是否采用trigger pattern 的 ground-truth mask


	parser.add_argument('--ssl-method', default="fixmatch", type=str)			# fixmatch or lpssl or none



	# Fixmatch Parameters
	parser.add_argument('--fixmatch-learning-rate', type=float, default=0.03)
	parser.add_argument('--fixmatch-wdecay', type=float, default=0.0005)
	parser.add_argument('--fixmatch-epochs', type=int, default=200)
	parser.add_argument('--fixmatch-k-img', type=int, default=8192*2)
	parser.add_argument('--fixmatch-mix-alpha', default=0.0, type=float, help='for supervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
	parser.add_argument('--fixmatch-mu', default=4, type=int, help='coefficient of unlabeled batch size')
	parser.add_argument('--fixmatch-lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
	parser.add_argument('--fixmatch-threshold', default=0.95, type=float, help='pseudo label threshold')


	args = add_common_train_args(parser)

	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

	logger.info(dict(args._get_kwargs()))


	# 原始数据
	labeled_ds, unlabeled_ds, test_ds = get_dataset_for_ssl(args)


	enable_multiprocessing_gpu_training(args)
	logger.warning(
		f"Process rank: {args.local_rank}, "
		f"device: {args.device}, "
		f"n_gpu: {args.n_gpu}, "
		f"distributed training: {bool(args.local_rank != -1)}",)




	# 模型
	model = get_model(args)



	# 触发器
	target_class = (args.target_class + args.num_classes) % args.num_classes
	args.target_class = target_class
	trigger = Trigger(target_class, trigger_id=args.trigger_id, patch_size=args.patch_size, rand_loc=args.rand_loc)






	# 准备预训练的模型
	output_dir = os.path.join(args.out, args.ssl_dataset_name + "_" + args.model_name + "_" + trigger.name, get_pretrain_path(args))
	os.makedirs(output_dir, exist_ok=True)
	if not os.path.exists(os.path.join(output_dir, 'checkpoint.pth.tar')):
		state = load_pretrain_parameters(args)
		trainer = basetrainer.Trainer(args, model, output_dir, trigger=trigger)
		labeled_ds.transform = get_transform(args.dataset, train=True)
		test_ds.transform = get_transform(args.dataset, train=False)
		trainer.train(labeled_ds, test_ds)
		restore_parameters(args, state)
	load_checkpoint(args, os.path.join(output_dir, 'checkpoint.pth.tar'), model)




	# 数据中毒
	if args.poison_method != "none":

		output_dir = os.path.join(args.out, args.ssl_dataset_name + "_" + args.model_name + "_" + trigger.name, get_pretrain_path(args), get_poison_path(args), "poisoned")
		poison_images = check_poison_images(output_dir, args.num_poisoned)

		if poison_images is None:

			# generate poisoned examples
			output_dir = os.path.join(args.out, args.ssl_dataset_name + "_" + args.model_name + "_" + trigger.name, get_pretrain_path(args), get_poison_path(args))
			os.makedirs(output_dir, exist_ok=True)

			if args.poison_method == "dehib":
				poisoned_ds = dehib.dehib_poison(
					args, labeled_ds, unlabeled_ds, args.num_poisoned, model, 
					get_transform(args.dataset, only_norm=True), trigger, output_dir, seed=args.poison_seed, batch_size=args.poison_batch_size)

			elif args.poison_method == "naive":
				poisoned_ds = TriggerPastedDataset(unlabeled_ds.clone(), trigger, poison_num=args.num_poisoned, seed=args.poison_seed, untargeted=False)
				os.makedirs(os.path.join(output_dir, "poisoned"), exist_ok=True)
				poisoned_ds.save_imgs_to_dir(os.path.join(output_dir, "poisoned"))
			else:
				raise ValueError("")

		else:
			poisoned_ds = DatasetWrapper(poison_images, np.zeros([len(poison_images), ], dtype=np.int32), 
						transform=unlabeled_ds.transform)


	if args.use_ema:
		ema_model = ModelEMA(args, model, args.ema_decay, args.device)
	else:
		ema_model = None


	if args.debias_method == "none":
		pass



	elif args.debias_method == "mixup":

		state = load_debias_parameters(args)
		output_dir = os.path.join(args.out, args.ssl_dataset_name + "_" + args.model_name + "_" + trigger.name, get_pretrain_path(args), get_poison_path(args), get_debias_path(args))
		debias_model = get_model(args, num_classes=2)
	
		os.makedirs(output_dir, exist_ok=True)
		if not os.path.exists(os.path.join(output_dir, 'checkpoint.pth.tar')):
			train_discriminator(args, labeled_ds, unlabeled_ds, poisoned_ds, test_ds, debias_model, output_dir)
	
		load_checkpoint(args, os.path.join(output_dir, 'checkpoint.pth.tar'), debias_model)
		restore_parameters(args, state)


		unlabeled_ds = ConcatDataset([
			unlabeled_ds, poisoned_ds
		])


		if args.debias_strategy == "dp":

			if args.debias_optimal_mask:
			
				unlabeled_optimal_mask = np.zeros(unlabeled_ds.data.shape[0:3], dtype=np.float32)
				logger.info("unlabeled_optimal_mask : {}".format(unlabeled_optimal_mask.shape))
				unlabeled_optimal_mask[-len(poisoned_ds):, (32-8-5):(32-5), (32-8-5):(32-5)] = 1.0
				unlabeled_ds = dp_fixmatch.DPDataset(unlabeled_ds, unlabeled_optimal_mask, eps=float(args.debias_eps))
				debiaser_name = "_dpop%d"%args.debias_eps

			else:

				# cam_mask.pkl file is generated by running train_defense_test.py
				output_file_path = os.path.join(args.out, args.ssl_dataset_name + "_" + args.model_name + "_" + trigger.name, get_pretrain_path(args), get_poison_path(args), get_debias_path(args), 'cam_mask.pkl')
				assert os.path.isfile(output_file_path), "cam_mask.pkl file is generated by running train_defense_test.py, please run it first"
				unlabeled_mask = pkl.load(open(output_file_path, 'rb'))
				logger.info("unlabeled_mask : {}".format(unlabeled_mask.shape))
				unlabeled_ds = dp_fixmatch.DPDataset(unlabeled_ds, unlabeled_mask, eps=float(args.debias_eps))
				debiaser_name = "_dp%d"%args.debias_eps

		elif args.debias_strategy == "oc":

			from dataset.dataset_utils import cifar10_mean

			if args.debias_optimal_mask:
				unlabeled_optimal_mask = np.zeros(unlabeled_ds.data.shape[0:3], dtype=np.float32)
				unlabeled_optimal_mask[-len(poisoned_ds):, (32-8-5):(32-5), (32-8-5):(32-5)] = 1.0
				unlabeled_ds = dp_fixmatch.OccludeDataset(unlabeled_ds, unlabeled_optimal_mask, mean=cifar10_mean)
				debiaser_name = "_ocop"
				
			else:
				# cam_mask.pkl file is generated by running train_defense_test.py
				output_file_path = os.path.join(args.out, args.ssl_dataset_name + "_" + args.model_name + "_" + trigger.name, get_pretrain_path(args), get_poison_path(args), get_debias_path(args), 'cam_mask.pkl')
				assert os.path.isfile(output_file_path), "cam_mask.pkl file is generated by running train_defense_test.py, please run it first"
				unlabeled_mask = pkl.load(open(output_file_path, 'rb'))
				unlabeled_ds = dp_fixmatch.OccludeDataset(unlabeled_ds, unlabeled_mask, mean=cifar10_mean)
				debiaser_name = "_oc"


		elif args.debias_strategy == "ocdp" or args.debias_strategy == "dpoc":

			from dataset.dataset_utils import cifar10_mean

			if args.debias_optimal_mask:
				unlabeled_optimal_mask = np.zeros(unlabeled_ds.data.shape[0:3], dtype=np.float32)
				unlabeled_optimal_mask[-len(poisoned_ds):, (32-8-5):(32-5), (32-8-5):(32-5)] = 1.0
				unlabeled_ds = dp_fixmatch.DPOccludeDataset(unlabeled_ds, unlabeled_optimal_mask, mean=cifar10_mean, eps=float(args.debias_eps))
				debiaser_name = "_ocdpop%d"%args.debias_eps
				
			else:
				# cam_mask.pkl file is generated by running train_defense_test.py
				output_file_path = os.path.join(args.out, args.ssl_dataset_name + "_" + args.model_name + "_" + trigger.name, get_pretrain_path(args), get_poison_path(args), get_debias_path(args), 'cam_mask.pkl')
				assert os.path.isfile(output_file_path), "cam_mask.pkl file is generated by running train_defense_test.py, please run it first"
				unlabeled_mask = pkl.load(open(output_file_path, 'rb'))
				unlabeled_ds = dp_fixmatch.DPOccludeDataset(unlabeled_ds, unlabeled_mask, mean=cifar10_mean, eps=float(args.debias_eps))
				debiaser_name = "_ocdp%d"%args.debias_eps

		elif args.debias_strategy == "none":
			return



		state = load_fixmatch_parameters(args)
		output_dir = os.path.join(args.out, args.ssl_dataset_name + "_" + args.model_name + "_" + trigger.name, get_pretrain_path(args), get_poison_path(args), get_debias_path(args), fixmatch.FixmatchTrainer.get_experiment_name(args) + debiaser_name)
		os.makedirs(output_dir, exist_ok=True)
		trainer = fixmatch.FixmatchTrainer(args, model, output_dir, 
					ema_model=ema_model, trigger=trigger)
		trainer.train(labeled_ds, unlabeled_ds, test_ds)
		restore_parameters(args, state)







if __name__ == '__main__':
	cudnn.benchmark = True
	main()

