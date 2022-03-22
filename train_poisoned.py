import os
import sys
import argparse
import logging

from torch.utils.data.dataset import Dataset
from torchvision import transforms

os.environ["OMP_NUM_THREADS"] = "1"


import numpy as np
import torch
import torch.backends.cudnn as cudnn


from utils.args_utils import *
from utils.train_utils import enable_multiprocessing_gpu_training, ModelEMA, load_checkpoint
from utils.model_utils import get_model
from utils.dataset_utils import get_dataset_for_ssl, get_transform, ConcatDataset
from utils.trigger_utils import Trigger, TriggerPastedDataset

from torch.utils.data import DataLoader
from dataset.dataset_utils import DatasetWrapper


from backdoorlib import dehib
from ssllib import plain
from ssllib import basetrainer
from ssllib import fixmatch


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

	if args.poison_method in ["turner", ]:
		sub_list = [
		"n%d"%args.num_poisoned,
		"tc%d"%args.target_class,
		"lr%0.4f"%args.poison_lr,
		"ni%d"%args.poison_num_iter,
		"e%d"%args.poison_eps,
		]

		return args.poison_method + '_' + '_'.join(sub_list)


	elif args.poison_method in ["clean"]:
		return "clean_n%d"%args.num_poisoned + "_tc%d"%args.target_class

	elif args.poison_method in ["naive"]:
		return "naive_n%d"%args.num_poisoned + "_tc%d"%args.target_class

	elif args.poison_method == "none":
		return "none"



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


def restore_parameters(args, state):
	args.epochs = state["epochs"]
	args.k_img = state["k_img"]
	args.lr = state["lr"]
	args.wdecay = state["wdecay"]
	args.mix_alpha = state["mix_alpha"]




def main():
	parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')



	parser.add_argument('--batch-size', default=64, type=int, help='train batchsize')
	parser.add_argument('--wdecay', default=0.0005, type=float, help='weight decay')
	parser.add_argument('--wdecay-l1', default=0.0, type=float, help='weight decay')
	parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
	parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
	parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')
	parser.add_argument('--k-img', default=8192, type=int, help='number of labeled examples per epoch')
	parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
	parser.add_argument('--early-stop-epoch', default=-1, type=int, help='manual epoch number (useful on restarts)')
	parser.add_argument('--mix-alpha', default=0.0, type=float, help='for supervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
	parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, help='initial learning rate')
	parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer to use')

	parser.add_argument('--learning-rate-schedule', default='cosine_annealing', type=str, help="learning rate schedule ( cosine_annealing | consistent)")
	parser.add_argument('--learning-rate-cosine-cycles', default=1.0, type=float, help='learning rate schedule')
	parser.add_argument('--learning-rate-cosine-end-lr-decay', default=0.001, type=float, help='the ending learning rate will be lr * lr_decay')

	parser.add_argument('--warmup-epochs', default=0, type=float, help='warmup epochs (unlabeled data based)')


	# 预训练模型参数
	parser.add_argument('--pretrain-learning-rate', type=float, default=0.03)
	parser.add_argument('--pretrain-wdecay', type=float, default=0.0)
	parser.add_argument('--pretrain-epochs', type=int, default=100)
	parser.add_argument('--pretrain-k-img', type=int, default=8192)
	parser.add_argument('--pretrain-mix-alpha', default=0.0, type=float, help='for supervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')


	# 数据中毒参数
	parser.add_argument('--poison-method', default="dehib", type=str)

	parser.add_argument('--poison-lr', default=0.03, type=float)
	parser.add_argument('--poison-lr-decay-steps', default=150, type=int)
	parser.add_argument('--poison-num-iter', default=1000, type=int)

	parser.add_argument('--poison-lam', default=1.0, type=float)
	parser.add_argument('--poison-eps', default=32.0, type=float)
	
	parser.add_argument('--poison-alpha', default=1.0, type=float)
	parser.add_argument('--poison-batch-size', default=100, type=int)
	parser.add_argument('--num-poisoned', default=200, type=int)
	parser.add_argument('--poison-seed', type=int, default=100)
	parser.add_argument('--target-class', default=-1, type=int)
	parser.add_argument('--trigger-id', default=10, type=int, help='trigger id')
	parser.add_argument('--patch-size', default=8, type=int, help='patch size')
	parser.add_argument('--rand-loc', action="store_true")



	# Fixmatch Parameters
	parser.add_argument('--fixmatch-learning-rate', type=float, default=0.03)
	parser.add_argument('--fixmatch-wdecay', type=float, default=0.0000)
	parser.add_argument('--fixmatch-epochs', type=int, default=400)
	parser.add_argument('--fixmatch-k-img', type=int, default=8192)
	parser.add_argument('--fixmatch-mix-alpha', default=0.0, type=float, help='for supervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
	parser.add_argument('--fixmatch-mu', default=7, type=int, help='coefficient of unlabeled batch size')
	parser.add_argument('--fixmatch-lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
	parser.add_argument('--fixmatch-threshold', default=0.95, type=float, help='pseudo label threshold')


	# 输出根目录
	parser.add_argument('--out', default='result', help='directory to output the result')

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
				poisoned_ds = dehib.dehib_poison(args, labeled_ds, unlabeled_ds, args.num_poisoned, model, 
					get_transform(args.dataset, only_norm=True), trigger, output_dir, seed=args.poison_seed, batch_size=args.poison_batch_size)

			elif args.poison_method == "naive":
				poisoned_ds = TriggerPastedDataset(unlabeled_ds.clone(), trigger, poison_num=args.num_poisoned, seed=args.poison_seed, untargeted=False)
				os.makedirs(os.path.join(output_dir, "poisoned"), exist_ok=True)
				poisoned_ds.save_imgs_to_dir(os.path.join(output_dir, "poisoned"))
			
			else:
				raise ValueError("Unknown poison method")

		else:
			poisoned_ds = DatasetWrapper(poison_images, np.zeros([len(poison_images), ], dtype=np.int32), 
						transform=unlabeled_ds.transform)

		if args.poison_method != "clean":
			# clean的代码是直接在原图上修改，不需要进行混合
			unlabeled_ds = ConcatDataset([unlabeled_ds, poisoned_ds])



	if args.use_ema:
		ema_model = ModelEMA(args, model, args.ema_decay, args.device)
	else:
		ema_model = None


	# 半监督训练
	state = load_fixmatch_parameters(args)
	output_dir = os.path.join(args.out, args.ssl_dataset_name + "_" + args.model_name + "_" + trigger.name, get_pretrain_path(args), get_poison_path(args), fixmatch.FixmatchTrainer.get_experiment_name(args))
	os.makedirs(output_dir, exist_ok=True)
	trainer = fixmatch.FixmatchTrainer(args, model, output_dir, ema_model=ema_model, trigger=trigger)
	trainer.train(labeled_ds, unlabeled_ds, test_ds)
	restore_parameters(args, state)



if __name__ == '__main__':
	cudnn.benchmark = True
	main()

