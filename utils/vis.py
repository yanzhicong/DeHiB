import os
import sys
import shutil
import traceback
import logging

import numpy as np
import pandas as pd
import pickle as pkl

import contextlib
import base64


# import matplotlib
# matplotlib.use("Agg")

from yattag import Doc
from yattag import indent

import cv2

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.metrics import roc_curve, roc_auc_score

logger = logging.getLogger(__name__)

##################################################

class Plotter(object):
	'''
		class Plotter
		将训练过程中的数值化输出保存成html以及csv文件进行记录

		用法：
		1. 初始化：
			plotter = Plotter()

		2. 记录：
			plotter.scalar('loss1', step=10, value=0.1)

		3. 输出到html文件：
			plotter.to_html_report('./experiment/plt')
		
		4. 输出所有数据分别到单独的csv文件中：
			plotter.to_csv('./experiment/plt')
	'''

	def __init__(self, args=None):
		self._scalar_data_frame_dict = {}
		self._dist_data_frame_dict = {}
		
		self._out_svg_list = []
		self._upper_bound = {}
		self._lower_bound = {}
		self.args = args

	def _check_dir(self, path):
		p = os.path.dirname(path)
		if not os.path.exists(p):
			os.mkdir(p)
		return p

	def set_min_max(self, name, min_val=None, max_val=None):
		if min_val is not None:
			self._lower_bound[name] = min_val
		if max_val is not None:
			self._upper_bound[name] = max_val

	@property
	def scalar_names(self):
		return list(self._scalar_data_frame_dict.keys())

	def get(self, name):
		if name in self._scalar_data_frame_dict:
			return self._scalar_data_frame_dict[name]
		elif name in self._dist_data_frame_dict:
			return self._dist_data_frame_dict[name]
		else:
			return None

	def has(self, name):
		return name in self._scalar_data_frame_dict or name in self._dist_data_frame_dict

	def keys(self):
		return list(self._scalar_data_frame_dict.keys()) + list(self._dist_data_frame_dict.keys())


	def roc_auc(self, name, step, y_score, y_true):
		auc_score = roc_auc_score(y_true, y_score)
		self.scalar(name, step, auc_score)
	

	def add_roc_curve(self, name, y_score, y_true, output_dir):
		
		try:
			output_svg_filepath = os.path.join(output_dir, name + '.svg')
			plt.figure()
			plt.clf()
			fpr, tpr, thres = roc_curve(y_true, y_score)
			plt.plot(fpr, tpr)
			plt.grid(axis='y')
			plt.grid(axis='x', which='major')
			plt.grid(axis='x', which='minor', color=[0.9, 0.9, 0.9])
			
			plt.tight_layout()
			plt.savefig(output_svg_filepath)
			plt.close()

			self._out_svg_list.append([name, output_svg_filepath])
		except Exception as e:
			logger.info('draw %s failed : '%name)
			traceback.print_exc()


	def add_prob_distribution_hist(self, name, y_probs, y_labels, output_dir):
		
		label_list = np.unique(y_labels)
		label_list = sorted(label_list)
		alpha_list = np.linspace(0.0, 0.9, len(label_list)+2)[2:]
		color_list = ["#0000FF", "#00FF00", "#FF0000", "#FFFF00", "#FF00FF"]

		assert np.max(y_probs) <= 1.0
		assert np.min(y_probs) >= 0.0

		try:
			output_svg_filepath = os.path.join(output_dir, name + '.svg')
			output_txt_filepath = os.path.join(output_dir, name + '.txt')
			output_pkl_filepath = os.path.join(output_dir, 'prob_distribution_input.pkl')
			
			bins = np.linspace(0.0, 1.0, 100)

			plt.figure()
			plt.clf()

			for ind, l in enumerate(label_list):
				plt.hist(y_probs[np.where(y_labels == l)[0]], bins, density=True, color=color_list[ind%len(color_list)], alpha=alpha_list[ind])

			plt.tight_layout()
			plt.savefig(output_svg_filepath)
			plt.close()
			pkl.dump((y_probs, y_labels), open(output_pkl_filepath, 'wb'))

			self._out_svg_list.append([name, output_svg_filepath])
		except Exception as e:
			logger.info('draw %s failed : '%name)
			traceback.print_exc()





	def scalar(self, name, step, value, epoch=None):
		if isinstance(value, dict):
			data = value.copy()
			data.update({
				'step' : step
			})
		else:
			data = {
				'step' : step,
				name : value,
			}
		
		if epoch is not None:
			data['epoch'] = epoch
				
		df = pd.DataFrame(data, index=[0])

		if name not in self._scalar_data_frame_dict:
			self._scalar_data_frame_dict[name] = df
		else:
			self._scalar_data_frame_dict[name] = self._scalar_data_frame_dict[name].append(df, ignore_index=True)

	def scalar_probs(self, name, step, value_array, epoch=None):
		data_dict = {}
		for i in range(1, 10):
			thres = float(i) / 10.0
			data_dict[str(thres)] = float((value_array < thres).sum()) / float(len(value_array))
		self.scalar(name, step, data_dict, epoch=epoch)
		self.set_min_max(name, 0.0, 1.0)



	# def dist(self, name, step, mean, var, epoch=None):
	# 	if epoch is not None:
	# 		df = pd.DataFrame({'epoch' : epoch, 'step' : step, name+'_mean' : mean, name+'_var' : var }, index=[0])
	# 	else:
	# 		df = pd.DataFrame({'step' : step, name+'_mean' : mean, name+'_var' : var, }, index=[0])
	# 	if name not in self._dist_data_frame_dict:
	# 		self._dist_data_frame_dict[name] = df
	# 	else:
	# 		self._dist_data_frame_dict[name] = self._dist_data_frame_dict[name].append(df, ignore_index=True)
	# def dist2(self, name, step, value_list, epoch=None):
	# 	mean = np.mean(value_list)
	# 	var = np.var(value_list)
	# 	self.dist(name, step, mean, var, epoch=epoch)


	def to_csv(self, output_dir):
		" 将记录保存到多个csv文件里面，csv文件放在output_dir下面。"
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)

		for name, data_frame in self._scalar_data_frame_dict.items():
			csv_filepath = os.path.join(output_dir, 'scalar_'+name+'.csv')
			data_frame.to_csv(csv_filepath, index=False)
		for name, data_frame in self._dist_data_frame_dict.items():
			csv_filepath = os.path.join(output_dir, 'dist_'+name+'.csv')
			data_frame.to_csv(csv_filepath, index=False)

	def from_csv(self, output_dir):
		" 从output_dir下面的csv文件里面读取并恢复记录 "
		csv_name_list = [fn.split('.')[0] for fn in os.listdir(output_dir) if fn.endswith('csv')]
		for name in csv_name_list:
			if name.startswith('scalar_'):
				in_csv = pd.read_csv(os.path.join(output_dir, name+'.csv'))
				self._scalar_data_frame_dict[name[len('scalar_'):]] = in_csv
			elif name.startswith('dist_'):
				self._dist_data_frame_dict[name[len('dist_'):]] = pd.read_csv(os.path.join(output_dir, name+'.csv'))

	def write_svg_all(self, output_dir):
		" 将所有记录绘制成svg图片 "
		for ind, (name, data_frame) in enumerate(self._scalar_data_frame_dict.items()):
			try:
				min_val = self._lower_bound.get(name, None)
				max_val = self._upper_bound.get(name, None)
				output_svg_filepath = os.path.join(output_dir, name+'.svg')
				plt.figure()
				plt.clf()
				headers = [hd for hd in data_frame.columns if hd not in ['step', 'epoch']]
				if len(headers) == 1:
					plt.plot(data_frame['step'], data_frame[name])
				else:
					for hd in headers:
						plt.plot(data_frame['step'], data_frame[hd])
					plt.legend(headers)
				if min_val is not None:
					plt.ylim(bottom=min_val)
				if max_val is not None:
					plt.ylim(top=max_val)

				plt.grid(axis='y')
				plt.grid(axis='x', which='major')
				plt.grid(axis='x', which='minor', color=[0.9, 0.9, 0.9])
				
				plt.tight_layout()
				plt.savefig(output_svg_filepath)
				plt.close()
			except Exception as e:
				logger.info('draw %s failed : '%name)
				logger.info(data_frame)
				traceback.print_exc()


		for ind, (name, data_frame) in enumerate(self._dist_data_frame_dict.items()):
			output_svg_filepath = os.path.join(output_dir, name+'.svg')
			plt.figure()
			plt.clf()
			plt.errorbar(data_frame['step'], data_frame[name+'_mean'], yerr=data_frame[name+'_var'])
			plt.tight_layout()
			plt.savefig(output_svg_filepath)
			plt.close()


	def to_html_report(self, output_filepath):
		" 将所有记录整理成一个html报告 "
		self.write_svg_all(self._check_dir(output_filepath))
		doc, tag, text = Doc().tagtext()
		with open(output_filepath, 'w') as outfile:
			with tag('html'):
				with tag('body'):

					title_idx = 1

					if self.args is not None:
						with tag('h3'):
							text('{}. args'.format(title_idx))
							title_idx += 1
						
						key_list = list(self.args.keys())
						key_list = sorted(key_list)

						with tag('div', style='display:inline-block;width:850px;padding:5px;margin-left:20px'):
							for key in key_list:
								with tag('div', style='display:inline-block;width:400px;'):
									text('{} : {}\n'.format(key, self.args[key]))

					with tag('h3'):
						text('{}. scalars'.format(title_idx))
						title_idx += 1

					data_frame_name_list = [n for n, d in self._scalar_data_frame_dict.items()]
					data_frame_name_list = sorted(data_frame_name_list)

					for ind, name in enumerate(data_frame_name_list):
						with tag('div', style='display:inline-block'):
							with tag('h4', style='margin-left:20px'):
								text('(%d). '%(ind+1)+name)
							doc.stag("embed", style="width:800px;padding:5px;margin-left:20px", src=name+'.svg', type="image/svg+xml")


					with tag('h3'):
						text('{}. others'.format(title_idx))
						title_idx += 1

					for ind, (name, svg_path) in enumerate(self._out_svg_list):
						with tag('div', style='display:inline-block'):
							with tag('h4', style='margin-left:20px'):
								text('(%d). '%(ind+1)+name)
							doc.stag("embed", style="width:800px;padding:5px;margin-left:20px", src=name+'.svg', type="image/svg+xml")


					# with tag('h3'):
					# 	text('2. distributions')

					# for ind, (name, data_frame) in enumerate(self._dist_data_frame_dict.items()):
					# 	with tag('div', style='display:inline-block'):
					# 		with tag('h4', style='margin-left:20px'):
					# 			text('(%d). '%(ind+1)+name)
					# 		doc.stag("embed", style="width:800px;padding:5px;margin-left:20px", src=name+'.svg', type="image/svg+xml")

			result = indent(doc.getvalue())
			outfile.write(result)



class RecordInterface(object):
	def start_draw(self):
		raise NotImplementedError()

	def draw_image(self, image, title=''):
		raise NotImplementedError()

	# def draw_svg(self, image, title=''):
	# 	raise NotImplementedError();

	def _check_dir(self, path):
		if not os.path.exists(path):
			os.mkdir(path)
		return path



class RecordData(object):
	def draw(self, draw_interface):
		raise NotImplementedError()

def cv2_imread(filepath):
	filein = np.fromfile(filepath, dtype=np.uint8)
	cv_img = cv2.imdecode(filein, cv2.IMREAD_COLOR)
	return cv_img


def img_vertical_concat(images, pad=0, pad_value=255, pad_right=False):
	nb_images = len(images)
	h_list = [i.shape[0] for i in images]
	w_list = [w.shape[1] for w in images]
	if pad_right:
		max_w = np.max(w_list)
		images = [i if i.shape[1] == max_w else np.hstack([i, np.ones([i.shape[0], max_w-i.shape[1]]+list(i.shape[2:]), dtype=i.dtype)*pad_value])
				for i in images]
	else:
		assert np.all(np.equal(w_list, w_list[0]))

	if pad != 0:
		images = [np.vstack([i, np.ones([pad,]+list(i.shape[1:]), dtype=i.dtype)*pad_value]) for i in images[:-1]] + [images[-1],]

	if not isinstance(images, list):
		images = [i for i in images]
	return np.vstack(images)

def img_horizontal_concat(images, pad=0, pad_value=255, pad_bottom=False):
	nb_images = len(images)
	h_list = [i.shape[0] for i in images]
	w_list = [w.shape[1] for w in images]
	if pad_bottom:
		max_h = np.max(h_list)
		images = [i if i.shape[0] == max_h else np.vstack([i, np.ones([max_h-i.shape[0]]+list(i.shape[1:]), dtype=i.dtype)*pad_value])
				for i in images]
	else:
		assert np.all(np.equal(h_list, h_list[0]))

	if pad != 0:
		images = [np.hstack([i, np.ones([i.shape[0], pad,]+list(i.shape[2:]), dtype=i.dtype)*pad_value]) for i in images[:-1]] + [images[-1],]

	if not isinstance(images, list):
		images = [i for i in images]
	return np.hstack(images)
	

def img_grid(images, nb_images_per_row=10, pad=0, pad_value=255):
	ret = []
	while len(images) >= nb_images_per_row:
		ret.append(img_horizontal_concat(images[0:nb_images_per_row], pad_bottom=True, pad=pad, pad_value=pad_value))
		images = images[nb_images_per_row:]
	if len(images) != 0:
		ret.append(img_horizontal_concat(images, pad_bottom=True, pad=pad, pad_value=pad_value))
	return img_vertical_concat(ret, pad=pad, pad_right=True, pad_value=pad_value)



class RecordDataHelper(RecordData):

	def cv2_imread(self, filepath):
		filein = np.fromfile(filepath, dtype=np.uint8)
		cv_img = cv2.imdecode(filein, cv2.IMREAD_COLOR)
		return cv_img

	def img_vertical_concat(self, images, pad=0, pad_value=255, pad_right=False):
		nb_images = len(images)
		h_list = [i.shape[0] for i in images]
		w_list = [w.shape[1] for w in images]
		if pad_right:
			max_w = np.max(w_list)
			images = [i if i.shape[1] == max_w else np.hstack([i, np.ones([i.shape[0], max_w-i.shape[1]]+list(i.shape[2:]), dtype=i.dtype)*pad_value])
					for i in images]
		else:
			assert np.all(np.equal(w_list, w_list[0]))

		if pad != 0:
			images = [np.vstack([i, np.ones([pad,]+list(i.shape[1:]), dtype=i.dtype)*pad_value]) for i in images[:-1]] + [images[-1],]

		if not isinstance(images, list):
			images = [i for i in images]
		return np.vstack(images)

	def img_horizontal_concat(self, images, pad=0, pad_value=255, pad_bottom=False):
		nb_images = len(images)
		h_list = [i.shape[0] for i in images]
		w_list = [w.shape[1] for w in images]
		if pad_bottom:
			max_h = np.max(h_list)
			images = [i if i.shape[0] == max_h else np.vstack([i, np.ones([max_h-i.shape[0]]+list(i.shape[1:]), dtype=i.dtype)*pad_value])
					for i in images]
		else:
			assert np.all(np.equal(h_list, h_list[0]))

		if pad != 0:
			images = [np.hstack([i, np.ones([i.shape[0], pad,]+list(i.shape[2:]), dtype=i.dtype)*pad_value]) for i in images[:-1]] + [images[-1],]

		if not isinstance(images, list):
			images = [i for i in images]
		return np.hstack(images)
		
	def img_grid(self, images, nb_images_per_row=10, pad=0, pad_value=255):
		ret = []
		while len(images) >= nb_images_per_row:
			ret.append(self.img_horizontal_concat(images[0:nb_images_per_row], pad_bottom=True, pad=pad, pad_value=pad_value))
			images = images[nb_images_per_row:]
		if len(images) != 0:
			ret.append(self.img_horizontal_concat(images, pad_bottom=True, pad=pad, pad_value=pad_value))
		return self.img_vertical_concat(ret, pad_right=True)

	def label_img(self, img, lbl):
		img = np.vstack([img, np.ones([25,]+list(img.shape[1:]), dtype=img.dtype) * 255])
		return cv2.putText(img, str(lbl), (3, img.shape[0]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)


class HtmlRecorder(RecordInterface):

	@contextlib.contextmanager
	def start_draw(self, output_path, filename='index.html'):
		self._check_dir(output_path)
		self.doc, self.tag, self.text = Doc().tagtext()
		with open(os.path.join(output_path, filename), 'w') as outfile:
			with self.tag('html'):
				with self.tag('body'):
					self.index = 1
					yield
			result = indent(self.doc.getvalue())
			outfile.write(result)

	def draw_image(self, image, title=''):
			with self.tag('h3'):
				self.text('%d : %s'%(self.index, title))
				self.index += 1
			with self.tag('div', style='display:inline-block'):
				self.doc_image(image)

	def doc_image(self, img, width=500):
		img_buf = cv2.imencode('.jpg', img)[1]
		self.doc.stag("img", style="width:%dpx;padding:5px"%width, src="data:image/png;base64,"+str(base64.b64encode(img_buf))[2:-1])



if __name__ == "__main__":
	p = Plotter()

	p.scalar('loss', 1, 100)
	p.scalar('loss', 2, 100)
	p.scalar('loss', 3, 100)
	p.scalar('loss', 4, 100)
	p.scalar('loss', 5, 100)
	p.scalar('loss', 6, 100)
	p.scalar('loss', 7, 100)
	p.scalar('loss', 8, 100)
	p.scalar('loss', 9, 100)
	p.scalar('loss', 10, 100)
	p.scalar('loss', 11, 100)
	p.scalar('loss', 12, 100)

	p.scalar('loss2', 1, 100)
	p.scalar('loss2', 2, 100)
	p.scalar('loss2', 3, 100)
	p.scalar('loss2', 4, 100)
	p.scalar('loss2', 5, 100)
	p.scalar('loss2', 6, 100)
	p.scalar('loss2', 7, 100)
	p.scalar('loss2', 8, 100)
	p.scalar('loss2', 9, 100)
	p.scalar('loss2', 10, 100)
	p.scalar('loss2', 11, 100)
	p.scalar('loss2', 12, 100)

	# p.from_csv('./test_plot_output')
	p.to_html_report('./test_plot_output/output.html')
	p.to_csv('./test_plot_output')

