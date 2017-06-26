# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import parlai.core.build_data as build_data

import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import os
import copy
import numpy as np
from PIL import Image

_greyscale = '  .,:;crsA23hHG#98&@'

class ImageLoader():
	"""Extract image feature using pretrained CNN network.
	"""	
	@staticmethod
	def add_cmdline_args(argparser):
		argparser.add_arg('--image-cnntype', type=str, default='resnet152',
			help='which CNN archtecture to use to extract the image feature'+
			'current pretrained option can be found https://github.com/pytorch/vision')
		argparser.add_arg('--image-layernum', type=int, default=-1,
			help='which CNN layer of feature to extract.')        
		argparser.add_arg('--image-size', type=int, default=256,
			help='')
		argparser.add_arg('--image-cropsize', type=int, default=224,
			help='')

	def __init__(self, opt):
		
		self.opt = copy.deepcopy(opt)
		self.netCNN = None
		self.transform = None
		self.xs = None

	def init_cnn(self):
		"""Lazy initialization of preprocessor model in case we don't need any image preprocessing."""
		opt = self.opt
		self.cnn_type = opt['image_cnntype']
		self.layer_num = opt['image_layernum']
		self.image_size = opt['image_size']
		self.crop_size = opt['image_cropsize']
		self.datatype = opt['datatype']

		opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
		self.use_cuda = opt['cuda']

		if self.use_cuda:
			print('[ Using CUDA ]')
			torch.cuda.set_device(opt['gpu'])
		
		# initialize the pretrained CNN using pytorch.
		CNN = getattr(torchvision.models, self.cnn_type)
		
		# cut off the additional layer.
		self.netCNN = nn.Sequential(*list(CNN(pretrained=True).children())[:self.layer_num])
		
		# initialize the transform function using torch vision.
		self.transform = transforms.Compose([
							transforms.Scale(self.image_size),
							transforms.CenterCrop(self.crop_size),
							transforms.ToTensor(),
							transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
							])

		# container for single image
		self.xs = torch.FloatTensor(1, 3, self.crop_size, self.crop_size).fill_(0)

		if self.use_cuda:
			self.cuda()
			self.xs = self.xs.cuda()

		# make self.xs variable.
		self.xs = Variable(self.xs)

	def cuda(self):
		self.netCNN.cuda()

	def save(self, feature, path):
		feature = feature.cpu().data.numpy()
		np.save(path, feature)

	def extract(self, image, path):
		# check whether initlize CNN network.
		if not self.netCNN:
			self.init_cnn()
	
		self.xs.data.copy_(self.transform(image))
		# extract the image feature
		feature = self.netCNN(self.xs)
		# save the feature
		self.save(feature, path)
		return feature

	def img_to_ascii(self, path):
		im = Image.open(path)
		im.thumbnail((60, 40), Image.BICUBIC)
		im = im.convert('L')
		asc = []
		for y in range(0, im.size[1]):
			for x in range(0, im.size[0]):
				lum = 255 - im.getpixel((x, y))
				asc.append(_greyscale[lum * len(_greyscale) // 256])
			asc.append('\n')
		return ''.join(asc)

	def load(self, path):
		opt = self.opt
		mode = opt.get('image_mode', 'raw')
		if mode is None or mode == 'none':
			# don't need to load images
			return None
		elif mode == 'raw':
			# raw just returns RGB values
			return Image.open(path).convert('RGB')
		elif mode == 'ascii':
			# convert images to ascii ¯\_(ツ)_/¯
			return self.img_to_ascii(path)
		else:
			# otherwise, looks for preprocessed version under 'mode' directory
			prepath, imagefn = os.path.split(path)

			dpath = os.path.join(prepath, mode)

			if not os.path.exists(dpath):
				build_data.make_dir(dpath)

			imagefn = imagefn + '.npy'
			new_path = os.path.join(prepath, mode, imagefn)			
			
			if not os.path.isfile(new_path):
				return self.extract(Image.open(path).convert('RGB'), new_path)
			else:
				return np.load(new_path)



