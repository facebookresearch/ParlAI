# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import parlai.core.build_data as build_data

import os
from PIL import Image
from functools import wraps
from threading import Lock, Condition

_greyscale = '  .,:;crsA23hHG#98&@'
_cache_size = 84000

def first_n_cache(function):
    cache = {}
    cache_monitor = CacheMonitor()

    @wraps(function)
    def wrapper(*args):
        path = args[1]
        loader = args[0]
        if path in cache:
            img = cache[path]
        else:
            img = function(*args)
            if img is not None and len(cache) < _cache_size:
                cache_monitor.waitForCache()
                cache[path] = img
                cache_monitor.doneWithCache()
        if loader.use_cuda and loader.im not in [None, 'none', 'raw', 'ascii']:
            img = loader.torch.from_numpy(img).cuda()
        return img
    return wrapper


class CacheMonitor():
    def __init__(self):
        self.cache_lock = Lock()
        self.cache_available = Condition(self.cache_lock)
        self.cache_busy = False

    def waitForCache(self):
        with self.cache_lock:
            while self.cache_busy:
                self.cache_available.wait()
            self.cache_busy = True

    def doneWithCache(self):
        with self.cache_lock:
            self.cache_busy = False
            self.cache_available.notify_all()


class ImageLoader():
    """Extract image feature using pretrained CNN network.
    """
    def __init__(self, opt):
        self.opt = opt.copy()
        self.use_cuda = False
        self.netCNN = None
        self.im = opt.get('image_mode', 'none')
        if self.im not in ['none', 'raw', 'ascii']:
            self.init_cnn(self.opt)

    def init_cnn(self, opt):
        """Lazy initialization of preprocessor model in case we don't need any
        image preprocessing.
        """
        try:
            import torch
            self.use_cuda = (not opt.get('no_cuda', False)
                             and torch.cuda.is_available())
            self.torch = torch
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Need to install Pytorch: go to pytorch.org')
        from torch.autograd import Variable
        import torchvision
        import torchvision.transforms as transforms
        import torch.nn as nn

        try:
            import h5py
            self.h5py = h5py
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Need to install h5py')

        if 'image_mode' not in opt or 'image_size' not in opt:
            raise RuntimeError(
                'Need to add image arguments to opt. See '
                'parlai.core.params.ParlaiParser.add_image_args')
        self.image_mode = opt['image_mode']
        self.image_size = opt['image_size']
        self.crop_size = opt['image_cropsize']

        if self.use_cuda:
            print('[ Using CUDA ]')
            torch.cuda.set_device(opt.get('gpu', -1))

        cnn_type, layer_num = self.image_mode_switcher()

        # initialize the pretrained CNN using pytorch.
        CNN = getattr(torchvision.models, cnn_type)

        # cut off the additional layer.
        self.netCNN = nn.Sequential(
            *list(CNN(pretrained=True).children())[:layer_num])

        # initialize the transform function using torch vision.
        self.transform = transforms.Compose([
            transforms.Scale(self.image_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # container for single image
        self.xs = torch.zeros(1, 3, self.crop_size, self.crop_size)

        if self.use_cuda:
            self.netCNN.cuda()
            self.xs = self.xs.cuda()

        # make self.xs variable.
        self.xs = Variable(self.xs)

    def save(self, feature, path):
        with open(path, 'w'):
            hdf5_file = self.h5py.File(path, 'w')
            hdf5_file.create_dataset('feature', data=feature)
            hdf5_file.close()

    def image_mode_switcher(self):
        switcher = {
            'resnet152': ['resnet152', -1],
            'resnet101': ['resnet101', -1],
            'resnet50': ['resnet50', -1],
            'resnet34': ['resnet34', -1],
            'resnet18': ['resnet18', -1],
            'resnet152_spatial': ['resnet152', -2],
            'resnet101_spatial': ['resnet101', -2],
            'resnet50_spatial': ['resnet50', -2],
            'resnet34_spatial': ['resnet34', -2],
            'resnet18_spatial': ['resnet18', -2],
        }

        if self.image_mode not in switcher:
            raise NotImplementedError('image preprocessing mode' +
                                      '{} not supported yet'.format(self.image_mode))

        return switcher.get(self.image_mode)

    def extract(self, image, path):
        # check whether initialize CNN network.
        if not self.netCNN:
            self.init_cnn(self.opt)

        self.xs.data.copy_(self.transform(image))
        # extract the image feature
        feature = self.netCNN(self.xs)
        save_feature = feature.cpu().data.numpy()
        # save the feature
        self.save(save_feature, path)
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

    # @first_n_cache
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

            imagefn = imagefn.split('.')[0]
            imagefn = imagefn + '.hdf5'
            new_path = os.path.join(prepath, mode, imagefn)

            if not os.path.isfile(new_path):
                return self.extract(Image.open(path).convert('RGB'), new_path)
            else:
                with open(new_path):
                    hdf5_file = self.h5py.File(new_path, 'r')
                    feature = hdf5_file['feature'].value
                feature = self.torch.from_numpy(feature)
                return feature
