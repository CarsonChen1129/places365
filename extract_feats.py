import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os
import argparse
import math

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from munch import munchify

C, H, W = 3, 224, 224


class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


class TransformImage(object):

    def __init__(self, opts, scale=0.875, random_crop=False, random_hflip=False, random_vflip=False):
        if type(opts) == dict:
            opts = munchify(opts)
        self.input_size = opts.input_size
        self.input_space = opts.input_space
        self.input_range = opts.input_range
        self.mean = opts.mean
        self.std = opts.std

        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py#L294
        self.scale = scale
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip

        tfs = []
        tfs.append(transforms.Resize(int(math.floor(max(self.input_size) / self.scale))))

        if random_crop:
            tfs.append(transforms.RandomCrop(max(self.input_size)))
        else:
            tfs.append(transforms.CenterCrop(max(self.input_size)))

        if random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        if random_vflip:
            tfs.append(transforms.RandomVerticalFlip())

        tfs.append(transforms.ToTensor())
        tfs.append(ToSpaceBGR(self.input_space == 'BGR'))
        tfs.append(ToRange255(max(self.input_range) == 255))
        tfs.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor


class LoadImage(object):

    def __init__(self, space='RGB'):
        self.space = space

    def __call__(self, path_img):
        with open(path_img, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert(self.space)
        return img


class LoadTransformImage(object):

    def __init__(self, model, scale=0.875):
        self.load = LoadImage()
        self.tf = TransformImage(model, scale=scale)

    def __call__(self, path_img):
        img = self.load(path_img)
        tensor = self.tf(img)
        return tensor


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def load_image(path_img):
    with open(path_img, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')

    return img

def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            print(" cleanup: " + dst + "/")
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   '-y',
                                   '-i', video,  # input file
                                   '-vf', "scale=400:300",  # input file
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)

def extract_feats(params, model):
    global C, H, W
    model.eval()

    # load the image transformer
    centre_crop = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the class label
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    print("save video feats to %s" % (dir_fc))
    video_list = glob.glob(os.path.join(params['video_path'], '*.mp4'))
    for video in tqdm(video_list):
        video_id = video.split("/")[-1].split(".")[0]
        outfile = os.path.join(dir_fc, video_id + '.npy')

        if os.path.exists(outfile):
            continue

        dst = params['model'] + '_' + video_id
        extract_frames(video, dst)

        image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))

        if len(image_list)==0:
            continue

        samples = np.round(np.linspace(
            0, len(image_list) - 1, params['n_frame_steps']))

        image_list = [image_list[int(sample)] for sample in samples]
        images = torch.zeros((len(image_list), C, H, W))
        for iImg in range(len(image_list)):
            # img = load_image_fn(image_list[iImg])
            img = centre_crop(load_image(image_list[iImg]))
            images[iImg] = img
        fc_feats = model(Variable(images).cuda()).squeeze()
        img_feats = fc_feats.data.cpu().numpy()
        # Save the inception features
        np.save(outfile, img_feats)
        # cleanup
        shutil.rmtree(dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='data/feats/places365', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=40,
                        help='how many frames to sampler per video')

    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='data/train-video', help='path to video dataset')
    parser.add_argument("--model", dest="model", type=str, default='resnet152',
                        help='the CNN model you want to use to extract_feats')
    parser.add_argument("--saved_model", dest="saved_model", type=str, default='',
                        help='the pretrained CNN model you want to use to extract_feats')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)
    if params['model'] == 'resnet50':
        C, H, W = 3, 224, 224
        # th architecture to use
        arch = 'resnet50'
        # load the pre-trained weights
        model_file = 'whole_%s_places365_python36.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        if params['gpu'] is '0':
            model = torch.load(model_file)
        else:
            model = torch.load(model_file, map_location=lambda storage,
                                                               loc: storage)  # model trained in GPU could be deployed in CPU machine like this!
        # load_image_fn = LoadTransformImage(model)
    else:
        print("doesn't support %s" % (params['model']))

    model = nn.DataParallel(model)
    if params['saved_model'] != '':
        model.load_state_dict(torch.load(params['saved_model']), strict=False)
    model = model.cuda()
    extract_feats(params, model)
