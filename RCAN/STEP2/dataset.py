import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import glob
import numpy as np
import PIL.Image as pil_image

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#tf.enable_eager_execution(config=config)
import cv2


class Dataset(object):
    def __init__(self, images_dir, patch_size, scale, use_fast_loader=False):
        self.hr_image_files = sorted(glob.glob(HR_images_dir + '/*'))
        self.patch_size = patch_size
        self.scale = scale
        self.use_fast_loader = use_fast_loader

    def __getitem__(self, idx):
    #use_fast_loader = false
        if self.use_fast_loader:
            hr = tf.read_file(self.hr_image_files[idx])
            hr = tf.image.decode_jpeg(hr, channels=3)
            hr = pil_image.fromarray(hr.numpy())
        else:
            hr = pil_image.open(self.hr_image_files[idx].convert('RGB'))
        
        
        lr = hr.resize((hr.width // (opt.scale * 2), hr.height // (opt.scale* 2)), resample=pil_image.BICUBIC)
        
        lr = hr.resize((hr.width // opt.scale, hr.height // opt.scale), resample=pil_image.BICUBIC) 
        lr = lr.resize((hr.width // 2        , hr.height // 2        ), resample=pil_image.BICUBIC)
        
        # randomly crop patch from training set
        crop_x = random.randint(0, hr.width - self.patch_size)
        crop_y = random.randint(0, hr.height - self.patch_size)
        
        #hr lr no onazi iti kara patch wo kiridasi
        hr = hr.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))
        lr = lr.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))
        
        
        ###rotate augmentation
        angle = random.randint(-180, 180)
        hr = hr.rotate(angle)
        lr = lr.rotate(angle)  
        
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        
        hr = np.transpose(hr, axes=[2, 0, 1])
        lr = np.transpose(lr, axes=[2, 0, 1])
        
        hr /= 255.0
        lr /= 255.0

        return lr, hr

    def __len__(self):
        return len(self.hr_image_files)
