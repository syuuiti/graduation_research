import argparse
import os
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from model import RCAN
import pathlib

import numpy as np
import glob

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='RCAN')
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--SR1_images_dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)#scale needs multiple of 4
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    opt = parser.parse_args()

    image_list = sorted(glob.glob(opt.image_dir + '/*'))

    model = RCAN(opt)

    state_dict = model.state_dict()
    
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model = model.to(device)
    model.eval()
    
    
    for x in range(len(image_list)):
    
        #zyunbann henkou ato no file kara filename no banngou wo tyuusyutu
        filename = os.path.basename(str(image_list[x])).split('.')[0]
        
        
        input = pil_image.open(str(image_list[x])).convert('RGB')
        input = input.resize((input.width // opt.scale, input.height // opt.scale), pil_image.BICUBIC)   
        input = input.resize((input.width *  2, input.height *  2), pil_image.BICUBIC)     
        
        input = np.expand_dims(input, 2)        
        input = transforms.ToTensor()(input)        
        input = input.unsqueeze(0).to(device)
        

        with torch.no_grad():
            pred = model(input)
            print(pred.size())

        output = pred.mul_(255.0)        
        output = output.clamp_(0.0, 255.0)
        output = output.squeeze(0)
        output = output.permute(1, 2, 0)
        output = output.byte()
        output = output.cpu()       
        output = output.numpy()
        output = np.squeeze(output)
        
        
        output = pil_image.fromarray(output, mode='RGB')
        
        output.save(os.path.join(opt.SR1_images_dir, '{}.tif'.format(filename)))        
        
        
