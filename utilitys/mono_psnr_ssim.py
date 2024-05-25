import os
import cv2
from SSIM_PIL import compare_ssim
from PIL import Image
import PIL.Image as pil_image

import numpy as np

#name of folder
folder_test = 'testyougazou/haneda/genga'#genga
folder_bicubic = 'testyougazou/haneda/zounagare_s8_52'
folder_rcan = 'test0416/sr1/zounagare_haneda'
folder_rcann = 'test0416/sr2/zounagare_hanedas8'

test_filenames = os.listdir(folder_test)

sum_psnr_bicubic = 0
sum_psnr_RCAN = 0
sum_psnr_RCANN = 0

sum_ssim_bicubic = 0
sum_ssim_RCAN = 0
sum_ssim_RCANN = 0

x = 0

for i in test_filenames:
    
    x= x+1


    number = i.split('.')[0]

    test_filename = '%s.tif'%(number)
    bicubic_filename ='%s.tif'%(number)
    RCAN_filename = '%s.tif'%(number)
    RCANN_filename = '%s.tif'%(number)
    
    img_test = cv2.imread(folder_test + '/' + test_filename)
    #img_test = cv2.resize(img_test, None,fx = 4, fy = 4,interpolation=cv2.INTER_CUBIC)
    
    img_bicubic = cv2.imread(folder_bicubic + '/' + bicubic_filename)
    img_bicubic = cv2.resize(img_bicubic, None,fx = 4, fy = 4,interpolation=cv2.INTER_CUBIC)
    
    img_RCAN = cv2.imread(folder_rcan + '/' + RCAN_filename)
    img_RCAN = cv2.resize(img_RCAN, None,fx = 4, fy = 4,interpolation=cv2.INTER_CUBIC)
    
    img_RCANN = cv2.imread(folder_rcann + '/' + RCANN_filename)
    img_RCANN = cv2.resize(img_RCANN, None,fx = 2, fy = 2,interpolation=cv2.INTER_CUBIC)
    
    
    #####img_test = np.expand_dims(img_test, 0)  
    
    print(img_test.shape)
    print(img_bicubic.shape)
    print(img_RCAN.shape)
    print(img_RCANN.shape)
    
    a = cv2.PSNR(img_test, img_bicubic)
    b = cv2.PSNR(img_test, img_RCAN)
    c = cv2.PSNR(img_test, img_RCANN)
    
    
    sum_psnr_bicubic += a
    sum_psnr_RCAN += b
    sum_psnr_RCANN += c
    
    
    pilim_test = Image.open(folder_test + '/' + i)
    pilim_test = pilim_test.resize((256, 256), resample=pil_image.BICUBIC)
    
    pilim_bicubic = Image.open(folder_bicubic + '/' + bicubic_filename)
    pilim_bicubic = pilim_bicubic.resize((256, 256), resample=pil_image.BICUBIC)
    
    pilim_RCAN = Image.open(folder_rcan + '/' + RCAN_filename)   
    pilim_RCAN = pilim_RCAN.resize((256, 256), resample=pil_image.BICUBIC)
    
    pilim_RCANN = Image.open(folder_rcann + '/' + RCANN_filename)  
    pilim_RCANN = pilim_RCANN.resize((256, 256), resample=pil_image.BICUBIC)
    
    
    valueA = compare_ssim(pilim_test, pilim_bicubic)
    valueB = compare_ssim(pilim_test, pilim_RCAN)
    valueC = compare_ssim(pilim_test, pilim_RCANN)

    
    sum_ssim_bicubic += valueA
    sum_ssim_RCAN    += valueB
    sum_ssim_RCANN   += valueC
    
    avg_psnr_bicubic = sum_psnr_bicubic/x
    avg_psnr_RCAN    = sum_psnr_RCAN/x
    avg_psnr_RCANN   = sum_psnr_RCANN/x
 
    avg_ssim_bicubic = sum_ssim_bicubic/x
    avg_ssim_RCAN    = sum_ssim_RCAN/x
    avg_ssim_RCANN   = sum_ssim_RCANN/x

    print(avg_psnr_bicubic)
    print(avg_psnr_RCAN)
    print(avg_psnr_RCANN)
    
    print(avg_ssim_bicubic)
    print(avg_ssim_RCAN)
    print(avg_ssim_RCANN)
    
    print(x)
    
    
    

avg_psnr_bicubic = sum_psnr_bicubic/len(test_filenames)
avg_psnr_RCAN    = sum_psnr_RCAN   /len(test_filenames)
avg_psnr_RCANN   = sum_psnr_RCANN  /len(test_filenames)

avg_ssim_bicubic = sum_ssim_bicubic/len(test_filenames)
avg_ssim_RCAN    = sum_ssim_RCAN   /len(test_filenames)
avg_ssim_RCANN   = sum_ssim_RCANN  /len(test_filenames)

print(avg_psnr_bicubic)
print(avg_psnr_RCAN)
print(avg_psnr_RCANN)

print(avg_ssim_bicubic)
print(avg_ssim_RCAN)
print(avg_ssim_RCANN)


