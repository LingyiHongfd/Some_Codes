import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import *
import os


'''
combine mask and image, and add contour to the mask

auto save

the colourmap can be changed to get better visual effect, default is cv2.COLORMAP_AUTUMN, whose effect is similar to the quality result
in some paper

'''



video_name='dogs-jump'

test_img_list_root_path=r'E:\Dataset\DAVIS\DAVIS\DAVIS\JPEGImages\480p'
test_mask_list_root_path=r'F:\MyCFBI\full_train\davis2017_resnet101_cfbi_davis_ckpt_30000_2_sc_u+f_21\Annotations\480p'
save_list_root_path=r'C:\Users\hongl\Desktop\result\colour'

test_img_list_path=os.path.join(test_img_list_root_path,video_name)
test_mask_list_path=os.path.join(test_mask_list_root_path,video_name)
save_list_path=os.path.join(save_list_root_path,video_name)

if os.path.isdir(save_list_path)==False:
    os.mkdir(save_list_path)


jpg_dir_path=os.listdir(test_img_list_path)
png_dir_path=os.listdir(test_mask_list_path)
dir_len=len(jpg_dir_path)

for i in range(dir_len):
    test_img_path=os.path.join(test_img_list_path,jpg_dir_path[i])
    test_mask_path=os.path.join(test_mask_list_path,png_dir_path[i])
    print (test_img_path)
    img = cv2.imread(test_img_path)
    m=cv2.imread(test_mask_path)
    mask = cv2.imread(test_mask_path)

    

    mc=m
    contour_mask = cv2.imread(test_mask_path,0)
    t=(contour_mask>0)
    t=t.astype(np.int8)
    t=t[:,:,np.newaxis].repeat(3,axis=2)
    tp=1-t

    # cv2.COLORMAP_RAINBOW
    # cv2.COLORMAP_AUTUMN  default
    # cv2.COLORMAP_JET
    # cv2.COLORMAP_WINTER
    # cv2.COLORMAP_HSV
    m = cv2.resize(m,(854,480))
    m = np.uint8(255 * m)
    m = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    m=tp*mc+t*m



    superimposed_img=img+m


    contours, _ = cv2.findContours(contour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(superimposed_img, contours, -1, (0, 0, 0), 1)
    save_path=os.path.join(save_list_path,png_dir_path[i])
    #superimposed_img = cv2.addWeighted(img,1,m,1,0)
    cv2.imwrite(save_path, superimposed_img)

'''
test_img_path=r'E:\Dataset\DAVIS\DAVIS\DAVIS\JPEGImages\480p\drift-straight\00021.jpg'
test_mask_path=r'C:\Users\hongl\Desktop\result\without_optim_000021.png'

img = cv2.imread(test_img_path)
m=cv2.imread(test_mask_path)
mask = cv2.imread(test_mask_path)



mc=m
contour_mask = cv2.imread(test_mask_path,0)
t=(contour_mask>0)
t=t.astype(np.int8)
t=t[:,:,np.newaxis].repeat(3,axis=2)
tp=1-t

    # cv2.COLORMAP_RAINBOW
    # cv2.COLORMAP_AUTUMN  default
    # cv2.COLORMAP_JET
    # cv2.COLORMAP_WINTER
    # cv2.COLORMAP_HSV
m = cv2.resize(m,(854,480))
m = np.uint8(255 * m)
m = cv2.applyColorMap(m, cv2.COLORMAP_AUTUMN)
m=tp*mc+t*m



superimposed_img=img+m


contours, _ = cv2.findContours(contour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(superimposed_img, contours, -1, (0, 0, 0), 1)
#save_path=os.path.join(save_list_path,png_dir_path[i])
#superimposed_img = cv2.addWeighted(img,1,m,1,0)
save_path=r'C:\Users\hongl\Desktop\result\without_optim_00021_img.png'
cv2.imwrite(save_path, superimposed_img)


'''