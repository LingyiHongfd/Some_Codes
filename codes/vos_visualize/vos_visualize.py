from skimage import io
import skimage
from skimage import morphology
import numpy as np
from PIL import Image
import cv2
from tqdm import *
import os
import argparse

"""
Visualize sequence annotations.
"""


parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str,help="path to the jpg")
parser.add_argument("--msk_path", type=str,help="path to the mask")
parser.add_argument("--save_path",type=str,help="path to save output")
parser.add_argument("--viz", default='show',type=str,help="visualize action: save or show or all ")
parser.add_argument("--mode", default='val',type=str,help="mode: train or val or test")
parser.add_argument("--year", default='davis2017',type=str,help="year:davis2016 or davis2017 or ytb")
parser.add_argument("--single_obj", default=False,type=bool,help="single object or multi objects")
parser.add_argument("--default_palette", type=bool,help="whether to use the palette of the mask. True using the mask default palette, False using customized palette file")

args = parser.parse_args()


mode=args.mode
year=args.year
viz=args.viz
default_palette=args.default_palette
_palette=None
data_root_path=args.img_path
teq_root_path=args.msk_path
save_path=args.save_path

#mode='val'  ## train or val
#year='davis2017'  ## davis2016 or davis2017 or ytb
#viz='show'        ## show or save or all
#default_palette=False  ## True using mask deafult palette ,False using customized palette
#data_root_path=r'E:\Dataset\DAVIS\DAVIS\DAVIS'
#teq_root_path=r'F:\MyCFBI\test\test\Annotations\480p'
#teq_root_path=r'F:\MyCFBI\full_train_c\youtubevos_resnet101_cfbi_davis_ckpt_10000\Annotations'
#save_root_path=r'C:\Users\hongl\Desktop\colour_all\davis2016'

def read_palette():
    txt_path='./palette.txt'
    with open(txt_path) as f:
        _palette=f.readlines()
    _palette_len=len(_palette)
    for i in range (_palette_len):
        _palette[i]=list(map(int,_palette[i].strip().split()))
    _palette=np.array(_palette)
    return _palette

def get_video_seq(data_root_path,mode,year):
    if year=='davis2016':
        single_obj=True
    else:
        single_obj=False
    
    if year=='davis2016':
        txt_path=os.path.join(data_root_path,'ImageSets','2016',mode+'.txt')
        with open(txt_path) as f:
            video_names=f.readlines()
    if year=='davis2017':
        txt_path=os.path.join(data_root_path,'ImageSets','2017',mode+'.txt')
        with open(txt_path) as f:
            video_names=f.readlines()
    if year=='ytb':
        if mode=='val':
            pmode='valid'
        else:
            pmode=mode
        tmp_pth=os.path.join(data_root_path,pmode,'JPEGImages')
        video_names=os.listdir(tmp_pth)

    return video_names,single_obj


def overlay_video(video_name,date_root_path,teq_root_path,save_root_path,year,mode,single_obj,_palette,viz):
    video_name=video_name.strip()
    if year=='davis2016' or year=='davis2017':
        img_root_path=os.path.join(date_root_path,'JPEGImages','480p',video_name)
    else:
        if mode=='val':
            pmode='valid'
        else:
            pmode=mode
        img_root_path=os.path.join(date_root_path,pmode,'JPEGImages',video_name)

    msk_root_path=os.path.join(teq_root_path,video_name)
    msk_paths=os.listdir(msk_root_path)
    save_path=os.path.join(save_root_path,video_name)
    if os.path.isdir(save_path)==False:
        os.mkdir(save_path)

    msk_paths_len=len(msk_paths)
    for i in range (msk_paths_len):
        mask_str=msk_paths[i]
        mask_str_len=len(mask_str)
        img_str=mask_str[0:mask_str_len-4]+'.jpg'
        mask_path=os.path.join(msk_root_path,mask_str)
        img_path=os.path.join(img_root_path,img_str)
        save_img_path=os.path.join(save_path,img_str)
        rimg=overlay_frame(mask_path,img_path,single_obj,_palette)
        if viz=='show':
            img_show(rimg)
        if viz=='save':
            img_save(rimg,save_img_path)
        if viz=='all':
            img_show(rimg)
            img_save(rimg,save_img_path)

def overlay_frame(mask_path,img_path,single_obj,_palette):
    img=np.asarray(Image.open(img_path))
    mask=np.asarray(Image.open(mask_path))

    if single_obj==True:
        mask=(mask>=1)
        mask=mask.astype(int)

    if default_palette ==True:
        pimg=Image.open(mask_path)
        palette=pimg.getpalette()
        palette=np.array(palette).reshape(-1,3)
    else:
        palette=_palette

    cscale=2
    alpha=0.4
    colors = np.atleast_2d(palette) * cscale

    im_overlay    = img.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask

        foreground  = img*alpha + np.ones(img.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        countours = morphology.binary.binary_dilation(binary_mask) ^ binary_mask
        im_overlay[countours,:] = 0
    rimg=im_overlay.astype(img.dtype)
    return rimg[...,[2,1,0]]
            
def img_show(rimg):
    cv2.imshow("Sequence",rimg)
    cv2.waitKey()

def img_save(rimg,save_path):
    cv2.imwrite(save_path,rimg)
    





video_seq,single_obj =get_video_seq(data_root_path,mode,year)
single_obj=args.single_obj
video_seq_len=len(video_seq)
if default_palette!=True:
    _palette=read_palette()

for i in tqdm(range (video_seq_len),ascii=True,):
    video_name=video_seq[i]
    overlay_video(video_name,data_root_path,teq_root_path,save_root_path,year,mode,single_obj,_palette,viz)




