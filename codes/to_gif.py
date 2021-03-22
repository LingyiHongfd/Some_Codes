from PIL import Image 
from images2gif import writeGif
import numpy as np
import os
import imageio
outfilename = r"C:\Users\hongl\Desktop\Paper\picture\my.gif"       
filenames = []        
root_path=r'E:\Dataset\DAVIS\DAVIS\DAVIS\Show\480p\parkour'
img_list=os.listdir(root_path)
print (img_list)
for i in range(len(img_list)):  
    filename = os.path.join(root_path,img_list[i])    
    filenames.append(filename)             
frames = []
for file_name in filenames:
    frames.append(imageio.imread(file_name))
    # Save them as frames into a gif
imageio.mimsave(outfilename, frames, 'GIF', duration = (1/24))


'''
for image_name in filenames:                
    im = Image.open(image_name)             
    im = im.convert("RGB")                 
    im = np.array(im)                     
    frames.append(im)                      
writeGif(outfilename.encode(encoding='UTF-8'), frames, duration=0.1, subRectangles=False) 
'''




