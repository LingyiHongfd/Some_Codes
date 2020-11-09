import torch
import numpy as np
from visdom import Visdom


'''
create gaussian kernel
'''



# using sactter or index_fill
viz=Visdom()
shape =(3,3)
sigma=1
m, n = [(ss - 1.) / 2. for ss in shape]
y, x = np.ogrid[-m:m + 1, -n:n + 1]

h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
h[h < np.finfo(h.dtype).eps * h.max()] = 0
gaussian=torch.from_numpy(h)
#gaussian=h

heatmap=torch.zeros((1,5,5))
height,width=5,5
x,y=1,2

radius=3

heatmap=torch.zeros((1,1,30,54))
heatmap[0,0,15,27]=1
viz.image(heatmap.squeeze(0).squeeze(0))
heatmap=torch.nn.functional.interpolate(heatmap,size=(480,854),mode='bilinear')
print ('heatmap',torch.max(heatmap))
viz.image(heatmap.squeeze(0).squeeze(0))

'''
print (x,radius)
left= min(x, radius)
right=min(width - x, radius + 1)
top, bottom = min(y, radius),min(height - y, radius + 1)

masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
masked_gaussian = gaussian[radius - top:radius + bottom, 
                               radius - left:radius + right]
'''
'''
X=torch.arange(x-1,x+2)
Y=torch.arange(y-1,y+2)
print (X,Y)
XY=torch.cat((X.unsqueeze(0),Y.unsqueeze(0)),dim=0).unsqueeze(0)
print (XY)
heatmap.scatter_(0,XY,gaussian)
print (heatmap)
'''

