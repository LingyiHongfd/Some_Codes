import numpy as np
import cv2
from matplotlib import pyplot as plt


'''
using matplotlib to convert heatmap whose value rangs from 0 to 1 to colour heatmap

and remove the border and axis

'''


img = cv2.imread(r'C:\Users\hongl\Desktop\result\result1\80.png')


c=img[:,:,0]
c=c/255

fig = plt.gcf()

plt.imshow(c)
plt.xticks([])
plt.yticks([])
#plt.gca().xaxis.set_major_locator(plt.NullLocator())
#plt.gca().yaxis.set_major_locator(plt.NullLocator())
fig.set_size_inches(854/10,480/10)#输出width*height像素
plt.axis('off')
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)

plt.savefig(r'C:\Users\hongl\Desktop\result\s.png',pad_inches = 0)
#plt.show()


