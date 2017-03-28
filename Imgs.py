import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import pylab

img = cv2.imread('/home/olusiak/fotoscale.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#gray = cv2.imread('/home/olusiak/water_coins.jpg',0)

#plt.imshow(gray,cmap='gray')
#plt.show()
#for i in gray.ravel():
#    print '> ',i#gray.ravel()
#plt.hist(gray.ravel(), bins=256, range=(75.0, 254.0))#, fc='k', ec='k')
#plt.show()
#i=Image.open('/home/olusiak/water_coins.jpg').convert('L')
#plt.imshow(i)
#print i
print '----'
#i.show()
#gray = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,240,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
plt.imshow(thresh)#,cmap='gray')
plt.show()
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
plt.imshow(dist_transform)
#plt.show()
ret, sure_fg = cv2.threshold(dist_transform,1.2*dist_transform.min(),255,0)

#plt.imshow(ret)
#plt.show()
plt.imshow(sure_fg)
#plt.show()
sure_fg = cv2.erode(sure_fg,kernel,iterations=3)
plt.imshow(sure_fg)
#plt.show()
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
print img

#plt.matshow(img)
#pylab.show()
plt.imshow(markers)#,cmap='gray')
plt.show()