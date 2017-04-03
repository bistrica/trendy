import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import colors
from PIL import Image

import pylab

img = cv2.imread('/home/olusiak/Obrazy/rois/10978_13_001.png-2.jpg')#41136_001.png-2.jpg')# densities/143_13_001.png')#558_13_002.png')#fotoscale.jpg')143_13_001.png')#
#hsv=colors.rgb_to_hsv(img)
#plt.imshow(hsv)
#plt.show()
#img=hsv

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#imgM = cv2.imread('/home/olusiak/Obrazy/densities/558_13_002Markers_Counter Window -.png_density.png')
filename='/home/olusiak/Obrazy/densities/558_13_002Markers_Counter Window -.png_density.png'
imgM=Image.open("{0}".format(filename))
im_arr = np.fromstring(imgM.tobytes(), dtype=np.uint8)
im_arr = im_arr.reshape((imgM.size[1], imgM.size[0], 4))
kernel = np.ones((3, 3), np.uint8)
kernel2 = np.ones((9, 9), np.uint8)
erosion = cv2.erode(im_arr, kernel, iterations=1)
dilate = cv2.dilate(erosion, kernel2, iterations=1)
    #        plt.imshow(dilate)
    #        plt.show()
imgM = dilate#Image.fromarray(dilate)
grayM = cv2.cvtColor(imgM,cv2.COLOR_BGR2GRAY)#cv2.imread('/home/olusiak/water_coins.jpg',0)

equ = cv2.equalizeHist(gray)
#gray=equ


data=dict()
for i in gray.ravel():
    if data.has_key(i):
        data[i]+=1
    else:
        data[i]=1
#if data.has_key(255):
max=0
id=None
for k in data.keys():
    #print 'k ',k
    if data[k]>=max and k<240:#50:#240:
        id=k
        max=data[k]
print 'key: ',id,' - ',max
id-=15

plt.imshow(img)
plt.show()




plt.hist(gray.ravel(), bins=256, range=(0.0, 254.0))#, fc='k', ec='k')
plt.show()


#i=Image.open('/home/olusiak/water_coins.jpg').convert('L')
#plt.imshow(i)
#print i
print '----'
#i.show()
#gray = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
#thresh=gray
ret, thresh = cv2.threshold(gray,id,255,cv2.ADAPTIVE_THRESH_MEAN_C+cv2.THRESH_BINARY_INV)#+cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+

#ret, threshM = cv2.threshold(grayM,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+
#plt.imshow(threshM)
#plt.show()
plt.imshow(thresh)
plt.show()
#edges = cv2.Canny(thresh,50,100)
#plt.matshow(edges)
#plt.show()

#circles = cv2.HoughCircles(thresh,cv2.HOUGH_GRADIENT,1,20,
#                            param1=50,param2=30,minRadius=0,maxRadius=0)

#circles = np.uint16(np.around(circles))
#for i in circles[0,:]:
#    print' i',i
    # draw the outer circle
#    cv2.circle(thresh,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
#    cv2.circle(thresh,(i[0],i[1]),2,(0,0,255),3)

#plt.imshow(thresh)#,cmap='gray')
#plt.show()
#rgb=colors.hsv_to_rgb(thresh)#?
#plt.imshow(rgb)#?
#plt.show()#?

#thresh=rgb#?


#thresh=

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 5)
print 'open'
plt.imshow(opening)#,cmap='gray')
plt.show()

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)#OPENING
plt.imshow(dist_transform)
plt.show()

#em=cv2.EM(2)
#em = cv2.ml.EM_create()
#em.train(img,0,2)
#means = em.getMat('means')
#opening = cv2.morphologyEx(dist_transform,cv2.MORPH_OPEN,kernel, iterations = 5)
#print 'open2'
#plt.imshow(means)#opening)#,cmap='gray')
#plt.show()

ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)


print 'ret'
#plt.imshow(ret)
#plt.show()
plt.imshow(sure_fg)
plt.show()
sure_fg = cv2.erode(sure_fg,kernel,iterations=3)#3

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
print 's fg'
plt.imshow(sure_fg)
plt.show()

#dist_transform = cv2.distanceTransform(sure_fg,cv2.DIST_L2,5)#OPENING
#plt.imshow(dist_transform)
#plt.show()
#print 'unk'
#plt.imshow(unknown)#sure_fg)
#plt.show()
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

plt.imshow(markers)
plt.show()
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

plt.imshow(img)#,cmap='gray')
plt.show()
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
print img

#plt.matshow(img)
#pylab.show()
plt.imshow(markers)#,cmap='gray')
plt.show()