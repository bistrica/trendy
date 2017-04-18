import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import colors
from PIL import Image
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image
#import statistics
#from statistics import median
import scipy.misc
from scipy.ndimage.interpolation import zoom
import pylab
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import segmentation

import scipy


img = cv2.imread('/home/olusiak/Obrazy/rois/41136_001.png-2.jpg')#data.astronaut()

img = rgb2gray(img)



ratio=8
size_orig = 1536/ratio,2048/ratio
window=(30,30)
img = cv2.imread('/home/olusiak/Obrazy/rois/41136_001.png-2.jpg')#10978_13_001.png-1.jpg')# densities/143_13_001.png')#558_13_002.png')#fotoscale.jpg')143_13_001.png')#
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#img = Image.open('/home/olusiak/Obrazy/rois/10978_13_001.png-2.jpg')
#img.thumbnail(size_orig, Image.ANTIALIAS)
#img=zoom(img, 0.25)
#img=scipy.misc.imresize(img, size_orig)
#hsv=colors.rgb_to_hsv(img)
#plt.imshow(hsv)
#plt.show()
#img=hsv




#imgM = cv2.imread('/home/olusiak/Obrazy/densities/558_13_002Markers_Counter Window -.png_density.png')
#filename='/home/olusiak/Obrazy/densities/558_13_002Markers_Counter Window -.png_density.png'
#imgM=Image.open("{0}".format(filename))
#im_arr = np.fromstring(imgM.tobytes(), dtype=np.uint8)
#im_arr = im_arr.reshape((imgM.size[1], imgM.size[0], 4))
#kernel = np.ones((3, 3), np.uint8)
#kernel2 = np.ones((9, 9), np.uint8)
#erosion = cv2.erode(im_arr, kernel, iterations=1)
#dilate = cv2.dilate(erosion, kernel2, iterations=1)
    #        plt.imshow(dilate)
    #        plt.show()
#imgM = dilate#Image.fromarray(dilate)
#grayM = cv2.cvtColor(imgM,cv2.COLOR_BGR2GRAY)#cv2.imread('/home/olusiak/water_coins.jpg',0)

#equ = cv2.equalizeHist(gray)
#gray=equ

kernel2 = np.ones((9, 9), np.uint8)
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

#plt.imshow(img)
#plt.show()




#plt.hist(gray.ravel(), bins=256, range=(0.0, 254.0))#, fc='k', ec='k')
#plt.show()


#i=Image.open('/home/olusiak/water_coins.jpg').convert('L')
#plt.imshow(i)
#print i
#print '----'
#i.show()
#gray = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
#thresh=gray
#plt.matshow(gray)
#plt.show()

#laplacian = cv2.Laplacian(img,cv2.CV_64F)
#plt.matshow(laplacian)#,cmap='gray')
#plt.show()

ret, thresh = cv2.threshold(gray,id,255,cv2.ADAPTIVE_THRESH_MEAN_C+cv2.THRESH_BINARY_INV)#+cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+



#ret, threshM = cv2.threshold(grayM,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+
#plt.imshow(threshM)
#plt.show()
#plt.imshow(thresh)
#plt.show()
#edges = cv2.Canny(thresh,50,100)
#plt.matshow(edges)
#plt.show()

#rgb=colors.hsv_to_rgb(thresh)#?
#plt.imshow(rgb)#?
#plt.show()#?

#thresh=rgb#?


#thresh=

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 5)
#print 'open'
#plt.imshow(opening)#,cmap='gray')
#plt.show()

#img=opening

#laplacian = cv2.Laplacian(img,cv2.CV_64F)
#sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(opening,cv2.CV_64F,0,1,ksize=5)

gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel2)
plt.matshow(gradient)
plt.show()

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
done = False
size = np.size(gradient)
skel = np.zeros(gradient.shape,np.uint8)
while (not done):
    print 'a'
    eroded = cv2.erode(gradient, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(gradient, temp)
    skel = cv2.bitwise_or(skel, temp)
    gradient = eroded.copy()

    zeros = size - cv2.countNonZero(gradient)
    if zeros == size:
        done = True
    plt.matshow(gradient)
    plt.show()
plt.matshow(gradient)
plt.show()
gradient=opening
#ret, markers = cv2.connectedComponents(gradient)
im2, contours, hierarchy = cv2.findContours(gradient,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(gradient, contours, 4, (150,255,150), -1)
#cv2.drawContours(gradient, gradient, contourID, COLOR, Core.FILLED);
#cv2.fillPoly(gradient, pts =gradient, color=(255,255,255))
sobely=gradient
#plt.subplot(2,1,1),plt.imshow(img,cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,2),plt.imshow(sobely,cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.imshow(sobely,cmap = 'gray')
plt.show()

def mean_shift():
    print 'MEAN SHI'
    flat_image = np.reshape(gray, [-1, 3])
    print 'fl'
    # Estimate bandwidth
    bandwidth2 = estimate_bandwidth(flat_image,
                                    quantile=.2, n_samples=500)
    print 'est'
    ms = MeanShift(bandwidth2, bin_seeding=True)
    print 'ms'
    ms.fit(flat_image)
    print 'ms f'
    labels = ms.labels_

    # Plot image vs segmented image
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.imshow(gray)
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(np.reshape(labels, [851, 1280]))
    plt.axis('off')
    plt.show()


#plt.imshow(opening)#,cmap='gray')
#plt.show()
def hsv():
    print 'PEI'
    piet_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # threshold for hue channel in blue range
    blue_min = np.array([100, 50, 50], np.uint8)
    blue_max = np.array([140, 255, 255], np.uint8)
    threshold_blue_img = cv2.inRange(piet_hsv, blue_min, blue_max)
    #plt.imshow(threshold_blue_img)
    #plt.show()
    #threshold_blue_img = cv2.cvtColor(threshold_blue_img, cv2.COLOR_GRAY2RGB)
    opening2 = cv2.morphologyEx(threshold_blue_img,cv2.MORPH_OPEN,kernel, iterations = 5)
#plt.imshow(opening2)
#plt.show()

def hsv_hist():
    #mean_shift()
    print 'hsv'
    im_hsv = colors.rgb_to_hsv(img[...,:3])
    # pull out just the s channel
    lu=im_hsv[...,1].flatten()
    lu=lu.tolist()
    print 'lu ',lu
    i=list()
    for ii in lu:
        if ii!=0:
            i.append(ii)
    lu.remove(0.)
    lu=np.asarray(i)
    print 'lu ',lu
    plt.hist(lu,250)
    #plt.show()
# sure background area


sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)#OPENING
#plt.imshow(dist_transform)
#plt.show()

#em=cv2.EM(2)
#em = cv2.ml.EM_create()
#em.train(img,0,2)
#means = em.getMat('means')
#opening = cv2.morphologyEx(dist_transform,cv2.MORPH_OPEN,kernel, iterations = 5)
#print 'open2'
#plt.imshow(means)#opening)#,cmap='gray')
#plt.show()

ret, sure_fg = cv2.threshold(dist_transform,1.2*dist_transform.min(),255,0)

#fg=cv2.cvtColor(sure_fg,cv2.COLOR_BGR2GRAY)
off=0
off2=0

#v=400
ran=2
ox=off
oy=off2
pix=sure_fg
#size=(size_orig[0]/ran,size_orig[1]/ran)
size=(size_orig[1]/ran,size_orig[0]/ran)

new_pic=list()#np.array(int)
pics=list()
for i in range(sure_fg.size):
    new_pic.append(255.0)
new_pic=np.asarray(new_pic,float)
new_pic=np.reshape(new_pic,sure_fg.shape)
cz=0
sett=set()
maxes=list()

for i in range(ran):
    try:
        test = ox + size[1] * i
        test = pix[test, 0]
        #test = ox + size[1] * (i + 1)
        #test = pix[test, 0]
    except:
        print 'cont'
        continue
    for j in range(ran):  # i, ran, 1):
        all_pixels = list()  # np.array(int)

        try:

            test = oy + size[0] * j
            test = pix[0, test]
            #test = oy + size[0] * (j + 1)
            #test = pix[0, test]
        except:
            print 'cont3'
            continue
        l = list()

        for x in range(ox + size[1] * i, ox + size[1] * (i + 1), 1):

            for y in range(oy+size[0] * j, oy+size[0] * (j + 1), 1):

                l.append(sure_fg[x][y])
        l = np.asarray(l)
        #print 'll ', l.shape, l.size
        l = np.fromstring(l.tobytes(), dtype=np.uint8)
        l = np.reshape(l, (size[1], size[0], 4))
        l = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)

        l = cv2.distanceTransform(l, cv2.DIST_L2, 5)  #
        l = np.asarray(l,float)

        print 'max ', l.max()
        max=l.max()
        if l.max() != 0:
            maxes.append(l.max())

        ret, le = cv2.threshold(l, 0.8 * l.max(), 255, cv2.ADAPTIVE_THRESH_MEAN_C + cv2.THRESH_BINARY)  # _INV)

        ####
        if False:
            plt.matshow(le)
            plt.show()
            for x1 in range(le.shape[0]):
                for y1 in range(le.shape[1]):
                    if le[x1,y1]==255:
                        print '? ',x1,y1
                        if True:
                            s = np.linspace(0, 2 * np.pi, 400)
                            x = x1 + 20 * np.cos(s)
                            y = y1 + 20 * np.sin(s)
                            init = np.array([x, y]).T

                            snake = active_contour(gaussian(le, 3),
                                                   init, alpha=0.015, beta=10, gamma=0.001)

                            fig = plt.figure(figsize=(7, 7))
                            ax = fig.add_subplot(111)
                            plt.gray()
                            ax.imshow(le)
                            ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
                            ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
                            ax.set_xticks([]), ax.set_yticks([])
                            ax.axis([0, img.shape[1], img.shape[0], 0])
                            plt.show()

        ###



        #print len(l)
        pics.append(l)
   #     plt.matshow(l)
   #     plt.show()
        #l=np.asarray(l)
        #l=le
        l = np.reshape(l, l.size)
        #f=l.tolist()
        #print f
        ind=0

        if cz!=-1:
            for x in range(ox + size[1] * i, ox + size[1] * (i + 1), 1):

                for y in range(oy+size[0] * j, oy+size[0] * (j + 1), 1):
                    #if (x,y) in sett:
                        #print '!!!',x,y
                        #h=9/0
                    sett.add((x,y))
                    #print 'x y ',x,y
                    #print 'L ',l[ind]
                    #l=np.asarray(l)

                    new_pic[x,y]=l[ind]
                    #if ind>729476 and ind<735370:
                        #print 'N ',ind,' ',new_pic[x,y]
                    ind+=1
        cz+=1
        print ox + size[1] * i, ox + size[1] * (i + 1),' / ', oy+size[0] * j, oy+size[0] * (j + 1)
        ind=0
#        for x in range(ox + size[1] * i, ox + size[1] * (i + 1), 1):

#            for y in range(oy + size[0] * j, oy + size[0] * (j + 1), 1):
                #if ind>729476 and ind<735370:
                    #print 'NN ',ind,' ',new_pic[x,y]
 #               ind+=1
        #new_pic=cv2.cvtColor(new_pic, cv2.COLOR_GRAY2BGR)
        #plt.matshow(new_pic)
        #plt.show()
        #print 'll2 ', l.size

        # l=cv2.cvtColor(l,cv2.COLOR_BGR2GRAY)

        # print 'llg ',l.type

#plt.matshow(new_pic)
#plt.show()

#c=9/0




#print 'nm ',new_pic.size, len(maxes)
#ret, new_pic2 = cv2.threshold(new_pic,0.6*maxes[len(maxes)/2],255,0)
#plt.matshow(new_pic2)
#plt.show()
#x=0
#c=0
#print 'sett ',sett
#for n in new_pic:
#    for ni in n:
#        #print 'N ',n#len(n)
#        if ni is None or ni==0:
#            c+=1

            #print 'X None ',x
#        x+=1
#print 'no: ',x, '[',c
#plt.matshow(new_pic)

#plt.show()

#for i in pics:
#    plt.matshow(i)
#    plt.show()
#v=400
#for i in range(0,v,1):
#    for j in range(0,v,1):
#        #print ': ',sure_fg[off+i][off2+j]
#        l.append(sure_fg[off+i][off2+j])
#print '::: ',len(l)
#l=np.asarray(l)
#print 'll ',l.shape, l.size
#l=np.fromstring(l.tobytes(), dtype=np.uint8)
#l=np.reshape(l,(v,v,4))
#l=cv2.cvtColor(l,cv2.COLOR_BGR2GRAY)
#print 'll2 ',l.size

#l=cv2.cvtColor(l,cv2.COLOR_BGR2GRAY)

#print 'llg ',l.type
#l=cv2.distanceTransform(l,cv2.DIST_L2,5)#

#plt.matshow(l)
#plt.show()

#print 'ret'
#plt.imshow(ret)
#plt.show()

#plt.imshow(sure_fg)
#plt.show()

#print 'sure'
#sure_fg=opening2
sure_fg = cv2.erode(sure_fg,kernel,iterations=3)#3

#sure_fg=np.uint8(new_pic)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
#print 's fg'
#plt.imshow(sure_fg)
#plt.show()

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
#print 'mark'
plt.imshow(markers)
plt.show()
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

plt.imshow(img)#,cmap='gray')
plt.show()
print 'fx'
fx=segmentation.random_walker(img, markers)
plt.matshow(fx)
plt.show()
#markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
print img

#plt.matshow(img)
#pylab.show()


plt.imshow(markers)#,cmap='gray')
plt.show()
#markers= cv2.cvtColor(markers,cv2.COLOR_BGR2GRAY)
#circles = cv2.HoughCircles(markers,cv2.HOUGH_GRADIENT,1,20,
       #                     param1=50,param2=30,minRadius=0,maxRadius=0)
#circles = np.uint16(np.around(markers))
#for i in circles[0,:]:
#    print' i',i
    # draw the outer circle
#    cv2.circle(markers,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
#    cv2.circle(markers,(i[0],i[1]),2,(0,0,255),3)
#plt.imshow(markers)#,cmap='gray')
#plt.show()

markers=sobely
circles = cv2.HoughCircles(markers,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(markers))
for i in circles[0,:]:
    cv2.circle(markers,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(markers,(i[0],i[1]),2,(0,0,255),3)
#plt.imshow(markers)#,cmap='gray')
#plt.show()

plt.matshow(markers)
plt.show()