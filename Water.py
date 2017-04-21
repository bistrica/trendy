import numpy as np
import cv2
from skimage import data, segmentation, filters, color
from random import shuffle
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


def process_water(path):
    #img = cv2.imread(path)

    #img = rgb2gray(img)



    ratio=8

    img = cv2.imread(path)#10978_13_001.png-1.jpg')# densities/143_13_001.png')#558_13_002.png')#fotoscale.jpg')143_13_001.png')#
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

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
        if data[k]>=max and k<240 and k>130:#50:#240:
            id=k
            max=data[k]
    print 'key: ',id,' - ',max
    id-=15

    plt.matshow(gray,cmap='gray')
    plt.show()


    ret, thresh = cv2.threshold(gray,id,255,cv2.ADAPTIVE_THRESH_MEAN_C+cv2.THRESH_BINARY_INV)#+cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+

    plt.matshow(thresh,cmap='gray')
    plt.show()

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 5)

    plt.matshow(opening,cmap='gray')
    plt.show()


    #plt.imshow(sobely,cmap = 'gray')
    #plt.show()

    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)#OPENING

    plt.matshow(dist_transform)
    plt.show()
    ret, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)

    sure_fg = cv2.erode(sure_fg,kernel,iterations=3)#3
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    plt.matshow(markers)
    plt.show()
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    #print 'mark'
    #plt.imshow(markers)
    #plt.show()
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    #plt.imshow(img)#,cmap='gray')
    #plt.show()
    #print 'fx'
    #fx=segmentation.random_walker(img, markers)
    #markers = segmentation.quickshift(img)#, markers)

    #fx = color.label2rgb(fx, img, kind='avg')
    #plt.imshow(fx)
    #plt.show()
    markers = cv2.watershed(img,markers)

    img[markers == -1] = [255,0,0]
    plt.imshow(markers)
    plt.show()
    im = Image.fromarray(markers)
    #plt.imshow(im)
    #plt.show()
    return im
    #print img



def process_quick(path):
        # img = cv2.imread(path)

        # img = rgb2gray(img)

    if True:

        #ratio = 8

        img = cv2.imread(
            path)  # 10978_13_001.png-1.jpg')# densities/143_13_001.png')#558_13_002.png')#fotoscale.jpg')143_13_001.png')#
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel2 = np.ones((9, 9), np.uint8)
        data = dict()
        for i in gray.ravel():
            if data.has_key(i):
                data[i] += 1
            else:
                data[i] = 1
        # if data.has_key(255):
        max = 0
        id = None
        for k in data.keys():
            # print 'k ',k
            if data[k] >= max and k < 240 and k>130:  # 50:#240:
                id = k
                max = data[k]
        print 'key: ', id, ' - ', max
        id -= 15

        ret, thresh = cv2.threshold(gray, id, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C + cv2.THRESH_BINARY_INV)  # +cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)

        for i in range(opening.shape[0]):
            for j in range(opening.shape[1]):
                if opening[i][j] == 0:
                    img[i][j] = (0, 0, 0)

        #plt.imshow(img)
        #plt.show()

        fx = segmentation.quickshift(img)  # , markers)
        plt.imshow(fx)
        plt.show()
        ss = set()
        for i in range(fx.shape[0]):
            for j in range(fx.shape[1]):
               if isinstance(fx[i][j],int):
                   ss.add(fx[i][j])
               else:
                   ss.add((fx[i][j][0],fx[i][j][1]  ,fx[i][j][2]))
        print 'ssle ',len(ss)
        cols=list()
        for r in range(0,256,1):
            for r2 in range(0, 256, 1):
                for r3 in range(0, 256, 1):
                    cols.append((r,r2,r3))
        print 'cols',len(cols)
        shuffle(cols)

        img2 = np.zeros_like(img)
        img2[:, :, 0] = opening
        img2[:, :, 1] = opening
        img2[:, :, 2] = opening
        fx = color.label2rgb(fx, img2, colors=cols, kind='overlay')
        for i in range(fx.shape[0]):
            for j in range(fx.shape[1]):
                if opening[i][j]==0:
                    fx[i][j]=(0,0,0)
#                ss.add((fx[i][j][0], fx[i][j][1], fx[i][j][2]))
        plt.imshow(fx)
        plt.show()
        ss = set()
        for i in range(fx.shape[0]):
            for j in range(fx.shape[1]):
                ss.add((fx[i][j][0], fx[i][j][1], fx[i][j][2]))
        print 'ssle22 ', len(ss)

        #labels = np.unique(fx, return_inverse=True)[1]
        #print 'labels',labels,len(labels),fx.shape
        #labels=np.reshape(labels,fx.shape)

        #plt.imshow(labels)
        #plt.show()
        #fx=labels
        #fx = np.uint8(fx)
        #for l in labels:
        #    print 'labe; ',l

        #plt.imshow(fx)#segmentation.mark_boundaries(img, fx))
        #plt.show()

        #fx = segmentation.felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
        #plt.imshow(fx)
        #plt.show()



        plt.imshow(fx)
        plt.show()
        # markers = cv2.watershed(img,markers)

        #img[fx == -1] = [255, 0, 0]
        #plt.imshow(fx)
        #plt.show()
        fx = np.uint8(fx)
        im = Image.fromarray(fx)
        # plt.imshow(im)
        # plt.show()
        return im
        # print img




def process_felzen(path):
    if True:
        if True:
            # img = cv2.imread(path)

            # img = rgb2gray(img)

            if True:

                # ratio = 8

                img = cv2.imread(
                    path)  # 10978_13_001.png-1.jpg')# densities/143_13_001.png')#558_13_002.png')#fotoscale.jpg')143_13_001.png')#
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                kernel2 = np.ones((9, 9), np.uint8)
                data = dict()
                for i in gray.ravel():
                    if data.has_key(i):
                        data[i] += 1
                    else:
                        data[i] = 1
                # if data.has_key(255):
                max = 0
                id = None
                for k in data.keys():
                    # print 'k ',k
                    if data[k] >= max and k < 240 and k > 130:  # 50:#240:
                        id = k
                        max = data[k]
                print 'key: ', id, ' - ', max
                id -= 15

                ret, thresh = cv2.threshold(gray, id, 255,
                                            cv2.ADAPTIVE_THRESH_MEAN_C + cv2.THRESH_BINARY_INV)  # +cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+

                kernel = np.ones((3, 3), np.uint8)
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)

                for i in range(opening.shape[0]):
                    for j in range(opening.shape[1]):
                        if opening[i][j] == 0:
                            img[i][j] = (0, 0, 0)

                # plt.imshow(img)
                # plt.show()

                fx = segmentation.felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
                #fx = segmentation.quickshift(img)  # , markers)
                #fx = color.label2rgb(fx, img, kind='avg')

                # plt.imshow(fx)#segmentation.mark_boundaries(img, fx))
                # plt.show()

                # fx = segmentation.felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
                # plt.imshow(fx)
                # plt.show()



                # plt.imshow(fx)
                # plt.show()
                # markers = cv2.watershed(img,markers)

                # img[fx == -1] = [255, 0, 0]
                # plt.imshow(fx)
                # plt.show()
                im = Image.fromarray(fx)
                # plt.imshow(im)
                # plt.show()
                return im
                # print img



                # plt.matshow(markers)
        # plt.show()

#plt.matshow(markers)
#plt.show()

###########

#segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
#ax[0, 0].imshow(mark_boundaries(img, segments_fz))