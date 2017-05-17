
import cv2
from skimage import data, segmentation, filters, color
from random import shuffle

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

from skimage import segmentation

import scipy


def process_water(path,THRESH,ADAP,DIST_PERC):

    img = cv2.imread(path)#10978_13_001.png-1.jpg')# densities/143_13_001.png')#558_13_002.png')#fotoscale.jpg')143_13_001.png')#
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel2 = np.ones((9, 9), np.uint8)



    #plt.matshow(gray,cmap='gray')
    #plt.show()


    if ADAP:
        thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 20)
    else:
        data = dict()
        for i in gray.ravel():
            if data.has_key(i):
                data[i] += 1
            else:
                data[i] = 1

        max = 0
        id = None
        for k in data.keys():
            # print 'k ',k
            if data[k] >= max and k < 240 and k > 130:  # 50:#240:
                id = k
                max = data[k]
        print 'key: ', id, ' - ', max
        id -= THRESH  # 15
        ret, thresh = cv2.threshold(gray,id,255,cv2.ADAPTIVE_THRESH_MEAN_C+cv2.THRESH_BINARY_INV)#+cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+


    #plt.matshow(thresh,cmap='gray')
    #plt.show()

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 5)

    #plt.matshow(opening,cmap='gray')
    #plt.show()


    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)#OPENING

    #plt.matshow(dist_transform)
    #plt.show()
    #print dist_transform.min(), dist_transform.max()
    ret, sure_fg = cv2.threshold(dist_transform,int(dist_transform.max()*DIST_PERC),255,0)

    sure_fg = cv2.erode(sure_fg,kernel,iterations=3)#3
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    #plt.matshow(markers)
    #plt.show()
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    markers[unknown==255] = 0


    markers = cv2.watershed(img,markers)

    img[markers == -1] = [0,0,0]
    #plt.imshow(markers)
    #plt.show()



    ss = set()
    for e in range(len(markers)):
        for ej in range(len(markers[0])):

            if markers[e][ej]==-1 or markers[e][ej]==1:
                markers[e][ej]=0
            #if  markers[e][ej]==0:
            #    img2[e][ej]=(0,0,0)
            ss.add(markers[e][ej])
    #cols=list()

    img2 = np.zeros_like(img)
    img2[:, :, 0] = opening
    img2[:, :, 1] = opening
    img2[:, :, 2] = opening
    c=1
    dic=dict()
    dic[0]=(0,0,0)
    for r in range(1,255,1):
        for r2 in range(1, 255, 1):
            for r3 in range(1, 255, 1):
                #cols.add((r,r2,r3))
                dic[c]=(r,r2,r3)
                if c==len(ss)+5:
                    break
                c+=1
            if c==len(ss)+5:
                break
        if c == len(ss)+5:
            break
    for e in range(len(markers)):
        for ej in range(len(markers[0])):
            img2[e][ej]=dic[markers[e][ej]]

    #plt.imshow(markers)
    #plt.show()



    markers = np.uint8(img2)
    im = Image.fromarray(markers)
    #print im
    #print 'mar ', len(ss)


    return im




def process_quick(path):

    if True:
        im = Image.open(path)
        im.thumbnail((im.size[0]/4,im.size[1]/4), Image.ANTIALIAS)
        im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
        print 'im ',im_arr.size, im.size
        im_arr = im_arr.reshape((im.size[1], im.size[0],im_arr.size/(im.size[1]*im.size[0])))
        img=im_arr
 #       plt.imshow(img)
 #       plt.show()

        #img = cv2.imread(
        #    path)  # 10978_13_001.png-1.jpg')# densities/143_13_001.png')#558_13_002.png')#fotoscale.jpg')143_13_001.png')#
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel2 = np.ones((9, 9), np.uint8)
        data = dict()
        for i in gray.ravel():
            if data.has_key(i):
                data[i] += 1
            else:
                data[i] = 1

        max = 0
        id = None
        for k in data.keys():
            # print 'k ',k
            if data[k] >= max and k < 240 and k>130:  # 50:#240:
                id = k
                max = data[k]
        #print 'key: ', id, ' - ', max
        id -= 30

        ret, thresh = cv2.threshold(gray, id, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C + cv2.THRESH_BINARY_INV)  # +cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        for i in range(opening.shape[0]):
            for j in range(opening.shape[1]):
                if opening[i][j] == 0:
                    img[i][j] = (0, 0, 0)

        #plt.imshow(img)
        #plt.show()

        fx = segmentation.quickshift(img,kernel_size=3)  # , markers)
        #plt.imshow(fx)
        #plt.show()
        ss = set()
        for i in range(fx.shape[0]):
            for j in range(fx.shape[1]):
               if isinstance(fx[i][j],int):
                   ss.add(fx[i][j])
               else:
                   ss.add((fx[i][j][0],fx[i][j][1]  ,fx[i][j][2]))
        #print 'ssle ',len(ss)
        c=0
        cols=list()
        for r in range(1,250,1):
            for r2 in range(1, 250, 1):
                for r3 in range(1, 250, 1):
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
        #plt.imshow(fx)
        #plt.show()
        ss = set()
        for i in range(fx.shape[0]):
            for j in range(fx.shape[1]):
                ss.add((fx[i][j][0], fx[i][j][1], fx[i][j][2]))
        print 'ssle22 ', len(ss)


        #plt.imshow(fx)
        #plt.show()

        fx = np.uint8(fx)
        im = Image.fromarray(fx)
        #plt.imshow(im)
        #plt.show()
        return im


def process_suzuki(path,THRESH,ADAP):
    img = cv2.imread(
        path)  # 10978_13_001.png-1.jpg')# densities/143_13_001.png')#558_13_002.png')#fotoscale.jpg')143_13_001.png')#
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel2 = np.ones((9, 9), np.uint8)


    if ADAP:
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55,
                                       20)  # + cv2.THRESH_BINARY_INV)  # +cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+

    else:
        data = dict()
        for i in gray.ravel():
            if data.has_key(i):
                data[i] += 1
            else:
                data[i] = 1

        max = 0
        id = None
        for k in data.keys():
            # print 'k ',k
            if data[k] >= max and k < 240 and k > 130:  # 50:#240:
                id = k
                max = data[k]
                #   print 'key: ', id, ' - ', max
        id -= THRESH
        ret, thresh = cv2.threshold(gray, id, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C + cv2.THRESH_BINARY_INV)  # +cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
    opening2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
    #plt.imshow(opening)
    #plt.show()
    _, contours, hierarchy = cv2.findContours(opening.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)



    img2 = np.zeros_like(img)
    img2[:, :, 0] = opening
    img2[:, :, 1] = opening
    img2[:, :, 2] = opening

    cp=10000
    cols = list()
    cols2 = set()
    for r in range(1, 255, 1):
        for r1 in range(1, 255, 1):
            for r2 in range(1, 255, 1):
                cols.append((r, r1, r2))
                cp-=1
                if cp==0:
                    break
            if cp == 0:
                break
        if cp == 0:
            break
    ind=0
#    print len(contours)


    shuffle(cols)
    for c in cols:

        cv2.drawContours(img2, contours, ind, c, -1)
        ind += 1
        if ind==len(contours):
            break

    for x in range(opening.shape[0]):
        for y in range(opening.shape[1]):
            if opening[x][y]==0:
                img2[x][y]=(0,0,0)




 #   plt.imshow(img2)
 #   plt.show()
    #markers = np.uint8(opening)
    im = Image.fromarray(img2)
 #   plt.imshow(im)
 #   plt.show()
    return im

def process_felzen(path):
    if True:
        if True:

            if True:

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
                #print 'key: ', id, ' - ', max
                #id -= 30
                THRESH=30
                id -= THRESH  # 15
                ret, thresh = cv2.threshold(gray, id, 255, cv2.ADAPTIVE_THRESH_MEAN_C + cv2.THRESH_BINARY_INV)

                #thresh = cv2.adaptiveThreshold(gray, 255,
                #                            cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,55,20)# + cv2.THRESH_BINARY_INV)  # +cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+
                #plt.imshow(thresh)
                #plt.show()
                kernel = np.ones((3, 3), np.uint8)
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)

                for i in range(opening.shape[0]):
                    for j in range(opening.shape[1]):
                        if opening[i][j] == 0:
                            img[i][j] = (0, 0, 0)

                fx = segmentation.felzenszwalb(img, scale=100, sigma=0.8, min_size=50)

                ss = set()
                for e in range(len(fx)):
                    for ej in range(len(fx[0])):


                        ss.add(fx[e][ej])


                img2 = np.zeros_like(img)
                img2[:, :, 0] = opening
                img2[:, :, 1] = opening
                img2[:, :, 2] = opening
                c = 1
                dic = dict()
                dic[0] = (0, 0, 0)
                for r in range(1, 255, 1):
                    for r2 in range(1, 255, 1):
                        for r3 in range(1, 255, 1):
                            # cols.add((r,r2,r3))
                            dic[c] = (r, r2, r3)
                            if c == len(ss) + 5:
                                break
                            c += 1
                        if c == len(ss) + 5:
                            break
                    if c == len(ss) + 5:
                        break
                dic[0] = (0, 0, 0)
                for e in range(len(fx)):
                    for ej in range(len(fx[0])):
                        img2[e][ej] = dic[fx[e][ej]]

                #plt.imshow(fx)
                #plt.show()


                fx = np.uint8(img2)
                #s=set()
                #for x in fx:
                #    for y in  x:
                #        print 'x ',y
                #        s.add((y[0],y[1],y[2]))
                #print 'y.s ',len(s)
                im = Image.fromarray(fx)
                #plt.imshow(im)
                #plt.show()
                return im

###########

#segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
#ax[0, 0].imshow(mark_boundaries(img, segments_fz))