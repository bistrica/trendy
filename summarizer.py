import numpy
import sklearn
import numpy
import sklearn
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from random import shuffle
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.layers import Conv2DLayer, TransposedConv2DLayer
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import theano.sandbox.cuda

from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import keras

import os,sys
from PIL import Image
from os.path import isfile, join
from os import listdir
from skimage.color import rgb2gray
import theano.tensor as T
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.layers import Conv2DLayer, TransposedConv2DLayer
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import theano.sandbox.cuda

from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import keras

import os,sys
from PIL import Image
from os.path import isfile, join
from os import listdir


class Summarizer(object):

    def count_with_median(self,img2):
        im_arr = np.fromstring(img2.tobytes(), dtype=np.uint8)
        #print 'imm ',img2.size, im_arr.size
        im_arr = np.reshape(im_arr,(img2.size[0], img2.size[1],im_arr.size/(img2.size[0]*img2.size[1])))#img2.size[2]))
        img2=im_arr
       # pix_pic = img2.load()
        dic = dict()
        nonblack = 0
        for i in range(img2.shape[0]):
            for j in range(img2.shape[1]):
                # print '>',img2[i][j]
                if not (img2[i][j][0] == 0 and img2[i][j][1] == 0 and img2[i][j][2] == 0):  # ,0,0]:

                    nonblack += 1
                    key = str(img2[i][j][0]) + "_" + str(+img2[i][j][1]) + "_" + str(img2[i][j][2])
                    if key not in dic.keys():
                        dic[key] = 1
                    else:
                        dic[key] += 1
                        #                sets.add(str(out[i][j][0])+"_"+str(+out[i][j][1])+"_"+str(out[i][j][2]))
                    #print out[i][j]
        ids=[]
        for k in dic.keys():
            if dic[k]<30:
                ids.append(k)
        for k in ids:
            dic.pop(k)
        keys = dic.values()#keys()
        if len(keys)==0:
            print 'o!'
            plt.imshow(img2)
            plt.show()
        keys.sort()
        median = float(keys[len(keys) / 2])
        if len(keys) % 2 == 0:
            median += float(keys[len(keys) / 2 + 1])
            median /= 2
        #print median
        cells = float(nonblack) / median
        return (len(keys),cells)

    def count_tp(self, img, dens):
        #print 'cnt'
        #plt.imshow(img)
        #plt.show()
        if dens.size!=img.size:
            dens.thumbnail(img.size, Image.ANTIALIAS)
            #plt.imshow(dens)
            #plt.show()
        pix_pic = img.load()
        pix_dens = dens.load()
        colours=set()
        #print 'CP ',pix_pic[0,0], pix_dens[0,0]
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                cpixel=pix_dens[i,j]
                if (cpixel!=0):#0 and cpi[0],cpixel[1],cpixel[2])!=(0,0,0) and (cpixel[0],cpixel[1],cpixel[2])!=(255,255,255):
                    cpixelImg = pix_pic[i, j]
                    try:
                        if (cpixelImg[0], cpixelImg[1], cpixelImg[2]) != (0,0,0) and (cpixelImg[0], cpixelImg[1], cpixelImg[2])!= (-1,-1,-1):#(255,255,255):#0, 0, 0) and (cpixelImg[0], cpixelImg[1], cpixelImg[2]) != (255, 255, 255):
                            descr=cpixel#str(cpixelImg[0])+'_'+str(cpixelImg[1])+'_'+str(cpixelImg[2])+'.'+str(cpixel)
                            colours.add(descr)
                    except:

                        if (cpixelImg!=0 and cpixelImg!=-1):#0, 0, 0) and (cpixelImg[0], cpixelImg[1], cpixelImg[2]) != (255, 255, 255):
                            descr=cpixel#str(cpixelImg)+'.'+str(cpixel)
                            colours.add(descr)
                            #pix_pic[i,j]=99
        colours=list(colours)
        colours.sort()
        #for c in colours:
        #    print 'co ',c
        #plt.imshow(img)
        #plt.show()
        return len(colours)

    def make_density_map(self, path, is_red):
        im = Image.open(path)  # "/home/olusiak/Obrazy/schr.png")
        pix = im.load()
        red=(0,0,255)
        blue=(0,255,255)
        #plt.imshow(im)
        #plt.show()
        color=blue
        if is_red:
            color=red
        for i in range(im.size[0]):
            for j in range(im.size[1]):
                cpixel=pix[i,j]

                if (cpixel[0],cpixel[1],cpixel[2])!=color:
                    c=(0,0,0,cpixel[3])
                    pix[i,j]=c
                #else:
                #    print 'cpix',cpixel

        #plt.imshow(im)
        #plt.show()
        ratio=1
        im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((im.size[1], im.size[0], 4))
        kernel = np.ones((3, 3), np.uint8)
        kernel2 = np.ones((ratio * 2 + 1, ratio * 2 + 1), np.uint8)
        #plt.imshow(im_arr)
        #plt.show()
        erosion = cv2.erode(im_arr, kernel, iterations=1)
        erosion = cv2.dilate(erosion, kernel2, iterations=4)
        erosion=cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
        #plt.matshow(erosion)
        #plt.show()
        ret,thresh=cv2.threshold(erosion,1,255,cv2.THRESH_BINARY)#_INV)
        thresh = np.uint8(thresh)
        #print 'thres ',thresh.shape
        ret, markers = cv2.connectedComponents(thresh)
        #print 'ms ',markers.shape
        #plt.matshow(markers)
        #plt.show()
        #dilate = cv2.dilate(erosion, kernel2, iterations=1)
        #plt.imshow(dilate)
        #plt.show()
        #return dilate
        im = Image.fromarray(markers)
        return im