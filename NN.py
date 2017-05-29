
import cv2

import matplotlib.pyplot as plt

from random import shuffle
from urllib import urlretrieve
import cPickle as pickle

import gzip
import numpy as np

import lasagne
from lasagne import layers

from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import theano.sandbox.cuda


import os
from PIL import Image
from os.path import isfile, join
from os import listdir
from skimage.color import rgb2gray
import theano.tensor as T

theano.sandbox.cuda.use("gpu0")

path="/home/olusiak/Obrazy/densities/"
image_list=list()
output_list=list()
out=list()
imgs=list()
files = [f for f in listdir(path) if isfile(join(path, f))]
ran=16
ratio=4
size_orig = 2048/ratio,1536/ratio
size = 28,28#60,28# 28,28#size_orig[0] / ran, size_orig[1] / ran
PIXELS=size[0]*size[1]/4
print "> ",size[0], ' ',size[1]

def load_files():

    c=0
    files.sort()
    print files

    for filename in files:

        if 'xml' in filename or 'Colour' in filename or 'dens_map' in filename:# or 'density' in filename:
            continue


        if c == 20:
            break
        c += 1
        print 'f ',filename


        im = Image.open("{0}{1}".format(path, filename))  # "/home/olusiak/Obrazy/schr.png")
        if 'density' in filename:
            print 'erozja'

            im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((im.size[1], im.size[0], 4))
            kernel = np.ones((3, 3), np.uint8)

            kernel2 = np.ones((ratio*2+1, ratio*2+1), np.uint8)
            kernel2[0][0] = 0
            kernel2[0][2] = 0
            kernel2[2][0] = 0
            kernel2[2][2] = 0
            erosion = cv2.erode(im_arr, kernel, iterations=1)
            dilate = cv2.dilate(erosion, kernel2, iterations=1)
            #plt.imshow(dilate)
            #plt.show()
            im = Image.fromarray(dilate)
            #plt.imshow(im)
            #plt.show()



        im.thumbnail(size_orig, Image.ANTIALIAS)

        if 'density' not in filename:
            im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((im.size[1], im.size[0], 3))
            im_arr = rgb2gray(im_arr)
            im = Image.fromarray(im_arr)
        if False and 'density' not in filename:
            im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((im.size[1], im.size[0], 3))
            gray = cv2.cvtColor(im_arr, cv2.COLOR_BGR2GRAY)
            #gray = rgb2gray(im_arr)

            #plt.hist(gray.ravel(), bins=256, range=(0.0, 254.0))#, fc='k', ec='k')
            #plt.show()
            data = dict()
            for i in gray.ravel():
                if data.has_key(i):
                    data[i] += 1
                else:
                    data[i] = 1

            max = 0
            id = None
            for k in data.keys():
                print 'k ',k
                if data[k] >= max and k < 240:  # 50:#240:
                    id = k
                    max = data[k]
            print 'key: ', id, ' - ', max
            id -= 15

            ret, thresh = cv2.threshold(gray, id, 255,cv2.THRESH_BINARY_INV)
            im = Image.fromarray(thresh)

        pix = im.load()

        print size_orig[0],size_orig[1]
        for ox in range(0,size_orig[0],50):

            for oy in range(0,size_orig[1],50):

                for i in range(ran):

                    try:

                        test = ox + size[0] * i
                        test = pix[test, 0]
                        test = ox + size[0] * (i + 1) - 1
                        test = pix[test, 0]

                    except:

                        continue
                    for j in range(ran):
                        all_pixels = list()

                        try:

                            test=oy+size[1] * j
                            test=pix[0,test]
                            test=oy+size[1] * (j + 1) -1
                            test = pix[0, test]
                        except:

                            continue
                        for x in range(ox+size[0] * i, ox+size[0] * (i + 1), 1):
                            if True:

                                for y in range(oy+size[1] * j, oy+size[1] * (j + 1), 1):
                                    cpixel = pix[x,y]#x, y]



                                    if 'density' in filename:

                                        xx=0
                                        if cpixel[1]==255 and cpixel[2]==255:
                                            xx=1
                                        elif cpixel[2]==255:
                                            xx=1

                                        cpixel=xx
                                    all_pixels.append(cpixel)



                        all = list()
                        a1 = list()
                        a2 = list()
                        a2a=list()
                        a3 = list()

                        for a in all_pixels:

                            a2.append(a)
                            a1.append(a)

                        if 'density' not in filename:
                            aImg=list()
                            for ik in range(len(a2)):
                                aImg.append(a2[ik])

                                a2[ik]=float(a2[ik])#/255)
                            aImg= np.asarray(aImg)

                            aImg = np.reshape(aImg, (size[0], size[1]))  # okrr

                            aImg = np.rot90(aImg)  # okrr
                            aImg = np.flipud(aImg)
                            imgs.append(aImg)





                        a2 = np.asarray(a2)

                        a2 = np.reshape(a2, (size[0], size[1])) #okrr


                        a2=np.rot90(a2) #okrr
                        a2=np.flipud(a2)

                        all.append(a2)
                        all = np.asarray(all)


                        if 'density' in filename:

                            out_pixels = [item for sublist in a2 for item in sublist]  # in all_pixels

                            out_pixels=np.asarray(out_pixels)


                            a2=np.reshape(a2,(1,size[1],size[0]))

                            if True:
                                a = a2[0]
                                for row_id in range(len(a2[0])):
                                    for col_id in range(len(a2[0][0])):
                                        if a[row_id][col_id]==1:


                                            try:
                                                a[row_id-1][col_id]=-1
                                            except:

                                                f=1
                                            try:
                                                a[row_id+1][col_id] = -1
                                            except:

                                                f = 1
                                            try:
                                                a[row_id-1][col_id-1] = -1

                                            except:

                                                f = 1
                                            try:
                                                a[row_id-1][col_id+1] = -1
                                            except:

                                                f = 1
                                            try:
                                                a[row_id+1][col_id-1] = -1

                                            except:

                                                f = 1
                                            try:
                                                a[row_id+1][col_id+1] = -1

                                            except:

                                                f = 1
                                            try:
                                                a[row_id][col_id+1] = -1

                                            except:

                                                f = 1
                                            try:
                                                a[row_id][col_id-1] = -1

                                            except:

                                                f = 1

                                a = a2[0]
                                for row_id in range(len(a2[0])):
                                    for col_id in range(len(a2[0][0])):

                                        if a[row_id][col_id] == -1:
                                            a[row_id][col_id] = 1

                            out.append(a2)

                        else:
                            image_list.append(all)
load_files()


def load_dataset():
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        urlretrieve(url, filename)
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]
    X_train = X_train.reshape((-1, 1, 28, 28))
    X_val = X_val.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    dataset_size = len(X_train)
    #X_train = X_train.reshape(dataset_size, -1)

    dataset_size2 = len(X_test)
    #X_test = X_test.reshape(dataset_size2, -1)

    X_train=X_train#[:10000]
    y_train = y_train#[:10000]
    y_train2=list()
    y_test2=list()
    c=0


    return X_train, y_train2, X_val, y_val, X_test, y_test2




counterr=0

rem=list()
ind=0

for index in range(len(out)/10):
    ok=False
    ok2=False
    tab=out[index]
    for e in tab[0]:
        for f in range(len(e)):
            if e[f]!=1:
                ok2=True
            if e[f]!=0:
                ok=True
                if ok2:
                    break
        if ok and ok2:
            break
    if not (ok and ok2):
        rem.append(index)



print 'out len ',len(out), len(image_list)



counterr=len(rem)
for i in reversed(range(counterr)):
    out.pop(rem[i])
    #output_list.pop(rem[i])
    image_list.pop(rem[i])
print 'out len2 ',len(out)

def createConv2(attributes, labels, data, results):
    conv_nonlinearity = lasagne.nonlinearities.rectify
    if True:
        if True:
            X_train = attributes

            y_train = labels
            print 'X_train ', len(X_train),len(y_train)
            X_test = data
            y_test = results
            X_train = np.asarray(X_train)

            net1 = NeuralNet(
                layers=[('input', layers.InputLayer),
                        ('conv2d1', layers.Conv2DLayer),
                        ('conv2d2', layers.Conv2DLayer),
                        ('maxpool1', layers.MaxPool2DLayer),
                        ('conv2d3', layers.Conv2DLayer),
                        ('maxpool2', layers.MaxPool2DLayer),
                        #('flat',layers.ReshapeLayer), #added
                        ('dense',layers.DenseLayer),#added
                        ('resh',layers.ReshapeLayer),#added
                        ('up', layers.Upscale2DLayer),
                        ('transp1', layers.TransposedConv2DLayer),
                        ('up2', layers.Upscale2DLayer),
                        ('transp2', layers.TransposedConv2DLayer),
                        ('transp3', layers.TransposedConv2DLayer),
                        ('conv2d4', layers.Conv2DLayer),  # ,
                        ],


                # input layer
                input_shape=(None, 1, size[1], size[0]),
                # layer conv2d1
                conv2d1_num_filters=16,
                conv2d1_filter_size=(3, 3),
                conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
                conv2d2_num_filters=16,
                conv2d2_filter_size=(3, 3),
                conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
                #conv2d1_W=lasagne.init.GlorotUniform(),
                # layer maxpool1
                maxpool1_pool_size=(2, 2),
                # layer conv2d2
                conv2d3_num_filters=32,
                conv2d3_filter_size=(3, 3),
                conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
                # layer maxpool2
                maxpool2_pool_size=(2, 2),
                #flat
                #flat_outdim=1,
                #flat_shape=(1,32*4*6),
                #dense
                dense_num_units=32*5*5,#4*6,#('dense',layers.DenseLayer),
                dense_nonlinearity=lasagne.nonlinearities.rectify,
                #resh
                resh_shape=(-1,32,5,5),#4,6),

                #up
                up_scale_factor=2,

                #transp1
                transp1_num_filters=32,
                transp1_filter_size=(3, 3),
                transp1_nonlinearity=lasagne.nonlinearities.rectify,
                # up2
                up2_scale_factor=2,

                # transp2
                transp2_num_filters=16,
                transp2_filter_size=(3, 3),
                transp2_nonlinearity=lasagne.nonlinearities.rectify,
                # transp3
                transp3_num_filters=16,
                transp3_filter_size=(3, 3),
                transp3_nonlinearity=lasagne.nonlinearities.rectify,
                conv2d4_num_filters=1,
                conv2d4_filter_size=(1, 1),
                conv2d4_nonlinearity=lasagne.nonlinearities.rectify,


                update=nesterov_momentum,
                objective_loss_function=lasagne.objectives.squared_error,
                custom_score=("validation score", lambda x, y: np.mean(np.abs(x - y))),

                update_learning_rate=0.01,  # 0.01
                update_momentum=0.9,
                max_epochs=101,
                y_tensor_type=T.tensor4,
                verbose=1,  # ,
                regression=True
            )


            nn = net1.fit(X_train, y_train)
            preds = net1.predict(X_test)

            ifg = 0
            for pr in preds:
                plt.matshow(X_test[ifg][0],cmap='gray')
                plt.show()
                plt.matshow(y_test[ifg][0],cmap='gray')
                for c in y_test[ifg][0]:
                    print 'ef ',c
                plt.show()
                ifg+=1
                pr=pr[0]
                print 'PRR: ', pr
                #for p in pr:#range(pr.size):
                 #   for p1 in range(p.size):
                    #for p1 in range(p.size):
                        #print 'p[p1] ',p[p1]
                        #if p[p1]<0.5:
                         #   p[p1]=255#        pr[p][p1]=255

                plt.matshow(pr,cmap='gray')
                plt.show()
                print 'PRR2: ', pr





def createConv2b(attributes, labels, data, results):
    conv_nonlinearity = lasagne.nonlinearities.rectify
    if True:
        if True:
            X_train = attributes
            print 'X_train ', len(X_train)
            print 'len X_train ', len(X_train[0])
            #for x in X_train:
            #    print 'size ', x[0].size, x[0].shape
            #c=9/0
            y_train = labels
            X_test = data
            y_test = results
            X_train = np.asarray(X_train)
            print 'sh ', X_train.shape[1:]

            net1 = NeuralNet(
                layers=[('input', layers.InputLayer),
                        ('maxpool1', layers.MaxPool2DLayer),
                        ('conv2d1', layers.Conv2DLayer),
                        ('maxpool2', layers.MaxPool2DLayer),
                        ('conv2d2', layers.Conv2DLayer),
                        ('maxpool3', layers.MaxPool2DLayer),
                        ('conv2d3', layers.Conv2DLayer),
                        #('maxpool4', layers.MaxPool2DLayer),
                        ('conv2d4', layers.Conv2DLayer),
                        ('dense', layers.DenseLayer),#dense_num_units = 32 * 4 * 6,  #
                                          #dense_nonlinearity = lasagne.nonlinearities.rectify,
                        ('reshape',layers.ReshapeLayer),
                        ('up', layers.Upscale2DLayer),
                        ('transp1', layers.TransposedConv2DLayer),
                        ('up2', layers.Upscale2DLayer),
                        ('transp2', layers.TransposedConv2DLayer),
                        ('up3', layers.Upscale2DLayer),
                        ('transp3', layers.TransposedConv2DLayer),
                        ('conv2d5', layers.Conv2DLayer),  # ,
                        ],


                # input layer
                input_shape=(None, 1, size[1], size[0]),
                # layer conv2d1
                maxpool1_pool_size=(2, 2),

                conv2d1_num_filters=32,
                conv2d1_filter_size=(3, 3),
                conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
                #conv2d1_pad='same',
                maxpool2_pool_size=(2, 2),

                conv2d2_num_filters=64,
                conv2d2_filter_size=(3, 3),
                #conv2d2_pad='same',
                conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
                #conv2d1_W=lasagne.init.GlorotUniform(),
                # layer maxpool1
                maxpool3_pool_size=(2, 2),
                # layer conv2d2
                conv2d3_num_filters=128,
                conv2d3_filter_size=(3, 3),
                conv2d3_pad='same',
                conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
                # layer maxpool2
                #maxpool4_pool_size=(2, 2),
                conv2d4_num_filters=512,
                conv2d4_filter_size=(3, 3),

                conv2d4_pad='same',
                conv2d4_nonlinearity=lasagne.nonlinearities.rectify,

                dense_num_units=512 * 2 * 6,#2,  # ('dense',layers.DenseLayer),
                dense_nonlinearity=lasagne.nonlinearities.rectify,
                # resh
                reshape_shape=(-1,512,2,6),#2),
                # up
                up_scale_factor=2,
                #('up', layers.Upscale2DLayer),
                #transp1
                transp1_num_filters=128,
                transp1_filter_size=(3, 3),
                transp1_nonlinearity=lasagne.nonlinearities.rectify,
                # up2
                up2_scale_factor=2,
                #('transp1', layers.TransposedConv2DLayer),
                #('up2', layers.Upscale2DLayer),
                #('transp2', layers.TransposedConv2DLayer),
                #('transp3', layers.TransposedConv2DLayer),
                # transp2
                transp2_num_filters=64,
                transp2_filter_size=(3, 3),
                transp2_nonlinearity=lasagne.nonlinearities.rectify,
                # transp3
                up3_scale_factor=2,
                transp3_num_filters=16,
                transp3_filter_size=(3, 3),
                transp3_crop='same',
                transp3_nonlinearity=lasagne.nonlinearities.rectify,
                conv2d5_num_filters=1,
                #conv2d5_pad='same',
                conv2d5_filter_size=(1, 1),

                conv2d5_nonlinearity=lasagne.nonlinearities.rectify,

                objective_loss_function=lasagne.objectives.squared_error,
                custom_score=("validation score", lambda x, y: np.mean(np.abs(x - y))),
                update=nesterov_momentum,
                update_learning_rate=0.05,
                update_momentum=0.9,

                max_epochs=75,
                y_tensor_type=T.tensor4,
                verbose=1,  # ,

                regression=True
            )

            print 'yt ', len(y_train), len(y_train[0])

            nn = net1.fit(X_train, y_train)
            preds = net1.predict(X_test)

            ifg = 0
            for pr in preds:
                plt.matshow(X_test[ifg][0])
                plt.show()
                plt.matshow(y_test[ifg][0])
                plt.show()
                ifg+=1
                pr=pr[0]
                m=0
                print 'PRR: ', pr
                for p in pr:
                    print '=='
                    for p1 in range(p.size):

                        print 'p[ ',p[p1]
                        if p[p1]>m:
                            m=p[p1]
                        if p[p1]<0.15:
                            p[p1]=255
                print 'm ',m
                plt.matshow(pr)
                plt.show()
                print 'PRR2: ', pr



def load_caltech():
    path='/home/olusiak/Pobrane/CALTECH/CALTECH/'
    XX=100
    X=list()
    c=0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(".jpg")]:

            im = Image.open("{0}{1}".format(dirpath+'/', filename))
            c+=1
            if c==XX:
                break



            from PIL import ImageOps
            thumb = ImageOps.fit(im, (28,28), Image.ANTIALIAS)

            im_arr = np.fromstring(thumb.tobytes(), dtype=np.uint8)

            print im_arr.size, im.size[1], im.size[0]
            im_arr = im_arr.reshape((thumb.size[1], thumb.size[0], im_arr.size/(thumb.size[1]*thumb.size[0])))

            try:
                im_arr = rgb2gray(im_arr)
            except:
                print "im ",im_arr.size, thumb.size
            im_arr = im_arr.reshape((1,im_arr.shape[0],im_arr.shape[1]))
            print "im sh ", im_arr.shape
            X.append(im_arr)

        if c==XX:
            break
    shuffle(X)
    per=0.9
    X = [X[:int(per * len(X))], X[int(per * len(X)):]]
    return X

def load_SVHN():
    path='/home/olusiak/Pobrane/train/'
    XX=50000
    X=list()
    c=0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(".png")]:

            im = Image.open("{0}{1}".format(dirpath+'/', filename))
            c+=1
            if c==XX:
                break



            from PIL import ImageOps
            thumb = ImageOps.fit(im, size, Image.ANTIALIAS)

            im_arr = np.fromstring(thumb.tobytes(), dtype=np.uint8)

            print im_arr.size, im.size[1], im.size[0]
            im_arr = im_arr.reshape((thumb.size[1], thumb.size[0], im_arr.size/(thumb.size[1]*thumb.size[0])))
         #   plt.imshow(im_arr)
          #  plt.show()
            try:
                im_arr = rgb2gray(im_arr)
            except:
                print "im ",im_arr.size, thumb.size
            im_arr = im_arr.reshape((1,im_arr.shape[0],im_arr.shape[1]))
            print "im sh ", im_arr.shape
            X.append(im_arr)

        if c==XX:
            break
    shuffle(X)
    per=0.9
    X = [X[:int(per * len(X))], X[int(per * len(X)):]]
    return X



print 'COUNT ',len(out), ' ',len(image_list), counterr

per=0.9
ids=range(0,len(image_list))
shuffle(ids)
image_list2=list()
out2=list()
imgs2=list()
for i in ids:
    image_list2.append(image_list[i])
    out2.append(out[i])
    imgs2.append(imgs[i])
image_list=image_list2
out=out2
imgs=imgs2
X = [image_list[:int(per*len(image_list))],image_list[int(per*len(image_list)):]]
Xt = [imgs[:int(per*len(imgs))],imgs[int(per*len(imgs)):]]
print 'IMA ',len(image_list)

output_list=out

X = [image_list[:int(per*len(image_list))],image_list[int((per)*len(image_list)):]]


print len(X[0]),len(X[1])
Y = [output_list[:int(per*len(output_list))],output_list[int((per)*len(output_list)):]]

#tup=load_dataset()
#X=load_caltech()
#X=load_SVHN()
#X_train=tup[0]
#X_test=tup[4]
result=createConv2(X[0],Y[0],X[1],Y[1])
