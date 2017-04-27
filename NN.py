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
from NNos import train
theano.sandbox.cuda.use("gpu0")
print 'gpu?'
path="/home/olusiak/Obrazy/densities/"
image_list=list()
output_list=list()
out=list()
imgs=list()
files = [f for f in listdir(path) if isfile(join(path, f))]
ran=8
ratio=4
size_orig = 2048/ratio,1536/ratio
size =  size_orig[0] / ran, size_orig[1] / ran
PIXELS=size[0]*size[1]/4
print "> ",size[0], ' ',size[1]

def load_files():
    #print len(files)
    c=0
    files.sort()
    print files

    for filename in files:



        if 'dens_map' in filename or 'xml' in filename or 'Colour' in filename:
            continue
        c += 1
        print 'f ',filename
        #if c == 3:
        #    break
        print filename, c
        #if c==60:
            #break
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
    #        plt.imshow(dilate)
    #        plt.show()
            im = Image.fromarray(dilate)
            #plt.imshow(im)
            #plt.show()

        #print size

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
            # if data.has_key(255):
            max = 0
            id = None
            for k in data.keys():
                print 'k ',k
                if data[k] >= max and k < 240:  # 50:#240:
                    id = k
                    max = data[k]
            print 'key: ', id, ' - ', max
            id -= 15

            ret, thresh = cv2.threshold(gray, id, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C + cv2.THRESH_BINARY_INV)  # +cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+
            im = Image.fromarray(thresh)

        pix = im.load()
        #plt.imshow(im)
        #plt.show()
        print size_orig[0],size_orig[1]
        for ox in range(0,size_orig[0],50):#size_orig[0],10):
            #print 'OX ',ox
            for oy in range(0,size_orig[1],50):
                #print 'OX OY ',ox,' ',oy
                for i in range(ran):
                    #pix2=np.copy(pix[0])
                    #pix2 = np.asarray(pix2)
                    #print 'pix2 ',pix2.size
                    #pix2 = np.reshape(pix2, (size[1], size[0],3))
                    #pix2[20, 0] = 31

                    #plt.matshow(pix2)
                    try:

                        test = ox + size[0] * i
                        test = pix[test, 0]
                        test = ox + size[0] * (i + 1) - 1
                        test = pix[test, 0]

                    except:
                        #print ox,' conti ', i
                        #break
                        continue
                    for j in range(ran):#i, ran, 1):
                        all_pixels = list()#np.array(int)
                        #aa=list()
                        #print '>', size[0] * i, ' ', size[0] * (i + 1),  ' / ', size[1] * j, ' ', size[1] * (j + 1)
                        d = 0
                        dd = 0
                        try:

                            test=oy+size[1] * j
                            test=pix[0,test]
                            test=oy+size[1] * (j + 1) -1
                            test = pix[0, test]
                        except:
                            #break
                            continue
                        for x in range(ox+size[0] * i, ox+size[0] * (i + 1), 1):
                            #for j in range(ran):  # i, ran, 1):
                            if True:

                                for y in range(oy+size[1] * j, oy+size[1] * (j + 1), 1):
                                    #print 'x,y',x,' ',y, ox, size[0], i, j, ran
                                    #print ox+size[0] * i, ox+size[0] * (i + 1),'/',oy+size[1] * j, oy+size[1] * (j + 1)
                                    cpixel = pix[x,y]#x, y]
                                    #if 'density' in filename:
                                    #    if pix[x,y][1]==255:
                                    #        print 'PX', pix[x,y]
                                    cpixel2=pix[x,y]
                                    #if 'density' in filename:
                                    #    if pix[x,y][1]==35 or pix[x,y][1]==55:
                                    #        print 'px',pix[x,y],', ',cpixel
                                    if 'density' in filename:
                                    #if len(cpixel)>3:
                                        xx=1
                                        if cpixel[1]==255 and cpixel[2]==255:
                                            xx=0#25#cpixel[0]=125
                                        elif cpixel[2]==255:
                                            xx=0#55#cpixel[1]
                                        #if cpixel[1]==255:
                                        #    xx=255#cpixel[0]=255
                                        cpixel=xx#(xx,xx,xx)#cpixel[0],xx,cpixel[2])
                                        #if xx!=0:
                                        #    print 'cc ',cpixel

                                        #if xx==125:
                                        #    d+=1
                                        #if xx==255:
                                        #    dd+=1

                                        #cpixel2=(cpixel2[1],cpixel2[2],cpixel2[3])#,cpixel2[3])
                                    #if len(cpixel) > 3:
                                    #    print ":: ", len(cpixel)
                                    #if 'density' in filename:
                                    #    print 'cp ',pix[x,y], ' ',cpixel
                                    all_pixels.append(cpixel)
                                    #aa.append(cpixel2)
                                    #all_pixels=np.asarray(all_pixels)
                                    #print 'Al ',all_pixels.shape
                                    # print 'L: ',len(all_pixels)

                        #print 'D: ', d, ' ,', dd
                        #print 'ALL ',len(all_pixels)

                        #plt.show()
                        #print 'o: ',all_pixels[0]
                        all = list()
                        a1 = list()
                        a2 = list()
                        a2a=list()
                        a3 = list()
                        #a3a = list()
                        for a in all_pixels:

                            a2.append(a)
                            a1.append(a)#[0])
                            #a2.append(a[1])
                            #a2a.append(a[1])
                            #a3.append(a[2])

                        if 'density' not in filename:
                            aImg=list()
                            for ik in range(len(a2)):
                                aImg.append(a2[ik])
                                a2[ik]=float(a2[ik]/255)
                            aImg= np.asarray(aImg)
                            # print 'a ',a2.shape
                            aImg = np.reshape(aImg, (size[0], size[1]))  # okrr

                            aImg = np.rot90(aImg)  # okrr
                            aImg = np.flipud(aImg)
                            imgs.append(aImg)

                        if False and 'density' not in filename:
                            a1=np.asarray(a1)
#                            a2a = np.asarray(a2a)
#                            a3 = np.asarray(a3)
                            a1=np.reshape(a1,(size[1],size[0]))#0 1
#                            a2a=np.reshape(a2,(size[0], size[1]))
#                            a3=np.reshape(a3,(size[0], size[1]))

 #                           a2a = np.rot90(a2a)
 #                           a2a = np.flipud(a2a)
                            #a1 = np.rot90(a1) #rr
                            #a1 = np.flipud(a1)#rr
 #                           a3 = np.rot90(a3)
 #                           a3 = np.flipud(a3)
                            #print a1
                            #plt.matshow(a1)
                            #plt.show()
 #                           plt.matshow(a2a)
 #                           plt.show()
 #                           plt.matshow(a3)
 #                           plt.show()
                        #for a in aa:
                        #    a3a.append(a[0])
                        #a1 = np.asarray(a1)
                        #a1 = np.reshape(a1, (size[1], size[0]))

                        if False and 'density' in filename:
                            for e in range(len(a2)):
                                if a2[e]==2:
                                    a2[e]=0
                                else:# a2[e]==1:
                                    a2[e]=255
                        a2 = np.asarray(a2)
                        #print 'a ',a2.shape
                        a2 = np.reshape(a2, (size[0], size[1])) #okrr


                        a2=np.rot90(a2) #okrr
                        a2=np.flipud(a2)
                        #a2 = np.rot90(a2)  # okrr
                        #if 'density' not in filename:
                        #    plt.matshow(a2)
                        #    plt.show()
                        all.append(a2)
                        all = np.asarray(all)

                        #print 'a2 ',a2.shape, ', ' ,all.shape
                        if 'density' in filename:
                            #ou=list()
                            out_pixels = [item for sublist in a2 for item in sublist]  # in all_pixels
                            #for p in out_pixels:
                            #    if p==0:
                            #        ou.extend([1,0,0])
                            #    elif p==1:
                            #        ou.extend([0,1,0])
                            #    else:
                            #        ou.extend([0,0,1])

                            #ou=np.asarray(ou)

                            #out.append(ou)
                            #all_pixels=[item for sublist in a2 for item in sublist]#in all_pixels
                            out_pixels=np.asarray(out_pixels)
                            #output_list.append(ou)#all_pixels)
                            #out_pixels=np.reshape(out_pixels,(1,1,out_pixels.size))
                            #plt.matshow(a2)
                            #plt.show()
                            a2=np.reshape(a2,(1,size[1],size[0]))
                            out.append(a2)#out_pixels)#a2)
                            # output_list.append(all_pixels2)
                            # output_list.append(all_pixels3)
                            # output_list.append(all_pixels4)
                        else:
        #                    all_pixels = [item for sublist in all_pixels for item in sublist]

                            #print 'all sh ',all.shape

                            #print 'SHAPE ',all.shape
                            #print 'a3 ',a3.shape
                            image_list.append(all)#_pixels)
                            # image_list.append(all_pixels2)
                            # image_list.append(all_pixels3)
                            # image_list.append(all_pixels4)

                        #print len(image_list), ' ', len(output_list)

        #break
#np.asarray(a)

load_files()

        #print 'pix ',len(all_pixels)#im.size

from theano.tensor.signal import pool

#from sklearn.neural_network import MLPClassifier

#clf=MLPClassifier(solver='lbfgs', alpha=1e-5,
#                        hidden_layer_sizes=(5, 2), random_state=1)
relations=[] #wszystkie lub hiponimy hiperonimy antonimia wlasciwa


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

    X_train=X_train[:10000]
    y_train = y_train[:10000]
    y_train2=list()
    y_test2=list()
    c=0
    for y in y_train:
        c+=1
        if c%2==0:
            y_train2.append((1,0,0,0,0,1))#((y,0),(0,1)))
        else:
            y_train2.append((0,1,0,1,0,0))#((0,y), (1,0)))
    for y in y_test:
        c+=1
        if c%2==0:
            y_test2.append((1,0,0,1,0,0))#(((y,0), (0,1)))
        else:
            y_test2.append((0,1,0,1,0,0))#(((0,y), (1,0)))
        #y_test2.append((y,c%3))

    #print y_train
    #X_test = X_test[:100]
    #y_test = y_test[:100]
    #print 'y test ', y_test

    return X_train, y_train2, X_val, y_val, X_test, y_test2

def createNeural(attributes, labels,data):
    xxx=0

    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                    hidden_layer_sizes=(5, 2), random_state=1)

    #clf.fit(attributes, labels)
    #result = clf.predict(data)
    #return result
    #print '>', clf.predict([[2., 2.], [-1., -2.]])

def createConv(attributes,labels,data,results):
    X_train=attributes
    print 'X_train ',len(X_train)
    print 'len X_train ', len(X_train[0])
    for x in X_train:
        print 'size ', x[0].size, x[0].shape

    y_train=labels
    X_test=data
    y_test=results
    X_train=np.asarray(X_train)
    print 'sh ',X_train.shape[1:]

    input_var = T.tensor4('X')
    output_var = T.tensor4('X')

    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                ('maxpool1', layers.MaxPool2DLayer),
                ('conv2d2', layers.Conv2DLayer),
                ('maxpool2', layers.MaxPool2DLayer),
                ('conv2d3', layers.Conv2DLayer),
                #('maxpool2', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
                ('dense', layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),

                ('output', layers.DenseLayer)#TransposedConv2DLayer),#,
                ],
        # input layer
        input_shape=(None,1, size[1],size[0]),#,input_var),
        # layer conv2d1
        conv2d1_num_filters=32,
        conv2d1_filter_size=(5, 5),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),
        # layer maxpool1
        maxpool1_pool_size=(2,2),
        # layer conv2d2
        conv2d2_num_filters=32,
        conv2d2_filter_size=(5, 5),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        # layer maxpool2
        maxpool2_pool_size=(2, 2),
        # dropout1
        dropout1_p=0.5,
        # dense
        dense_num_units=size[0]*size[1]*60,
        dense_nonlinearity=lasagne.nonlinearities.rectify,
        # dropout2
        dropout2_p=0.5,
        conv2d3_num_filters=32,
        conv2d3_filter_size=(1, 1),
        #stride=2,

        conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        #output_num_filters=1,
        #output_stride=(2,2),
#        output_incoming=(None),
        #output_filter_size=(2,2),
        #output_output_size=(size[1],size[0]),
        #output
        output_num_units=3*size[0]*size[1],#*3,
        #output_output_size=(size[1],size[0]),
        #output_incoming=(None,1,)
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.01,#0.01
        update_momentum=0.9,
        max_epochs=30,
        verbose=1,#,
        regression=True
        )
    # Train the network
    print 'yt ',len(y_train[0])

    nn = net1.fit(X_train, y_train)
    preds = net1.predict(X_test)

    i=0
    for pr in preds:

        print 'pr ',pr
        print 'len ',pr.size
        if True:
            ind=0
            arr=list()#np.array(int)
            while ind<pr.size:
                res=0
                m=max(pr[ind],pr[ind+1],pr[ind+2])
                if pr[ind]==m:
                    arr.append(50)
                elif pr[ind+1]==m:
                    arr.append(125)
                elif pr[ind+2]==m:
                    arr.append(255)
                else:
                    print 'BAD', pr[ind],pr[ind+1],pr[ind+2]
                    pr[ind]=0
                ind+=3
            arr=np.asarray(arr)
            print 'arr',arr
            print 'lenarr ',arr.size
            pr=arr

            pr=np.reshape(pr,(size[1],size[0]))
            #pr = np.rot90(pr)
            #pr = np.flipud(pr)
            plt.matshow(pr)
            plt.show()
        #for p in pr:
        print 'PR: ',pr
        #plt.matshow(pr)
        #plt.show()
        plt.matshow(X_test[i][0])
        i+=1
        plt.show()
    print preds.shape

    #cm = confusion_matrix(y_test, preds)
    #plt.matshow(preds)
    #plt.title('Confusion matrix')
    #plt.colorbar()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    #plt.show()



def predict(data):
    #result=clf.predict(data)
    return result

def compare(result, ground_truth):
    good=0
    bad=0
    for i in range(len(result)):
        if result[i]==ground_truth[i]:
            good+=1
        else:
            bad+=1
    return [good, bad]

#def load_mnist():
#set=load_dataset()

#print 'so ',set[0]
#print 'l so ',len(set[0][0])
#print 's1 ',len(set[1])
#print 'len ', len(set[0]), ' ',len(set[1])
#result=createNeural(set[0], set[1],set[4])
#print 'im ',len(image_list)
#Y=range(len(image_list))#



#print ': ',X
#print 'im ',image_list[0]
#print 'len im ',len(image_list[0])
#for x in X[0]:
#    print 'x:',len(x)

#for x in X[1]:
#    print 'x1:',len(x)
#print 'out ',output_list[0]
counterr=0

rem=list()
ind=0
for lista in output_list:

    x=True
    #for elem in lista:
    i=0
    while i<len(lista):
        if not (lista[i]==1 and lista[i+1]==0 and lista[i+2]==0):
            x=False
            break
        i+=3
        rem.append(ind)#lista)
    if not x:
        counterr+=1
    ind+=1
counterr=counterr*3/4





for i in reversed(range(counterr)):
    out.pop(rem[i])
    #output_list.pop(rem[i])
    image_list.pop(rem[i])

def createConv2(attributes, labels, data, results):
    conv_nonlinearity = lasagne.nonlinearities.rectify
    if True:
        if True:
            X_train = attributes
            print 'X_train ', len(X_train)
            print 'len X_train ', len(X_train[0])
            for x in X_train:
                print 'size ', x[0].size, x[0].shape
            #c=9/0
            y_train = labels
            X_test = data
            y_test = results
            X_train = np.asarray(X_train)
            print 'sh ', X_train.shape[1:]

            net1 = NeuralNet(
                layers=[('input', layers.InputLayer),
                        ('conv2d1', layers.Conv2DLayer),
                        ('conv2d2', layers.Conv2DLayer),
                        ('maxpool1', layers.MaxPool2DLayer),
                        ('conv2d3', layers.Conv2DLayer),
                        ('maxpool2', layers.MaxPool2DLayer),
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
                #up
                up_scale_factor=2,
                #('up', layers.Upscale2DLayer),
                #transp1
                transp1_num_filters=32,
                transp1_filter_size=(3, 3),
                transp1_nonlinearity=lasagne.nonlinearities.rectify,
                # up2
                up2_scale_factor=2,
                #('transp1', layers.TransposedConv2DLayer),
                #('up2', layers.Upscale2DLayer),
                #('transp2', layers.TransposedConv2DLayer),
                #('transp3', layers.TransposedConv2DLayer),
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
                conv2d4_nonlinearity=lasagne.nonlinearities.sigmoid,
                #('conv2d4', layers.Conv2DLayer),  # ,

                #output_nonlinearity=lasagne.nonlinearities.sigmoid,
                # output_num_filters=1,
                # output_stride=(2,2),
                #        output_incoming=(None),
                # output_filter_size=(2,2),
                # output_output_size=(size[1],size[0]),
                # output
                #output_num_units=2 * size[0] * size[1],  # *3,
                # output_output_size=(size[1],size[0]),
                # output_incoming=(None,1,)
                # optimization method params
                update=nesterov_momentum,
                objective_loss_function=lasagne.objectives.squared_error(),
                update_learning_rate=0.01,  # 0.01
                update_momentum=0.9,
                max_epochs=5,
                y_tensor_type=T.tensor4,
                verbose=1,  # ,
                regression=True
            )
            # Train the network
            print 'yt ', len(y_train[0])

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
                print 'PRR: ', pr
                for p in pr:#range(pr.size):
                    for p1 in range(p.size):
                    #for p1 in range(p.size):
                        if p[p1]!=0:
                            p[p1]=255#        pr[p][p1]=255

                plt.matshow(pr)
                plt.show()
                print 'PRR2: ', pr







            return
            #c=9/0
            for pr in preds:
                plt.matshow(X_test[ifg][0])
                i = ifg
                ifg += 1
                plt.show()

                ind = 0
                arr = list()  # np.array(int)

                while ind < y_test[i].size:
                    res = 0
                    # print 'sigm ', (pr[ind], pr[ind + 1])  # ,pr[ind+2])
                    m = max(y_test[i][ind], y_test[i][ind + 1])  # ,pr[ind+2])
                    if y_test[i][ind] == m:
                        arr.append(50)
                    elif y_test[i][ind + 1] == m:
                        arr.append(125)
                    # elif pr[ind+2]==m:
                    #    arr.append(255)
                    else:
                        # print 'BAD', pr[ind], pr[ind + 1], pr[ind + 2]
                        y_test[i][ind] = 0
                    ind += 2
                arr = np.asarray(arr)
                # print 'arr', arr
                # print 'lenarr ', arr.size
                y_test[i] = arr

                y_test[i] = np.reshape(y_test[i], (size[1], size[0]))
                # pr = np.rot90(pr)
                # pr = np.flipud(pr)
                plt.matshow(y_test[i])
                plt.show()

                print 'pr ', pr
                print 'len ', pr.size
                num = list()
                for i in range(size[0] * size[1]):
                    num.append(0)
                num = np.asarray(num)  # np.array(int)
                num.reshape(size[0], size[1])

                if False:
                    for i in range(10):
                        ind1 = int(pr[i])
                        ind2 = int(pr[i + 1])
                        if ind1 < 0:
                            ind1 = ind1 * (-1)
                        if ind2 < 0:
                            ind2 = ind2 * (-1)
                        print 'ind ', ind1, ind2
                        num[ind1, ind2] = 2
                        ind1 = int(pr[10 + i])
                        ind2 = int(pr[10 + i + 1])
                        if ind1 < 0:
                            ind1 = ind1 * (-1)
                        if ind2 < 0:
                            ind2 = ind2 * (-1)

                        num[ind1, ind2] = 1
                    plt.matshow(num)
                    plt.show()

                if True:
                    ind = 0
                    arr = list()  # np.array(int)
                    while ind < pr.size:
                        res = 0
                        print 'sigm ', (pr[ind], pr[ind + 1])  # ,pr[ind+2])
                        m = max(pr[ind], pr[ind + 1])  # ,pr[ind+2])
                        if pr[ind] == m:
                            arr.append(50)
                        elif pr[ind + 1] == m:
                            arr.append(125)
                        # elif pr[ind+2]==m:
                        #    arr.append(255)
                        else:
                            print 'BAD', pr[ind], pr[ind + 1], pr[ind + 2]
                            pr[ind] = 0
                        ind += 2
                    arr = np.asarray(arr)
                    print 'arr', arr
                    print 'lenarr ', arr.size
                    pr = arr

                    pr = np.reshape(pr, (size[1], size[0]))
                    # pr = np.rot90(pr)
                    # pr = np.flipud(pr)
                    plt.matshow(pr)
                    plt.show()
                # for p in pr:
                print 'PR: ', pr
                # plt.matshow(pr)
                # plt.show()
                i += 1
            print preds.shape




def createConv2b(attributes, labels, data, results):
    conv_nonlinearity = lasagne.nonlinearities.rectify
    if True:
        if True:
            X_train = attributes
            print 'X_train ', len(X_train)
            print 'len X_train ', len(X_train[0])
            for x in X_train:
                print 'size ', x[0].size, x[0].shape
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
                #up
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

                conv2d5_nonlinearity=lasagne.nonlinearities.sigmoid,
                #('conv2d4', layers.Conv2DLayer),  # ,

                #output_nonlinearity=lasagne.nonlinearities.sigmoid,
                # output_num_filters=1,
                # output_stride=(2,2),
                #        output_incoming=(None),
                # output_filter_size=(2,2),
                # output_output_size=(size[1],size[0]),
                # output
                #output_num_units=2 * size[0] * size[1],  # *3,
                # output_output_size=(size[1],size[0]),
                # output_incoming=(None,1,)
                # optimization method params
                objective_loss_function=lasagne.objectives.squared_error,
                update=nesterov_momentum,
                update_learning_rate=0.01,  # 0.01
                update_momentum=0.9,
                max_epochs=100,
                y_tensor_type=T.tensor4,
                verbose=1,  # ,
                regression=True
            )
            # Train the network
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
                print 'PRR: ', pr
                for p in pr:#range(pr.size):
                    for p1 in range(p.size):
                    #for p1 in range(p.size):
                        if p[p1]!=0:
                            p[p1]=255#        pr[p][p1]=255

                plt.matshow(pr)
                plt.show()
                print 'PRR2: ', pr







            return
            #c=9/0
            for pr in preds:
                plt.matshow(X_test[ifg][0])
                i = ifg
                ifg += 1
                plt.show()

                ind = 0
                arr = list()  # np.array(int)

                while ind < y_test[i].size:
                    res = 0
                    # print 'sigm ', (pr[ind], pr[ind + 1])  # ,pr[ind+2])
                    m = max(y_test[i][ind], y_test[i][ind + 1])  # ,pr[ind+2])
                    if y_test[i][ind] == m:
                        arr.append(50)
                    elif y_test[i][ind + 1] == m:
                        arr.append(125)
                    # elif pr[ind+2]==m:
                    #    arr.append(255)
                    else:
                        # print 'BAD', pr[ind], pr[ind + 1], pr[ind + 2]
                        y_test[i][ind] = 0
                    ind += 2
                arr = np.asarray(arr)
                # print 'arr', arr
                # print 'lenarr ', arr.size
                y_test[i] = arr

                y_test[i] = np.reshape(y_test[i], (size[1], size[0]))
                # pr = np.rot90(pr)
                # pr = np.flipud(pr)
                plt.matshow(y_test[i])
                plt.show()

                print 'pr ', pr
                print 'len ', pr.size
                num = list()
                for i in range(size[0] * size[1]):
                    num.append(0)
                num = np.asarray(num)  # np.array(int)
                num.reshape(size[0], size[1])

                if False:
                    for i in range(10):
                        ind1 = int(pr[i])
                        ind2 = int(pr[i + 1])
                        if ind1 < 0:
                            ind1 = ind1 * (-1)
                        if ind2 < 0:
                            ind2 = ind2 * (-1)
                        print 'ind ', ind1, ind2
                        num[ind1, ind2] = 2
                        ind1 = int(pr[10 + i])
                        ind2 = int(pr[10 + i + 1])
                        if ind1 < 0:
                            ind1 = ind1 * (-1)
                        if ind2 < 0:
                            ind2 = ind2 * (-1)

                        num[ind1, ind2] = 1
                    plt.matshow(num)
                    plt.show()

                if True:
                    ind = 0
                    arr = list()  # np.array(int)
                    while ind < pr.size:
                        res = 0
                        print 'sigm ', (pr[ind], pr[ind + 1])  # ,pr[ind+2])
                        m = max(pr[ind], pr[ind + 1])  # ,pr[ind+2])
                        if pr[ind] == m:
                            arr.append(50)
                        elif pr[ind + 1] == m:
                            arr.append(125)
                        # elif pr[ind+2]==m:
                        #    arr.append(255)
                        else:
                            print 'BAD', pr[ind], pr[ind + 1], pr[ind + 2]
                            pr[ind] = 0
                        ind += 2
                    arr = np.asarray(arr)
                    print 'arr', arr
                    print 'lenarr ', arr.size
                    pr = arr

                    pr = np.reshape(pr, (size[1], size[0]))
                    # pr = np.rot90(pr)
                    # pr = np.flipud(pr)
                    plt.matshow(pr)
                    plt.show()
                # for p in pr:
                print 'PR: ', pr
                # plt.matshow(pr)
                # plt.show()
                i += 1
            print preds.shape

print 'COUNT ',len(out), ' ',len(image_list), counterr

per=0.6
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
X = [image_list[:int(per*len(image_list))],image_list[int(per*len(image_list)):]]#int(len(image_list) * .25) : int(len(image_list) * .75)]
Xt = [imgs[:int(per*len(imgs))],imgs[int(per*len(imgs)):]]
print 'IMA ',len(image_list)
for im in image_list:
    print '>',len(im[0])
output_list=out


X = [image_list[:int(per*len(image_list))],image_list[int(per*len(image_list)):int((per+0.2)*len(image_list))],image_list[int((per+0.2)*len(image_list)):]]
print len(X[0]),len(X[1])
Y = [output_list[:int(per*len(output_list))],output_list[int(per*len(output_list)):int((per+0.2)*len(output_list))],output_list[int((per+0.2)*len(output_list)):]]

#Y = [output_list[:int(per*len(output_list))],output_list[int(per*len(output_list)):]]#int(len(image_list) * .25) : int(len(image_list) * .75)]
#print '>', Y
#print 'j ',len(Y[0][0])
#print 'out ',output_list[0][0] #"[Y] ",len(Y[0])," . 147456"

#Y=np.asarray(Y)

#Y = output_list[int(len(output_list) * .25) : int(len(output_list) * .75)]

#Y1 = range(75) #Y[:int(len(Y) * .25)]
#Y2= range(225) #Y[int(len(Y) * .25) : int(len(Y) * .75)]#
#print 'y ',len(Y1),'...',len(Y2), ' ',len(X[0]), ' ',len(X[1])#len(Y[0])

#result=train(X_train=X[0],y_train=Y[0],X_test=X[1],y_test=Y[1],X_val=X[2],y_val=Y[2])

result=createConv2(X[0],Y[0],X[1],Y[1])#set[0], set[1],set[4],set[5])

#result=createConv(set[0], set[1],set[4],set[5])



#print ': ',result#predict(set[4])
#print ':: ',set[5]
#print '>>> ',compare(result,set[5])

#polaryzacja
# [syno+, syno-, hipo+, hipo-, hiper+, hiper-, anto+, anto-]

#dobry [ , , , , , ]  1
#zly [ 0, 1, 0, 4, 2,0] 0
#madry [ , , , , , ]
#glupi [ , , , , , ]
#glina [ , , , , , ]
#pies [ , , , , , ]
#policjant [ , , , , , ]


#a=
#createNeural(a,l)
#predict(d)