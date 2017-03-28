import numpy
import sklearn

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
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import keras

import os,sys
from PIL import Image
from os.path import isfile, join
from os import listdir

path="/home/olusiak/Obrazy/densities/"
image_list=list()
output_list=list()
files = [f for f in listdir(path) if isfile(join(path, f))]
ran=2
size_orig = 2048/8,1536/8
size = size_orig[0] / ran, size_orig[1] / ran
PIXELS=size[0]*size[1]/4
print "> ",size[0]*size[1]*3

def load_files():
    #print len(files)
    c=0
    files.sort()
    print files
    for filename in files:
        c+=1
        if 'dens_map' in filename:
            continue
        print filename, c
        #if c==60:
            #break
        im = Image.open("{0}{1}".format(path, filename))  # "/home/olusiak/Obrazy/schr.png")


        #print size
        im.thumbnail(size_orig, Image.ANTIALIAS)
        pix = im.load()

        for i in range(ran):
            for j in range(ran):#i, ran, 1):
                all_pixels = list()#np.array(int)
                #print '>', size[0] * i, ' ', size[0] * (i + 1),  ' / ', size[1] * j, ' ', size[1] * (j + 1)
                for x in range(size[0] * i, size[0] * (i + 1), 1):
                    #for j in range(ran):  # i, ran, 1):
                    if True:

                        for y in range(size[1] * j, size[1] * (j + 1), 1):
                            cpixel = pix[x, y]
                            if len(cpixel)>3:
                                cpixel=(cpixel[0],cpixel[1],cpixel[2])
                            if len(cpixel) > 3:
                                print ":: ", len(cpixel)
                            all_pixels.append(cpixel)
                            #all_pixels=np.asarray(all_pixels)
                            #print 'Al ',all_pixels.shape
                            # print 'L: ',len(all_pixels)


                #print 'ALL ',len(all_pixels)

                #all_pixels=np.asarray(all_pixels)
                #print 'o: ',all_pixels[0]
                all = list()
                a1 = list()
                a2 = list()
                a3 = list()
                for a in all_pixels:
                    a1.append(a[0])
                    a2.append(a[1])
                    a3.append(a[2])
                a1 = np.asarray(a1)
                a1 = np.reshape(a1, (size[0], size[1]))
                a2 = np.asarray(a2)
                a2 = np.reshape(a2, (size[0], size[1]))
                a3 = np.asarray(a3)
                a3 = np.reshape(a3, (size[0], size[1]))
                all.append(a3)
                #all.append(a2)
                #all.append(a3)

                all = np.asarray(all)


                #plt.matshow(a3)  # ll_pixels)
                #plt.show()
                if 'density' in filename:

                    all_pixels=[item for sublist in a3 for item in sublist]#in all_pixels
                    output_list.append(all_pixels)
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

    X_train=X_train[:50000]
    y_train = y_train[:50000]
    y_train2=list()
    y_test2=list()
    for y in y_train:
        y_train2.append((y,2))
    for y in y_test:
        y_test2.append((y,2))

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


    y_train=labels
    X_test=data
    y_test=results
    X_train=np.asarray(X_train)
    print 'sh ',X_train.shape[1:]
    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                ('maxpool1', layers.MaxPool2DLayer),
                ('conv2d2', layers.Conv2DLayer),
                ('maxpool2', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
                ('dense', layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ('output', layers.DenseLayer),
                ],
        # input layer
        input_shape=(None,1, size[0],size[1]),
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
        dense_num_units=256,
        dense_nonlinearity=lasagne.nonlinearities.rectify,
        # dropout2
        dropout2_p=0.5,
        # output
        output_nonlinearity=lasagne.nonlinearities.rectify,#softmax,
        output_num_units=size[0]*size[1],#*3,
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        max_epochs=10,
        verbose=1,
        regression=True
        )
    # Train the network
    print 'yt ',len(y_train[0])
    nn = net1.fit(X_train, y_train)
    preds = net1.predict(X_test)

    print preds
    print preds.shape

    #cm = confusion_matrix(y_test, preds)
    plt.matshow(preds)
    #plt.title('Confusion matrix')
    #plt.colorbar()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    plt.show()

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
set=load_dataset()

#print 'so ',set[0]
#print 'l so ',len(set[0][0])
#print 's1 ',len(set[1])
#print 'len ', len(set[0]), ' ',len(set[1])
#result=createNeural(set[0], set[1],set[4])
#print 'im ',len(image_list)
#Y=range(len(image_list))#
X = [image_list[:75],image_list[75:]]#int(len(image_list) * .25) : int(len(image_list) * .75)]

print 'im ',image_list[0]
print 'len im ',len(image_list[0])
#for x in X[0]:
#    print 'x:',len(x)

#for x in X[1]:
#    print 'x1:',len(x)
Y = [output_list[:75],output_list[75:]]#int(len(image_list) * .25) : int(len(image_list) * .75)]

print 'j ',len(Y[0][0])
print 'out ',output_list[0][0] #"[Y] ",len(Y[0])," . 147456"

#Y=np.asarray(Y)

#Y = output_list[int(len(output_list) * .25) : int(len(output_list) * .75)]

#Y1 = range(75) #Y[:int(len(Y) * .25)]
#Y2= range(225) #Y[int(len(Y) * .25) : int(len(Y) * .75)]#
#print 'y ',len(Y1),'...',len(Y2), ' ',len(X[0]), ' ',len(X[1])#len(Y[0])
result=createConv(X[1],Y[1],X[0],Y[0])#set[0], set[1],set[4],set[5])

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